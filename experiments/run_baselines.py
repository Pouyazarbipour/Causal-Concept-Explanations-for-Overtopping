"""Stage D: nested, leakage-safe CV of all baseline learners on every validation scheme.

Usage: python -m experiments.run_baselines [--schemes geometry series ...] [--models rf xgb xgb_mono mlp]
Also runs (primary scheme): the explicitly SECONDARY ablation with RF/CF as mean-model inputs,
the category-4 filtering sensitivity (Experiment 10) and the wet/dry (zero-overtopping) model.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.baseline_models import FoldPrep, run_cv
from src.counterfactual_audit import features_from_raw, support_flags
from src.geometry_grouping import check_no_leakage
from src.pipeline import groups_for, load_model_data
from src.utils import ROOT, get_logger, out

log = get_logger("baselines")


def monotone_vector(features, spec):
    return [int(spec.get(f, 0)) for f in features]


def main() -> None:
    ap = argparse.ArgumentParser()
    D = load_model_data()
    cfg = D["cfg"]
    E, M = cfg["experiment"], cfg["model"]
    ap.add_argument("--schemes", nargs="*", default=E["cv_schemes"])
    ap.add_argument("--models", nargs="*", default=["xgb", "xgb_mono", "rf", "mlp"])
    ap.add_argument("--extras", type=int, default=1)
    a = ap.parse_args()
    df, y, Xv = D["df"], D["y"], D["X_views"]
    mono = monotone_vector(D["features"], M["xgboost"]["monotone"])
    grids = {"rf": M["random_forest"]["grid"], "xgb": M["xgboost"]["grid"], "xgb_mono": M["xgboost"]["grid"], "mlp": M["mlp"]["grid"]}
    for scheme in a.schemes:
        fold, groups = D["folds"][scheme].to_numpy(), groups_for(df, scheme)
        # model-independent common-support diagnostic of every counterfactual view
        sup = np.zeros(Xv.shape[:2], bool)
        for k in np.unique(fold):
            sup[fold == k] = support_flags(Xv[fold != k, 0], Xv[fold == k], E["audit"]["support_k"], E["audit"]["support_quantile"])
        np.savez_compressed(out("diagnostics", f"support__{scheme}.npz"), support=sup)
        for name in a.models:
            if name == "mlp" and scheme not in E["neural_schemes"] + ["random"]:
                continue
            t0 = time.time()
            seeds = E["nn_seeds"] if name == "mlp" else (0,)
            P, _ = run_cv(name, grids[name], Xv, y, fold, groups, seeds=seeds, n_inner=E["n_inner_folds"], base_seed=E["seed"],
                          monotone=mono, mcfg=M["mlp"], tag=f"{scheme}__{name}")
            rmse = np.sqrt(np.nanmean((P[:, :, 0] - y) ** 2, axis=1))
            log.info("%s %s rmse=%s (%.0fs)", scheme, name, np.round(rmse, 4), time.time() - t0)
    if not a.extras:
        return
    scheme = E["primary_scheme"]
    fold, groups = D["folds"][scheme].to_numpy(), groups_for(df, scheme)
    rf, cf = df["RF"].to_numpy(), df["CF"].to_numpy()
    # ---- SECONDARY ablation: RF/CF as ordinary inputs of the mean model (not the primary model) ----
    extra = np.column_stack([rf, cf]).astype(float)
    Xq = np.concatenate([Xv, np.repeat(extra[:, None, :], Xv.shape[1], axis=1)], axis=2)
    run_cv("xgb", grids["xgb"], Xq, y, fold, groups, n_inner=E["n_inner_folds"], base_seed=E["seed"], tag=f"{scheme}__xgb_rfcf_secondary")
    # ---- Experiment 10: category-4 filtering and quality weighting (training rows only) ----------
    X0 = Xv[:, :1]
    variants = {"full": (None, None),
                "excl_rf4": (rf != 4, None), "excl_cf4": (cf != 4, None), "excl_rf4_or_cf4": ((rf != 4) & (cf != 4), None),
                "clash_weight_(4-RF)(4-CF)": ((rf != 4) & (cf != 4), (4 - rf) * (4 - cf)),
                "soft_weight_(5-RF)(5-CF)": (None, (5 - rf) * (5 - cf)),
                "excl_remark_do_not_use": (~df["flag_remark_do_not_use"].to_numpy(), None)}
    for vname, (mask, w) in variants.items():
        for model in ("xgb", "rf"):
            run_cv(model, grids[model], X0, y, fold, groups, n_inner=E["n_inner_folds"], base_seed=E["seed"], train_mask=mask,
                   weight=None if w is None else w.astype(float), tag=f"filter__{model}__{vname}")
        log.info("filter variant %s done", vname)
    # ---- wet/dry model on positive + zero records (two-part strategy, part 1) --------------------
    import xgboost as xgb
    from sklearn.model_selection import GroupKFold
    m = D["master"]
    wd = m[m["use_wetdry"]].reset_index(drop=True)
    Xw = features_from_raw(wd, D["features"], cfg["data"]["log1p_features"], cfg["data"]["gravity"])
    wet = (wd["response_class"] == "positive").to_numpy().astype(int)
    g = wd["geometry"].to_numpy()
    uniq = np.unique(g)
    perm = dict(zip(uniq, np.random.RandomState(E["seed"]).permutation(len(uniq))))
    gi = np.array([perm[v] for v in g])
    p = np.full(len(wd), np.nan)
    for tr, te in GroupKFold(E["n_outer_folds"]).split(Xw, groups=gi):
        check_no_leakage(g[tr], g[te])
        prep = FoldPrep(Xw[tr])
        clf = xgb.XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                n_jobs=4, random_state=E["seed"], tree_method="hist").fit(prep.impute(Xw[tr]), wet[tr])
        p[te] = clf.predict_proba(prep.impute(Xw[te]))[:, 1]
    wd[["name", "RF", "CF", "family", "geometry", "series", "response_class"]].assign(p_wet=p).to_csv(
        out("predictions", "wetdry__xgb_geometry.csv"), index=False)
    log.info("wet/dry model done")


if __name__ == "__main__":
    main()
