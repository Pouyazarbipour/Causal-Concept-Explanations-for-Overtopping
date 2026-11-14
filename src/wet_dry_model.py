"""Phase IV (revision): two-stage wet/dry + positive-discharge model, fully nested and geometry-grouped.

Original wet/dry model (experiments/run_models.wet_dry) is kept: fixed XGBoost hyper-parameters, its own
GroupKFold over geometries (not the regression folds), probability only, AUC/Brier.  This module replaces
none of it; it adds a complete two-stage framework:

Folds.  Every wet/dry record (positive or zero discharge, usable inputs; n = 10 389) receives the outer fold
of its reconstructed geometry in the regression folds; geometries without any positive-discharge record
(zero-only geometries) are assigned to folds by a seeded permutation.  Positive records therefore keep
exactly their original regression fold.

Stage A  P(wet | X): XGBoost classifier, Optuna (TPE, 25 trials) on grouped inner folds of the outer-training
         partition (log loss, early stopping).  The decision threshold is chosen on grouped inner out-of-fold
         probabilities (maximum balanced accuracy); outer test rows are never used.
Stage B  E[log10 q* | wet, X]: the original per-fold XGBoost hyper-parameters (tuned on positive training rows);
         refitted here and checked against the original out-of-fold predictions.
Stage C  E[q | X] = P(wet | X) * E[q | wet, X], with E[q | wet, X] = 10^mu * S, S = Duan smearing factor from
         grouped inner out-of-fold residuals of the training positives; q = q* sqrt(g Hm0^3).
Policies P1-P4 are also applied to both stages (fixed hyper-parameters) to see whether exclusion changes the
prediction of occurrence or of total expected discharge.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

from .baseline_models import FoldPrep
from .features import features_from_raw
from .geometry_grouping import check_no_leakage, inner_group_folds
from .pipeline import load_model_data, load_pred
from .statistical_analysis import cluster_bootstrap
from .utils import ROOT

OUT = ROOT / "revision" / "outputs" / "wet_dry"
TAB = ROOT / "revision" / "tables"
G = 9.81


def wetdry_data(D):
    m = D["master"]
    wd = m[m.use_wetdry].reset_index(drop=True)
    X = features_from_raw(wd, D["features"], D["cfg"]["data"]["log1p_features"], D["cfg"]["data"]["gravity"])
    wet = (wd.response_class == "positive").to_numpy().astype(int)
    # folds: geometry fold of the regression set; zero-only geometries by seeded permutation
    reg = D["df"][["geometry"]].assign(fold=D["folds"]["geometry"].to_numpy()).drop_duplicates("geometry").set_index("geometry")["fold"]
    fold = wd.geometry.map(reg)
    extra = np.array(sorted(wd.geometry[fold.isna()].unique()))
    perm = np.random.RandomState(D["cfg"]["experiment"]["seed"]).permutation(len(extra)) % 5
    fold = fold.fillna(wd.geometry.map(dict(zip(extra, perm)))).astype(int).to_numpy()
    return wd, X, wet, fold


def _clf(params, seed, n=None):
    import xgboost as xgb
    kw = dict(params)
    n = kw.pop("n_estimators", n or 2000)
    return xgb.XGBClassifier(n_estimators=n, n_jobs=1, tree_method="hist", random_state=seed, eval_metric="logloss", **kw)


def suggest_clf(trial):
    return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True)}


def tune_clf(X, y, g, seed, n_trials=25):
    import optuna
    from sklearn.metrics import log_loss
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = inner_group_folds(g, 3, seed)

    def obj(trial):
        p = suggest_clf(trial)
        ll, trees = [], []
        for a, b in folds:
            prep = FoldPrep(X[a])
            m = _clf(p, seed)
            m.set_params(early_stopping_rounds=50)
            m.fit(prep.impute(X[a]), y[a], eval_set=[(prep.impute(X[b]), y[b])], verbose=False)
            ll.append(log_loss(y[b], m.predict_proba(prep.impute(X[b]))[:, 1], labels=[0, 1]) * len(b))
            trees.append(m.best_iteration + 1)
        trial.set_user_attr("n_trees", int(np.ceil(1.1 * np.mean(trees))))
        return float(np.sum(ll) / len(y))
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(obj, n_trials=n_trials)
    best = suggest_clf(optuna.trial.FixedTrial(st.best_params))
    best["n_estimators"] = st.best_trial.user_attrs["n_trees"]
    # inner OOF probabilities with the best configuration -> threshold (max balanced accuracy)
    oof = np.zeros(len(y))
    for a, b in folds:
        prep = FoldPrep(X[a])
        oof[b] = _clf(best, seed).fit(prep.impute(X[a]), y[a]).predict_proba(prep.impute(X[b]))[:, 1]
    ths = np.linspace(0.05, 0.95, 181)
    ba = [0.5 * (((oof >= t) & (y == 1)).sum() / max((y == 1).sum(), 1) + ((oof < t) & (y == 0)).sum() / max((y == 0).sum(), 1)) for t in ths]
    return best, float(ths[int(np.argmax(ba))]), float(st.best_value)


def fold_job(k, X, wet, fold, groups, wd, D, reg_best, seed):
    import xgboost as xgb
    f = OUT / "cache" / f"fold{k}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
    check_no_leakage(groups[tr], groups[te])
    best, thr, inner_ll = tune_clf(X[tr], wet[tr], groups[tr], seed + k)
    prep = FoldPrep(X[tr])
    p_wet = _clf(best, seed).fit(prep.impute(X[tr]), wet[tr]).predict_proba(prep.impute(X[te]))[:, 1]
    # stage B on positive training rows (original tuned hyper-parameters of this outer fold)
    pos = tr[wet[tr] == 1]
    prep_b = FoldPrep(X[pos])
    yb = wd.y.to_numpy()[pos]
    mdl = xgb.XGBRegressor(n_jobs=1, tree_method="hist", random_state=seed, **reg_best).fit(prep_b.impute(X[pos]), yb)
    mu = mdl.predict(prep_b.impute(X[te]))
    # Duan smearing from grouped inner OOF residuals of the training positives
    oof = np.zeros(len(pos))
    for a, b in inner_group_folds(groups[pos], 3, seed):
        pp = FoldPrep(X[pos][a])
        oof[b] = xgb.XGBRegressor(n_jobs=1, tree_method="hist", random_state=seed, **reg_best).fit(pp.impute(X[pos][a]), yb[a]).predict(pp.impute(X[pos][b]))
    smear = float(np.mean(10 ** (yb - oof)))
    # policies (fixed hyper-parameters): exclusion in BOTH stages
    rf, cf = wd.RF.to_numpy(int), wd.CF.to_numpy(int)
    pol = {}
    for name, keep in (("P2_excl_RF4", rf != 4), ("P3_excl_CF4", cf != 4), ("P4_excl_RF4_or_CF4", (rf != 4) & (cf != 4))):
        trp = tr[keep[tr]]
        pp = FoldPrep(X[trp])
        pw = _clf(best, seed).fit(pp.impute(X[trp]), wet[trp]).predict_proba(pp.impute(X[te]))[:, 1]
        posp = trp[wet[trp] == 1]
        pb = FoldPrep(X[posp])
        mup = xgb.XGBRegressor(n_jobs=1, tree_method="hist", random_state=seed, **reg_best).fit(pb.impute(X[posp]), wd.y.to_numpy()[posp]).predict(pb.impute(X[te]))
        pol[name] = (pw, mup)
    res = {"te": te, "p_wet": p_wet, "mu": mu, "smear": smear, "thr": thr, "best": best, "inner_logloss": inner_ll, "pol": pol}
    (OUT / "cache").mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f, "wb"))
    return res


def run():
    from joblib import Parallel, delayed
    D = load_model_data()
    wd, X, wet, fold = wetdry_data(D)
    groups = wd.geometry.to_numpy()
    seed = D["cfg"]["experiment"]["seed"]
    reg_best = {k: pickle.load(open(ROOT / "results/models/fold_cache" / f"geometry__xgb__fold{k}.pkl", "rb"))["best"] for k in range(5)}
    res = Parallel(n_jobs=5)(delayed(fold_job)(k, X, wet, fold, groups, wd, D, reg_best[k], seed) for k in range(5))
    n = len(wd)
    out = pd.DataFrame({"name": wd.name, "RF": wd.RF.astype(int), "CF": wd.CF.astype(int), "series": wd.series, "geometry": wd.geometry,
                        "family": wd.family, "wet": wet, "fold": fold, "hm0_toe": wd.hm0_toe, "q": wd.q, "y": wd.y,
                        "flag_depth_limited_gt1": wd.flag_depth_limited_gt1})
    for c in ("p_wet", "mu", "smear", "thr"):
        out[c] = np.nan
    for c in ("P2_excl_RF4", "P3_excl_CF4", "P4_excl_RF4_or_CF4"):
        out[f"{c}__p_wet"], out[f"{c}__mu"] = np.nan, np.nan
    for r in res:
        te = r["te"]
        out.loc[te, "p_wet"], out.loc[te, "mu"] = r["p_wet"], r["mu"]
        out.loc[te, "smear"], out.loc[te, "thr"] = r["smear"], r["thr"]
        for c, (pw, mu) in r["pol"].items():
            out.loc[te, f"{c}__p_wet"], out.loc[te, f"{c}__mu"] = pw, mu
    scale = np.sqrt(G * out.hm0_toe ** 3)
    out["Eq_wet"] = 10 ** out.mu * out.smear * scale
    out["Eq_total"] = out.p_wet * out.Eq_wet
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "two_stage_oof.csv", index=False)
    pd.DataFrame([{"fold": k, "threshold": r["thr"], "smearing": r["smear"], "inner_logloss": r["inner_logloss"], **r["best"]} for k, r in enumerate(res)]).to_csv(
        OUT / "stageA_best_per_fold.csv", index=False)
    # reproduction check: stage B on positives equals the original OOF predictions?
    orig = pd.Series(load_pred("geometry__xgb")[0], index=D["df"].name)
    pos = out[out.wet == 1]
    diff = np.abs(pos.mu.to_numpy() - orig.reindex(pos.name).to_numpy())
    pd.DataFrame([{"check": "stage B OOF prediction vs original XGBoost OOF (positives)", "n": len(pos), "max_abs_diff": float(diff.max()),
                   "mean_abs_diff": float(diff.mean())}]).to_csv(OUT / "stageB_reproduction_check.csv", index=False)
    return out


def _cls_metrics(y, p, thr):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    pred = (p >= thr).astype(int)
    tp, tn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
    fp, fn = int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1) if (tn + fp) else np.nan
    two = len(np.unique(y)) > 1
    return {"n": len(y), "n_zero": int((y == 0).sum()), "auc": roc_auc_score(y, p) if two else np.nan,
            "pr_auc_dry": average_precision_score(1 - y, 1 - p) if two else np.nan,
            "brier": brier_score_loss(y, p), "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1]),
            "sensitivity_wet": sens, "specificity_dry": spec, "balanced_accuracy": 0.5 * (sens + spec) if two else np.nan,
            "TP": tp, "TN": tn, "FP_false_wet": fp, "FN_false_dry": fn}


def evaluate(out=None, nb=500, seed=20261112):
    from sklearn.metrics import roc_auc_score
    if out is None:
        out = pd.read_csv(OUT / "two_stage_oof.csv")
    y, p, thr = out.wet.to_numpy(), out.p_wet.to_numpy(), out.thr.to_numpy()
    cl = out.geometry.to_numpy()
    rows = []
    for lab, mk in [("all", np.ones(len(out), bool))] + [(f"RF={v}", (out.RF == v).to_numpy()) for v in (1, 2, 3, 4)] + \
                   [(f"CF={v}", (out.CF == v).to_numpy()) for v in (1, 2, 3, 4)]:
        r = {"subset": lab, **_cls_metrics(y[mk], p[mk], thr[mk])}
        if len(np.unique(y[mk])) > 1:
            yy, pp = y[mk], p[mk]
            pt, lo, hi, _ = cluster_bootstrap(lambda i: [roc_auc_score(yy[i], pp[i])], cl[mk], nb, seed)
            r["auc_lo"], r["auc_hi"] = lo[0], hi[0]
        r["threshold_mean"] = float(thr[mk].mean())
        rows.append(r)
    A = pd.DataFrame(rows)
    A.to_csv(TAB / "W1_stageA_classification.csv", index=False)
    # calibration (reliability) table
    bins = np.linspace(0, 1, 11)
    b = np.clip(np.digitize(p, bins) - 1, 0, 9)
    rel = pd.DataFrame({"bin": b, "p": p, "y": y}).groupby("bin").agg(n=("y", "size"), mean_pred=("p", "mean"), obs_wet=("y", "mean")).reset_index()
    rel["ece_contrib"] = rel.n / len(p) * np.abs(rel.mean_pred - rel.obs_wet)
    rel.to_csv(TAB / "W2_stageA_reliability.csv", index=False)
    # stage C: combined prediction
    pos = out.wet == 1
    lq = np.log10(out.q.where(pos))
    rows = []
    for lab, mk in [("all", np.ones(len(out), bool))] + [(f"RF={v}", (out.RF == v).to_numpy()) for v in (1, 2, 3, 4)] + \
                   [(f"CF={v}", (out.CF == v).to_numpy()) for v in (1, 2, 3, 4)]:
        mp, mz = mk & pos.to_numpy(), mk & ~pos.to_numpy()
        e_reg = np.log10(out.Eq_wet[mp] / out.smear[mp]) - lq[mp]          # regression-only median prediction
        e_two = np.log10(out.Eq_total[mp]) - lq[mp]
        e_wet = np.log10(out.Eq_wet[mp]) - lq[mp]
        rows.append({"subset": lab, "n_positive": int(mp.sum()), "n_zero": int(mz.sum()),
                     "rmse_log10q_regression_only": float(np.sqrt(np.mean(e_reg ** 2))),
                     "rmse_log10q_conditional_mean_smeared": float(np.sqrt(np.mean(e_wet ** 2))),
                     "rmse_log10q_two_stage_total": float(np.sqrt(np.mean(e_two ** 2))),
                     "bias_log10q_two_stage_total": float(np.mean(e_two)),
                     "zero_tests_median_pred_q_l_per_s_per_m_regression_only": float(np.median(out.Eq_wet[mz] / out.smear[mz]) * 1000) if mz.sum() else np.nan,
                     "zero_tests_median_pred_q_l_per_s_per_m_two_stage": float(np.median(out.Eq_total[mz]) * 1000) if mz.sum() else np.nan,
                     "zero_tests_pct_pred_above_0.1_l_s_m_regression_only": float(100 * np.mean(out.Eq_wet[mz] / out.smear[mz] > 1e-4)) if mz.sum() else np.nan,
                     "zero_tests_pct_pred_above_0.1_l_s_m_two_stage": float(100 * np.mean(out.Eq_total[mz] > 1e-4)) if mz.sum() else np.nan})
    C = pd.DataFrame(rows)
    C.to_csv(TAB / "W3_two_stage_combined.csv", index=False)
    # policies: does exclusion change occurrence prediction or total expected discharge?
    rows = []
    rf, cf = out.RF.to_numpy(), out.CF.to_numpy()
    sub = {"common": (rf < 4) & (cf < 4), "RF4": rf == 4, "CF4": cf == 4}
    scale = np.sqrt(G * out.hm0_toe ** 3)
    for pol in ("P1_retain_all", "P2_excl_RF4", "P3_excl_CF4", "P4_excl_RF4_or_CF4"):
        pw = out.p_wet if pol == "P1_retain_all" else out[f"{pol}__p_wet"]
        mu = out.mu if pol == "P1_retain_all" else out[f"{pol}__mu"]
        Eq = pw * 10 ** mu * out.smear * scale
        for s, mk in sub.items():
            r = {"policy": pol, "subset": s, **{k: v for k, v in _cls_metrics(y[mk], pw.to_numpy()[mk], thr[mk]).items() if k in ("n", "n_zero", "auc", "brier", "balanced_accuracy")}}
            mp = mk & pos.to_numpy()
            r["rmse_log10q_two_stage_positives"] = float(np.sqrt(np.mean((np.log10(Eq[mp]) - lq[mp]) ** 2)))
            rows.append(r)
    Pt = pd.DataFrame(rows)
    Pt.to_csv(TAB / "W4_two_stage_policies.csv", index=False)
    pd.read_csv(OUT / "stageB_reproduction_check.csv").to_csv(TAB / "W5_stageB_reproduction_check.csv", index=False)
    return A, C, Pt


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    out = run()
    A, C, Pt = evaluate(out)
    print(A.to_string()); print(C.to_string()); print(Pt.to_string())


if __name__ == "__main__":
    main()
