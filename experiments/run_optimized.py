"""Revised (v2) predictive-model experiment.

Usage: python -m experiments.run_optimized [--learners ...] [--partitions]

* Same pre-specified outer geometry folds and all 9 333 records as the original study.
* Budgets and spaces in src/model_optimization.py (configs/optimized_models.yaml documents them).
* `--partitions`: repeated grouped CV, i.e. the baseline XGBoost (original code, 30 trials) on two further
  geometry partitions drawn with other seeds, the optimised XGBoost (50 trials) on the first of them and on
  the series-held-out folds.  These are sensitivity analyses; the pre-specified partition
  remains the primary result (no partition is chosen by its test error).
Writes revision/outputs/models/<tag>.npz (pred [seeds, n]) and <tag>__trials.csv / __best.csv / __budget.csv.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src import model_optimization as MO
from src.baseline_models import outer_fold as outer_fold_v1
from src.geometry_grouping import GroupKFold
from src.pipeline import groups_for, load_model_data
from src.utils import ROOT, get_logger

log = get_logger("optimized")
OUT = ROOT / "revision" / "outputs" / "models"
CACHE = ROOT / "revision" / "outputs" / "fold_cache"
SEEDS = [0, 1, 2]
BUDGETS = (10, 20, 30, 50)


def alt_partition(groups, seed, n_folds=5):
    uniq = np.unique(groups)
    perm = dict(zip(uniq, np.random.RandomState(seed).permutation(len(uniq))))
    gi = np.array([perm[v] for v in groups])
    f = np.full(len(groups), -1)
    for k, (_, te) in enumerate(GroupKFold(n_folds).split(gi, groups=gi)):
        f[te] = k
    return f


def run(name, tag, X, y, fold, groups, D, n_trials, version=2):
    E = D["cfg"]["experiment"]
    mcfg = D["cfg"]["model"]["mlp"]
    t0 = time.time()
    if version == 2:
        res = Parallel(n_jobs=4)(delayed(MO.outer_fold)(name, int(k), np.flatnonzero(fold != k), np.flatnonzero(fold == k), X, y, groups,
                                                        SEEDS, n_trials, E["n_inner_folds"], E["seed"], mcfg, CACHE / tag,
                                                        BUDGETS if name.startswith("xgb") else ()) for k in np.unique(fold))
    else:
        res = Parallel(n_jobs=4)(delayed(outer_fold_v1)(name, tag, int(k), np.flatnonzero(fold != k), np.flatnonzero(fold == k), X, y, groups,
                                                        [0], n_trials, E["n_inner_folds"], E["seed"], mcfg, CACHE / tag) for k in np.unique(fold))
    P = np.full((len(res[0]["pred"]), len(y)), np.nan, dtype=np.float32)
    best, hist, bud = [], [], []
    for r in res:
        P[:, r["te"]] = r["pred"]
        best.append({"fold": r["fold"], "inner_cv_rmse": r["inner_rmse"], "n_trials": len(r["trials"]), **r["best"]})
        hist.append(r["trials"].assign(fold=r["fold"]))
        for b, (v, p) in r.get("budget_pred", {}).items():
            bud.append(pd.DataFrame({"budget": b, "fold": r["fold"], "inner_rmse": v, "idx": r["te"], "pred": p}))
    assert np.isfinite(P).all()
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{tag}.npz", pred=P)
    pd.DataFrame(best).to_csv(OUT / f"{tag}__best.csv", index=False)
    pd.concat(hist).to_csv(OUT / f"{tag}__trials.csv", index=False)
    if bud:
        pd.concat(bud).to_csv(OUT / f"{tag}__budget.csv", index=False)
    log.info("%s: per-seed RMSE %s, ensemble %.4f (%.0fs)", tag,
             np.round(np.sqrt(((P - y) ** 2).mean(1)), 4), float(np.sqrt(((P.mean(0) - y) ** 2).mean())), time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--learners", nargs="*", default=["xgb", "xgb_phys", "lgbm", "hgb", "et", "catboost", "rf", "mlp"])
    ap.add_argument("--partitions", action="store_true")
    a = ap.parse_args()
    D = load_model_data()
    X, y, df = D["X"], D["y"], D["df"]
    Xp = np.column_stack([X, MO.phys_features(df)])
    fold, groups = D["folds"]["geometry"].to_numpy(), groups_for(df, "geometry")
    for name in a.learners:
        if (OUT / f"geometry__{name}.npz").exists():
            continue
        run(name, f"geometry__{name}", Xp if name == "xgb_phys" else X, y, fold, groups, D, MO.BUDGET[name])
    if a.partitions:
        seed = D["cfg"]["experiment"]["seed"]
        # budget-limited for runtime: baseline XGBoost (30 trials) on two further partitions; v2 XGBoost with 50 trials
        # on the first further partition and on the series folds (on the primary partition 50 trials give 0.498 vs 0.493 with 100, M3)
        for s, vers in ((1, ((1, 30), (2, 50))), (2, ((1, 30),))):
            f2 = alt_partition(groups, seed + 1000 * s)
            for ver, nt in vers:
                tag = f"geometry_alt{s}__xgb_v{ver}"
                if not (OUT / f"{tag}.npz").exists():
                    run("xgb", tag, X, y, f2, groups, D, nt, version=ver)
        fs, gs = D["folds"]["series"].to_numpy(), groups_for(df, "series")
        if not (OUT / "series__xgb.npz").exists():
            run("xgb", "series__xgb", X, y, fs, gs, D, 50)
    log.info("done")


if __name__ == "__main__":
    main()
