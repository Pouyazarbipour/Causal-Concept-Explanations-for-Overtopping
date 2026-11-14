"""Counterfactual data-selection policies, version 2 (fixed and policy-specific tuned hyper-parameters).

Usage: python -m experiments.run_policies_v2 [--mode fixed tuned] [--evaluate-only]
Writes revision/outputs/policies/policy_v2_<mode>.npz, ..._info.csv and revision/tables/P*.csv.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.counterfactual_policies import POLICIES, policy_fold_v2
from src.pipeline import groups_for, load_model_data
from src.statistical_analysis import cluster_bootstrap
from src.utils import ROOT, get_logger

log = get_logger("policies_v2")
OUT = ROOT / "revision" / "outputs" / "policies"
TAB = ROOT / "revision" / "tables"
CACHE = OUT / "cache"


def job(mode, k, D, fold, groups):
    f = CACHE / f"{mode}__fold{k}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    E = D["cfg"]["experiment"]
    best = pickle.load(open(ROOT / "results/models/fold_cache" / f"geometry__xgb__fold{k}.pkl", "rb"))["best"]
    df = D["df"]
    r = policy_fold_v2(best, D["X"], D["y"], df["RF"].to_numpy(int), df["CF"].to_numpy(int), df["flag_remark_do_not_use"].to_numpy(bool),
                       groups, np.flatnonzero(fold != k), np.flatnonzero(fold == k), E["seed"] + k, E["uncertainty"]["calib_fraction"],
                       E["uncertainty"]["alpha"], tuned=(mode == "tuned"))
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(r, open(f, "wb"))
    return r


def run(mode, D):
    fold, groups = D["folds"]["geometry"].to_numpy(), groups_for(D["df"], "geometry")
    n = len(D["y"])
    res = Parallel(n_jobs=4 if mode == "fixed" else 5)(delayed(job)(mode, int(k), D, fold, groups) for k in np.unique(fold))
    arr = {f"{p}__{a}": np.full(n, np.nan) for p in POLICIES for a in ("pred", "lo", "hi")}
    info = []
    for k, (o, vm) in zip(np.unique(fold), res):
        te = np.flatnonzero(fold == k)
        for p in POLICIES:
            for a in ("pred", "lo", "hi"):
                arr[f"{p}__{a}"][te] = o[p][a]
            info.append({"fold": int(k), "policy": p, **{c: o[p][c] for c in ("n_train", "ess", "w_min", "w_max", "w_cv", "inner_rmse")},
                         "n_trees": o[p]["params"].get("n_estimators"), "max_depth": o[p]["params"].get("max_depth"),
                         "learning_rate": o[p]["params"].get("learning_rate"), "shrinkage_tau2": vm["tau2"]})
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"policy_v2_{mode}.npz", **arr)
    pd.DataFrame(info).to_csv(OUT / f"policy_v2_{mode}_info.csv", index=False)
    log.info("%s done", mode)


def winkler(y, lo, hi, alpha):
    w = hi - lo
    return w + (2 / alpha) * (lo - y) * (y < lo) + (2 / alpha) * (y - hi) * (y > hi)


def evaluate(D, modes, nb=500, seed=20261112, alpha=0.1):
    df, y = D["df"], D["y"]
    rf, cf = df["RF"].to_numpy(), df["CF"].to_numpy()
    cl = df["geometry"].to_numpy()
    subsets = {"all": np.ones(len(y), bool), "common": (rf < 4) & (cf < 4), "RF4": rf == 4, "CF4": cf == 4, "CF3": cf == 3}
    rows, cov_rows = [], []
    for mode in modes:
        f = OUT / f"policy_v2_{mode}.npz"
        if not f.exists():
            continue
        Z = np.load(f)
        info = pd.read_csv(OUT / f"policy_v2_{mode}_info.csv")
        base = Z["P1_retain_all__pred"]
        for p in POLICIES:
            pred, lo, hi = Z[f"{p}__pred"], Z[f"{p}__lo"], Z[f"{p}__hi"]
            assert np.isfinite(pred).all() and len(pred) == len(base)   # identical held-out records for every policy
            ii = info[info.policy == p]
            row = {"mode": mode, "policy": p, "n_train_mean_per_fold": ii.n_train.mean(), "ess_mean_per_fold": ii.ess.mean(),
                   "w_min": ii.w_min.min(), "w_max": ii.w_max.max(), "w_cv_mean": ii.w_cv.mean(), "inner_rmse_mean": ii.inner_rmse.mean()}
            cov = ((y >= lo) & (y <= hi)).astype(float)
            wk = winkler(y, lo, hi, alpha)
            for s, mk in subsets.items():
                i0 = np.flatnonzero(mk)
                e1, e0 = (pred[i0] - y[i0]) ** 2, (base[i0] - y[i0]) ** 2

                def st(i):
                    return [np.sqrt(e1[i].mean()), np.sqrt(e1[i].mean()) - np.sqrt(e0[i].mean())]
                pt, l_, h_, _ = cluster_bootstrap(st, cl[i0], nb, seed)
                row.update({f"rmse_{s}": pt[0], f"d_rmse_{s}": pt[1], f"d_rmse_{s}_lo": l_[1], f"d_rmse_{s}_hi": h_[1],
                            f"mae_{s}": float(np.abs(pred[i0] - y[i0]).mean()), f"bias_{s}": float((pred[i0] - y[i0]).mean()),
                            f"r2_{s}": float(1 - e1.sum() / ((y[i0] - y[i0].mean()) ** 2).sum()),
                            f"coverage_{s}": float(cov[i0].mean()), f"mean_width_{s}": float((hi - lo)[i0].mean()),
                            f"median_width_{s}": float(np.median((hi - lo)[i0])), f"winkler_{s}": float(wk[i0].mean())})
            gm = pd.Series(pred - y).groupby(cl).mean()
            gn = pd.Series(cl).value_counts().reindex(gm.index)
            row["geometry_level_abs_bias"] = float(gm[gn >= 2].abs().mean())
            d = pred - base
            c4 = (rf == 4) | (cf == 4)
            row["pct_changed_factor2_common"] = float(100 * np.mean(np.abs(d[subsets["common"]]) > np.log10(2)))
            row["pct_changed_factor2_category4"] = float(100 * np.mean(np.abs(d[c4]) > np.log10(2)))
            rows.append(row)
            for fac, arr in (("RF", rf), ("CF", cf)):
                for v in (1, 2, 3, 4):
                    mk = arr == v
                    pt, l_, h_, _ = cluster_bootstrap(lambda i: [cov[mk][i].mean()], cl[mk], 300, seed)
                    cov_rows.append({"mode": mode, "policy": p, "subset": f"{fac}={v}", "n": int(mk.sum()), "coverage": pt[0], "lo": l_[0], "hi": h_[0],
                                     "mean_width": float((hi - lo)[mk].mean()), "winkler": float(wk[mk].mean())})
    TAB.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TAB / "P1_policies_v2.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(TAB / "P2_policies_v2_coverage_by_category.csv", index=False)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="*", default=["fixed", "tuned"])
    ap.add_argument("--evaluate-only", action="store_true")
    a = ap.parse_args()
    D = load_model_data()
    if not a.evaluate_only:
        for m in a.mode:
            run(m, D)
    t = evaluate(D, ["fixed", "tuned"])
    cols = ["mode", "policy", "n_train_mean_per_fold", "ess_mean_per_fold", "d_rmse_common", "d_rmse_RF4", "d_rmse_CF4", "coverage_RF4", "coverage_CF3", "mean_width_all"]
    print(t[cols].to_string())


if __name__ == "__main__":
    main()
