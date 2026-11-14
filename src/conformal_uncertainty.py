"""Phase IX (revision): validation and sensitivity of quality-aware split-conformal intervals.

Design per outer geometry fold (XGBoost, original per-fold hyper-parameters, fixed):
  outer-training geometries -> proper-training geometries + calibration geometries (grouped split, fraction c)
  mean model fitted on proper-training rows only;
  every scale function sigma(.) fitted on grouped inner out-of-fold residuals of the PROPER-TRAINING rows only
  (independent of calibration and test outcomes);
  calibration score |y - mu| / sigma, hierarchical (group-weighted) conformal quantile (Lee et al. 2023):
  each calibration geometry carries weight 1/(K+1), mass 1/(K+1) at +infinity.
Scales compared:
  constant                       no quality information (split conformal)
  RF+CF main effects             retrospective audit: both ratings known
  RF x CF shrunk cells           partial pooling of the 16 cells towards the main-effects model
  CF only                        prospective, known geometry with a CF assessment, RF unknown
  physical predictors only       prospective proxy: no rating available
  physical + RF + CF             both
Sensitivity: calibration fraction {0.15, 0.25, 0.35} x 5 calibration seeds.
Guarantee: marginal over a new geometry only; subgroup coverages are empirical, not guaranteed.
Outputs: revision/tables/C*.csv
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .baseline_models import FoldPrep, val_split
from .counterfactual_policies import variance_models
from .geometry_grouping import check_no_leakage, inner_group_folds
from .pipeline import load_model_data
from .statistical_analysis import cluster_bootstrap
from .uncertainty_modeling import group_quantile
from .utils import ROOT

TAB = ROOT / "revision" / "tables"
OUT = ROOT / "revision" / "outputs" / "conformal"
SCALES = ["constant", "RF+CF main effects", "RFxCF shrunk cells", "CF only", "physical only", "physical+RF+CF"]
ALPHA = 0.1


def _xgb(params, seed):
    import xgboost as xgb
    return xgb.XGBRegressor(n_jobs=1, tree_method="hist", random_state=seed, **params)


def fold_run(X, y, rf, cf, groups, tr, te, params, seed, frac):
    check_no_leakage(groups[tr], groups[te])
    is_fit, is_cal = val_split(groups[tr], seed, frac)
    fit, cal = tr[is_fit], tr[is_cal]
    check_no_leakage(groups[fit], groups[cal])
    prep = FoldPrep(X[fit])
    Xf, Xc, Xt = prep.impute(X[fit]), prep.impute(X[cal]), prep.impute(X[te])
    oof = np.zeros(len(fit))
    for a, b in inner_group_folds(groups[fit], 4, seed):
        oof[b] = _xgb(params, seed).fit(Xf[a], y[fit][a]).predict(Xf[b])
    r_in = y[fit] - oof
    mean = _xgb(params, seed).fit(Xf, y[fit])
    mc, mt = mean.predict(Xc), mean.predict(Xt)
    _, s2_main, s2_shr, _ = variance_models(r_in, rf[fit], cf[fit])
    # CF-only log-variance model
    Dc = lambda c_: np.column_stack([np.ones(len(c_))] + [(c_ == v).astype(float) for v in (2, 3, 4)])  # noqa: E731
    lr = np.log(r_in ** 2 + 1e-4)
    bc = np.linalg.lstsq(Dc(cf[fit]), lr, rcond=None)[0]
    corr_c = np.log(np.mean(np.exp(lr - Dc(cf[fit]) @ bc)))
    s2_cf = lambda c_: np.exp(Dc(c_) @ bc + corr_c)  # noqa: E731
    sp = dict(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=5)
    la = np.log(np.abs(r_in) + 1e-3)
    s_phys = _xgb(sp, seed).fit(Xf, la)
    Q = lambda idx: np.column_stack([(rf[idx] == v).astype(float) for v in (2, 3, 4)] + [(cf[idx] == v).astype(float) for v in (2, 3, 4)])  # noqa: E731
    s_both = _xgb(sp, seed).fit(np.hstack([Xf, Q(fit)]), la)
    sig = {"constant": (np.ones(len(cal)), np.ones(len(te))),
           "RF+CF main effects": (np.sqrt(s2_main(rf[cal], cf[cal])), np.sqrt(s2_main(rf[te], cf[te]))),
           "RFxCF shrunk cells": (np.sqrt(s2_shr(rf[cal], cf[cal])), np.sqrt(s2_shr(rf[te], cf[te]))),
           "CF only": (np.sqrt(s2_cf(cf[cal])), np.sqrt(s2_cf(cf[te]))),
           "physical only": (np.exp(s_phys.predict(Xc)), np.exp(s_phys.predict(Xt))),
           "physical+RF+CF": (np.exp(s_both.predict(np.hstack([Xc, Q(cal)]))), np.exp(s_both.predict(np.hstack([Xt, Q(te)]))))}
    rc = np.abs(y[cal] - mc)
    out = {}
    for k, (sc, st) in sig.items():
        q = group_quantile(rc / sc, groups[cal], 1 - ALPHA)
        out[k] = (mt - q * st, mt + q * st)
    return te, out


def run(D, fracs=(0.15, 0.25, 0.35), reps=5):
    df, X, y = D["df"], D["X"], D["y"]
    fold, groups = D["folds"]["geometry"].to_numpy(), df["geometry"].to_numpy()
    rf, cf = df["RF"].to_numpy(int), df["CF"].to_numpy(int)
    seed = D["cfg"]["experiment"]["seed"]
    best = {k: pickle.load(open(ROOT / "results/models/fold_cache" / f"geometry__xgb__fold{k}.pkl", "rb"))["best"] for k in range(5)}
    jobs = [(fr, rep, k) for fr in fracs for rep in range(reps) for k in range(5)]
    res = Parallel(n_jobs=4)(delayed(fold_run)(X, y, rf, cf, groups, np.flatnonzero(fold != k), np.flatnonzero(fold == k), best[k],
                                               seed + 101 * rep, fr) for fr, rep, k in jobs)
    iv = {(fr, rep): {s: [np.full(len(y), np.nan), np.full(len(y), np.nan)] for s in SCALES} for fr in fracs for rep in range(reps)}
    for (fr, rep, k), (te, o) in zip(jobs, res):
        for s in SCALES:
            iv[(fr, rep)][s][0][te], iv[(fr, rep)][s][1][te] = o[s]
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(iv, open(OUT / "intervals.pkl", "wb"))
    return iv


def winkler(y, lo, hi, a=ALPHA):
    return (hi - lo) + (2 / a) * (lo - y) * (y < lo) + (2 / a) * (y - hi) * (y > hi)


def evaluate(D, iv, nb=300, seed=20261112):
    df, y = D["df"], D["y"]
    rf, cf = df["RF"].to_numpy(int), df["CF"].to_numpy(int)
    geo, ser = df["geometry"].to_numpy(), df["series"].to_numpy()
    rows, cat_rows, cell_rows = [], [], []
    keys = sorted(iv)
    for fr in sorted({k[0] for k in keys}):
        for s in SCALES:
            covs = [((y >= iv[(fr, r)][s][0]) & (y <= iv[(fr, r)][s][1])).astype(float) for r in range(len([k for k in keys if k[0] == fr]))]
            widths = [iv[(fr, r)][s][1] - iv[(fr, r)][s][0] for r in range(len(covs))]
            wks = [winkler(y, iv[(fr, r)][s][0], iv[(fr, r)][s][1]) for r in range(len(covs))]
            cov, width, wk = np.mean(covs, 0), np.mean(widths, 0), np.mean(wks, 0)
            pt, lo, hi, _ = cluster_bootstrap(lambda i: [cov[i].mean()], geo, nb, seed)
            gcov = pd.Series(cov).groupby(geo).mean()
            scov = pd.Series(cov).groupby(ser).mean()
            rep_cov = [c.mean() for c in covs]
            row = {"calib_fraction": fr, "scale": s, "coverage": pt[0], "cov_lo": lo[0], "cov_hi": hi[0],
                   "coverage_sd_over_calibration_seeds": float(np.std(rep_cov)),
                   "mean_width": float(width.mean()), "median_width": float(np.median(width)),
                   "median_width_as_factor": float(10 ** (np.median(width) / 2)), "winkler": float(wk.mean()),
                   "pct_geometries_coverage_below_0.8": float(100 * (gcov < 0.8).mean()),
                   "pct_series_coverage_below_0.8": float(100 * (scov < 0.8).mean()), "worst_series_coverage": float(scov.min())}
            for fac, arr in (("RF", rf), ("CF", cf)):
                for v in (1, 2, 3, 4):
                    mk = arr == v
                    row[f"cov_{fac}{v}"] = float(cov[mk].mean())
                    if fr == 0.25:
                        p2, l2, h2, _ = cluster_bootstrap(lambda i: [cov[mk][i].mean()], geo[mk], nb, seed)
                        cat_rows.append({"scale": s, "subset": f"{fac}={v}", "n": int(mk.sum()), "coverage": p2[0], "lo": l2[0], "hi": h2[0],
                                         "mean_width": float(width[mk].mean()), "winkler": float(wk[mk].mean())})
            row["max_abs_dev_RF"] = max(abs(row[f"cov_RF{v}"] - 0.9) for v in (1, 2, 3, 4))
            row["max_abs_dev_CF"] = max(abs(row[f"cov_CF{v}"] - 0.9) for v in (1, 2, 3, 4))
            m3 = (cf == 3) & (ser != "506")
            row["cov_CF3_without_series_506"] = float(cov[m3].mean())
            rows.append(row)
            if fr == 0.25:
                for a in (1, 2, 3, 4):
                    for b in (1, 2, 3, 4):
                        mk = (rf == a) & (cf == b)
                        if mk.sum() >= 50 and len(np.unique(ser[mk])) >= 3:
                            cell_rows.append({"scale": s, "RF": a, "CF": b, "n": int(mk.sum()), "coverage": float(cov[mk].mean()),
                                              "mean_width": float(width[mk].mean())})
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "C1_conformal_sensitivity.csv", index=False)
    pd.DataFrame(cat_rows).to_csv(TAB / "C2_conformal_coverage_by_category.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(TAB / "C3_conformal_coverage_by_cell.csv", index=False)
    # CF = 3 under-coverage: which series carry it?
    r = D["y"] - np.load(ROOT / "results/predictions/geometry__xgb.npz")["pred"][0]
    t = df[cf == 3].assign(r=r[cf == 3]).groupby("series").agg(n=("r", "size"), mean_resid_pred_minus_meas=("r", lambda s: -s.mean()),
                                                                 rmse=("r", lambda s: float(np.sqrt(np.mean(s ** 2)))), RF=("RF", "median"))
    t["share_of_CF3_squared_error"] = (t.rmse ** 2 * t.n) / (t.rmse ** 2 * t.n).sum()
    t.sort_values("share_of_CF3_squared_error", ascending=False).reset_index().to_csv(TAB / "C4_cf3_error_by_series.csv", index=False)
    return T


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    D = load_model_data()
    f = OUT / "intervals.pkl"
    iv = pickle.load(open(f, "rb")) if f.exists() else run(D)
    T = evaluate(D, iv)
    print(T[["calib_fraction", "scale", "coverage", "cov_RF4", "cov_CF3", "cov_CF4", "max_abs_dev_RF", "mean_width", "winkler", "cov_CF3_without_series_506"]].to_string())


if __name__ == "__main__":
    main()
