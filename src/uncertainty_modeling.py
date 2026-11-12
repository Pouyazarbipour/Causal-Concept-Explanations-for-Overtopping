"""Quality-aware uncertainty: conformal prediction with geometry-grouped calibration.

Design per outer fold (test geometries never seen):
  training geometries -> proper-training geometries + calibration geometries (grouped split).
  Mean model: XGBoost on physical predictors only (RF/CF never enter the mean).
  Scale models sigma(.) are fitted on grouped inner out-of-fold |residuals| of the proper-training part.
Exchangeable unit = geometry group: calibration scores are weighted 1/(K * n_g) (hierarchical conformal,
Lee, Barber & Willett 2023), with mass 1/(K+1) at +infinity.  The marginal guarantee is for a record drawn
from a NEW geometry; no guarantee is claimed per RF/CF category, except that Mondrian variants calibrate
within the category (guarantee then conditional on that category only, given enough groups).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline_models import FoldPrep, val_split
from .geometry_grouping import check_no_leakage, inner_group_folds
from .statistical_analysis import weighted_quantile

METHODS = ["split_abs", "norm_physical", "norm_quality_only", "norm_physical+quality", "mondrian_RF", "mondrian_CF",
           "mondrian_RFxCF_coarse", "cqr_physical", "cqr_physical+quality", "gaussian_physical_uncalibrated"]


def _xgb(params, **kw):
    import xgboost as xgb
    return xgb.XGBRegressor(n_jobs=1, tree_method="hist", **{**params, **kw})


def group_quantile(scores, groups, level: float) -> float:
    """Hierarchical-conformal quantile: each calibration GROUP carries total weight 1/(K+1)."""
    uniq, inv, cnt = np.unique(groups, return_inverse=True, return_counts=True)
    K = len(uniq)
    w = 1.0 / ((K + 1) * cnt[inv])
    return weighted_quantile(np.append(scores, np.inf), np.append(w, 1.0 / (K + 1)), level)


def onehot_q(rf, cf):
    return np.column_stack([(rf == v).astype(float) for v in (1, 2, 3, 4)] + [(cf == v).astype(float) for v in (1, 2, 3, 4)])


def run_fold(X, y, rf, cf, groups, tr, te, params, alphas, seed, calib_frac):
    """Returns dict method -> {alpha: (lo, hi)} for the test rows."""
    check_no_leakage(groups[tr], groups[te])
    is_fit, is_cal = val_split(groups[tr], seed, calib_frac)
    fit, cal = tr[is_fit], tr[is_cal]
    check_no_leakage(groups[fit], groups[cal])
    prep = FoldPrep(X[fit])
    Xf, Xc, Xt = prep.impute(X[fit]), prep.impute(X[cal]), prep.impute(X[te])
    Q = onehot_q(rf, cf)
    # grouped inner OOF residuals on the proper-training part -> training data for the scale models
    oof = np.zeros(len(fit))
    for a, b in inner_group_folds(groups[fit], 4, seed):
        oof[b] = _xgb(params, random_state=seed).fit(Xf[a], y[fit][a]).predict(Xf[b])
    lr = np.log(np.abs(y[fit] - oof) + 1e-3)
    mean = _xgb(params, random_state=seed).fit(Xf, y[fit])
    mc, mt = mean.predict(Xc), mean.predict(Xt)
    sp = dict(n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, min_child_weight=5)
    sig = {}
    s_phys = _xgb(sp, random_state=seed).fit(Xf, lr)
    sig["norm_physical"] = (np.exp(s_phys.predict(Xc)), np.exp(s_phys.predict(Xt)))
    s_both = _xgb(sp, random_state=seed).fit(np.hstack([Xf, Q[fit]]), lr)
    sig["norm_physical+quality"] = (np.exp(s_both.predict(np.hstack([Xc, Q[cal]]))), np.exp(s_both.predict(np.hstack([Xt, Q[te]]))))
    # quality-only scale: 8-parameter log-linear model in RF and CF categories
    beta = np.linalg.lstsq(np.hstack([Q[fit][:, 1:4], Q[fit][:, 5:8], np.ones((len(fit), 1))]), lr, rcond=None)[0]
    qo = lambda idx: np.exp(np.hstack([Q[idx][:, 1:4], Q[idx][:, 5:8], np.ones((len(idx), 1))]) @ beta)  # noqa: E731
    sig["norm_quality_only"] = (qo(cal), qo(te))
    rc, gc = np.abs(y[cal] - mc), groups[cal]
    res = {m: {} for m in METHODS}
    for al in alphas:
        qh = group_quantile(rc, gc, 1 - al)
        res["split_abs"][al] = (mt - qh, mt + qh)
        for m, (sc, st) in sig.items():
            qn = group_quantile(rc / sc, gc, 1 - al)
            res[m][al] = (mt - qn * st, mt + qn * st)
        z = {0.5: 0.6745, 0.2: 1.2816, 0.1: 1.6449, 0.05: 1.96}[al]
        s_t = sig["norm_physical"][1] * 1.2533  # E|r| -> sigma under normality (log-mean bias ignored)
        res["gaussian_physical_uncalibrated"][al] = (mt - z * s_t, mt + z * s_t)
        for name, cat_c, cat_t in (("mondrian_RF", rf[cal], rf[te]), ("mondrian_CF", cf[cal], cf[te]),
                                   ("mondrian_RFxCF_coarse", (rf[cal] >= 3) * 2 + (cf[cal] >= 2), (rf[te] >= 3) * 2 + (cf[te] >= 2))):
            lo, hi = np.empty(len(te)), np.empty(len(te))
            for v in np.unique(cat_t):
                mcal = cat_c == v
                # fall back to the pooled quantile when a category has fewer than 20 calibration groups
                qv = group_quantile(rc[mcal], gc[mcal], 1 - al) if len(np.unique(gc[mcal])) >= 20 else qh
                lo[cat_t == v], hi[cat_t == v] = mt[cat_t == v] - qv, mt[cat_t == v] + qv
            res[name][al] = (lo, hi)
        for name, A, C, T in (("cqr_physical", Xf, Xc, Xt),
                              ("cqr_physical+quality", np.hstack([Xf, Q[fit]]), np.hstack([Xc, Q[cal]]), np.hstack([Xt, Q[te]]))):
            ql = _xgb(sp, random_state=seed, objective="reg:quantileerror", quantile_alpha=al / 2).fit(A, y[fit])
            qu = _xgb(sp, random_state=seed, objective="reg:quantileerror", quantile_alpha=1 - al / 2).fit(A, y[fit])
            e = np.maximum(ql.predict(C) - y[cal], y[cal] - qu.predict(C))
            qe = group_quantile(e, gc, 1 - al)
            res[name][al] = (ql.predict(T) - qe, qu.predict(T) + qe)
    return res


def coverage_tables(df: pd.DataFrame, y, intervals: dict, alpha: float, n_boot: int, seed: int):
    """intervals: method -> (lo [R, n], hi [R, n]) for R repetitions.  Coverage is averaged over repetitions."""
    from .statistical_analysis import cluster_bootstrap
    rows, strat = [], []
    cl = df["geometry"].to_numpy()
    rf, cf, fam = df["RF"].to_numpy(), df["CF"].to_numpy(), df["family"].to_numpy()
    for m, (lo, hi) in intervals.items():
        cov = ((y >= lo) & (y <= hi)).mean(0)
        width = (hi - lo).mean(0)
        finite = np.isfinite(width)
        pt, l, h, _ = cluster_bootstrap(lambda idx: [cov[idx].mean()], cl, n_boot, seed)
        gcov = pd.Series(cov).groupby(cl).mean()
        by = lambda key, vals: {v: float(cov[key == v].mean()) for v in vals}  # noqa: E731
        brf, bcf, bfam = by(rf, (1, 2, 3, 4)), by(cf, (1, 2, 3, 4)), by(fam, np.unique(fam))
        cells = {(a, b): (float(cov[(rf == a) & (cf == b)].mean()), int(((rf == a) & (cf == b)).sum())) for a in (1, 2, 3, 4) for b in (1, 2, 3, 4)}
        ok_cells = {k: v[0] for k, v in cells.items() if v[1] >= 50}
        rows.append({"method": m, "nominal": 1 - alpha, "coverage": pt[0], "cov_lo": l[0], "cov_hi": h[0],
                     "geometry_mean_coverage": float(gcov.mean()), "mean_width": float(width[finite].mean()),
                     "median_width": float(np.median(width[finite])), "pct_infinite": float(100 * (~finite).mean()),
                     **{f"cov_RF{v}": brf[v] for v in brf}, **{f"cov_CF{v}": bcf[v] for v in bcf},
                     "max_abs_dev_RF": max(abs(v - (1 - alpha)) for v in brf.values()),
                     "max_abs_dev_CF": max(abs(v - (1 - alpha)) for v in bcf.values()),
                     "worst_RFxCF_cell(n>=50)": min(ok_cells, key=ok_cells.get), "worst_RFxCF_coverage": min(ok_cells.values()),
                     "worst_family": min(bfam, key=bfam.get), "worst_family_coverage": min(bfam.values())})
        for (a, b), (c, n) in cells.items():
            strat.append({"method": m, "RF": a, "CF": b, "n": n, "coverage": c, "mean_width": float(width[(rf == a) & (cf == b) & finite].mean()) if n else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(strat)
