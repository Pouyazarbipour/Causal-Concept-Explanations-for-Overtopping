"""Predictive-validity analysis of the CLASH Reliability (RF) and Complexity (CF) factors.

RF and CF are treated as ORDINAL: category effects are estimated per category (no equal-spacing
assumption) or through rank-based statistics.  Scale direction (documented): RF 1 = very reliable ...
4 = not reliable; CF 1 = simple ... 4 = very complex.  "Higher RF" therefore means LOWER reliability.

What this module can and cannot show: association of the ratings with out-of-fold prediction error,
residual dispersion and repeatability (predictive / criterion-related evidence).  It cannot show that a
rating is objectively correct; no independent re-rating exists in the repository.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from .statistical_analysis import cluster_bootstrap, jonckheere


def strata_table(df: pd.DataFrame, r: np.ndarray, factor: str, cluster: str, n_boot: int, seed: int) -> pd.DataFrame:
    """Residual statistics per category with cluster-bootstrap 95 % CIs."""
    f, cl = df[factor].to_numpy(), df[cluster].to_numpy()
    gmean = pd.Series(r).groupby(df["geometry"].to_numpy()).transform("mean").to_numpy()
    gsize = pd.Series(r).groupby(df["geometry"].to_numpy()).transform("size").to_numpy()
    within = np.where(gsize > 1, (r - gmean) * np.sqrt(gsize / np.maximum(gsize - 1, 1)), np.nan)
    rows = []
    for v in (1, 2, 3, 4):
        m = f == v

        def st(idx, m=m):
            s = idx[m[idx]]
            return [np.mean(np.abs(r[s])), np.sqrt(np.mean(r[s] ** 2)), np.mean(r[s]), np.sqrt(np.nanmean(within[s] ** 2)),
                    np.median(np.abs(r[s]))]
        pt, lo, hi, _ = cluster_bootstrap(st, cl, n_boot, seed)
        rows.append({"factor": factor, "category": v, "n": int(m.sum()), "n_geometries": df.loc[m, "geometry"].nunique(),
                     "n_series": df.loc[m, "series"].nunique(),
                     "mae": pt[0], "mae_lo": lo[0], "mae_hi": hi[0], "rmse": pt[1], "rmse_lo": lo[1], "rmse_hi": hi[1],
                     "bias": pt[2], "bias_lo": lo[2], "bias_hi": hi[2],
                     "within_geometry_sd": pt[3], "within_sd_lo": lo[3], "within_sd_hi": hi[3], "median_abs": pt[4]})
    return pd.DataFrame(rows)


def trend_tests(df, r, factor, cluster, n_boot, seed) -> dict:
    f, cl = df[factor].to_numpy(float), df[cluster].to_numpy()
    a = np.abs(r)

    def st(idx):
        rho = stats.spearmanr(a[idx], f[idx]).statistic
        hi_, lo_ = a[idx][f[idx] == 4], a[idx][f[idx] == 1]
        return [rho, hi_.mean() - lo_.mean(), np.sqrt(np.mean(r[idx][f[idx] == 4] ** 2)) / np.sqrt(np.mean(r[idx][f[idx] == 1] ** 2))]
    pt, lo, hi, reps = cluster_bootstrap(st, cl, n_boot, seed)
    z, p = jonckheere(a, f)
    pb = 2 * min((reps[:, 0] <= 0).mean(), (reps[:, 0] >= 0).mean())
    return {"factor": factor, "spearman_absres": pt[0], "spearman_lo": lo[0], "spearman_hi": hi[0], "boot_p_spearman": max(pb, 1 / n_boot),
            "mae_cat4_minus_cat1": pt[1], "diff_lo": lo[1], "diff_hi": hi[1], "rmse_ratio_cat4_cat1": pt[2], "ratio_lo": lo[2],
            "ratio_hi": hi[2], "jonckheere_z": z, "jonckheere_p_iid": p}


def ordinal_logit(df: pd.DataFrame, target: str, covars: list[str], cluster: str, n_boot: int, seed: int) -> pd.DataFrame:
    """Proportional-odds model  target(ordinal) ~ covariates  with cluster-bootstrap CIs (odds ratios per
    1 SD of each covariate).  A mixed-effects ordinal model is not available in the dependency set; the
    cluster bootstrap accounts for within-cluster dependence instead."""
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    X = df[covars].astype(float)
    X = (X - X.mean()) / X.std()
    yv = df[target].astype(int).to_numpy()
    Xv = X.to_numpy()

    def st(idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = OrderedModel(yv[idx], Xv[idx], distr="logit").fit(method="bfgs", disp=False, maxiter=200)
        return np.asarray(f.params)[:len(covars)]
    pt, lo, hi, reps = cluster_bootstrap(st, df[cluster].to_numpy(), n_boot, seed)
    p = [max(2 * min((reps[:, k] <= 0).mean(), (reps[:, k] >= 0).mean()), 1 / n_boot) for k in range(len(covars))]
    return pd.DataFrame({"outcome": target, "covariate": covars, "odds_ratio_per_sd": np.exp(pt), "or_lo": np.exp(lo),
                         "or_hi": np.exp(hi), "boot_p": p, "n": len(df), "n_clusters": df[cluster].nunique()})


def variance_model(df: pd.DataFrame, r: np.ndarray, cluster: str, n_boot: int, seed: int, interaction: bool = True) -> pd.DataFrame:
    """Multiplicative heteroscedastic model  E[r^2] = exp(a + RF_cat + CF_cat (+ linear-by-linear RFxCF)),
    Gamma GLM with log link.  exp(coef) are residual-variance ratios relative to RF=1 / CF=1."""
    import statsmodels.api as sm
    rf, cf = df["RF"].to_numpy(int), df["CF"].to_numpy(int)
    cols = [np.ones(len(df))] + [(rf == v).astype(float) for v in (2, 3, 4)] + [(cf == v).astype(float) for v in (2, 3, 4)]
    names = ["const", "RF=2", "RF=3", "RF=4", "CF=2", "CF=3", "CF=4"]
    if interaction:
        cols.append(((rf - 1) * (cf - 1)).astype(float))
        names.append("RFxCF (linear-by-linear, per step*step)")
    X = np.column_stack(cols)
    r2 = r ** 2 + 1e-8

    def st(idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(sm.GLM(r2[idx], X[idx], family=sm.families.Gamma(sm.families.links.Log())).fit(maxiter=100).params)
    pt, lo, hi, reps = cluster_bootstrap(st, df[cluster].to_numpy(), n_boot, seed)
    p = [max(2 * min((reps[:, k] <= 0).mean(), (reps[:, k] >= 0).mean()), 1 / n_boot) for k in range(len(names))]
    return pd.DataFrame({"term": names, "variance_ratio": np.exp(pt), "ratio_lo": np.exp(lo), "ratio_hi": np.exp(hi), "boot_p": p})


def geometry_level(df: pd.DataFrame, r: np.ndarray, r_random: np.ndarray, knn_dist: np.ndarray | None = None) -> pd.DataFrame:
    """One row per reconstructed geometry.  CF is aggregated by its maximum; discrepancies are kept."""
    t = df[["geometry", "series", "family", "CF", "RF"]].copy()
    t["r"], t["rr"] = r, r_random
    if knn_dist is not None:
        t["knn"] = knn_dist
    g = t.groupby("geometry")
    out = pd.DataFrame({"series": g["series"].first(), "family": g["family"].first(), "n": g.size(),
                        "cf_max": g["CF"].max(), "cf_min": g["CF"].min(), "rf_mean": g["RF"].mean(),
                        "mean_resid": g["r"].mean(), "abs_mean_resid": g["r"].mean().abs(),
                        "rmse_grouped": np.sqrt(g["r"].apply(lambda s: np.mean(s ** 2))),
                        "rmse_random": np.sqrt(g["rr"].apply(lambda s: np.mean(s ** 2))),
                        "within_sd": g["r"].std()})
    out["generalisation_gap"] = out["rmse_grouped"] - out["rmse_random"]
    if knn_dist is not None:
        out["knn_dist"] = g["knn"].mean()
    out["cf_inconsistent"] = out["cf_max"] != out["cf_min"]
    return out.reset_index()


def cf_geometry_table(G: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    """Geometry-level statistics per CF (geometries are the units; clusters for the bootstrap = series)."""
    rows = []
    cl = G["series"].to_numpy()
    cols = ["abs_mean_resid", "rmse_grouped", "generalisation_gap", "within_sd"] + (["knn_dist"] if "knn_dist" in G else [])
    A = G[cols].to_numpy(float)
    cf = G["cf_max"].to_numpy()
    for v in (1, 2, 3, 4):
        m = cf == v

        def st(idx, m=m):
            return np.nanmean(A[idx[m[idx]]], axis=0)
        pt, lo, hi, _ = cluster_bootstrap(st, cl, n_boot, seed)
        row = {"CF": v, "n_geometries": int(m.sum()), "n_series": G.loc[m, "series"].nunique(), "n_tests": int(G.loc[m, "n"].sum()),
               "n_cf_inconsistent": int(G.loc[m, "cf_inconsistent"].sum())}
        for k, c in enumerate(cols):
            row[c], row[c + "_lo"], row[c + "_hi"] = pt[k], lo[k], hi[k]
        rows.append(row)
    return pd.DataFrame(rows)


def joint_table(df, r, cluster, n_boot, seed, min_n: int = 50) -> pd.DataFrame:
    rows = []
    rf, cf, cl = df["RF"].to_numpy(), df["CF"].to_numpy(), df[cluster].to_numpy()
    for a in (1, 2, 3, 4):
        for b in (1, 2, 3, 4):
            m = (rf == a) & (cf == b)
            row = {"RF": a, "CF": b, "n": int(m.sum()), "n_geometries": df.loc[m, "geometry"].nunique(),
                   "n_series": df.loc[m, "series"].nunique()}
            if m.sum() >= 5:
                def st(idx, m=m):
                    s = idx[m[idx]]
                    return [np.mean(np.abs(r[s])), np.sqrt(np.mean(r[s] ** 2)), np.mean(r[s])]
                pt, lo, hi, _ = cluster_bootstrap(st, cl, n_boot, seed)
                row.update({"mae": pt[0], "mae_lo": lo[0], "mae_hi": hi[0], "rmse": pt[1], "rmse_lo": lo[1], "rmse_hi": hi[1],
                            "bias": pt[2]})
            row["support"] = "adequate" if (m.sum() >= min_n and row["n_series"] >= 3) else "INSUFFICIENT (do not interpret)"
            rows.append(row)
    return pd.DataFrame(rows)
