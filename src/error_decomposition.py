"""Hierarchical decomposition of out-of-fold residuals:  r = mu + sum_k (random intercept of level k) + e.

Exact REML for nested random intercepts, evaluated cluster by cluster and maximised numerically.
Levels are whatever identifiers exist: test series (original identifier) and reconstructed geometry;
a laboratory level is only identifiable for the records whose laboratory could be inferred.

Terminology.  The residual component e is "within-geometry residual variation".  It contains
measurement-associated uncertainty, natural experimental variability, input uncertainty and remaining
model discrepancy; it is NOT pure measurement error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _prepare(y, levels):
    """Per top-level cluster: residual vector and the 0/1 design blocks of every level."""
    top = levels[0]
    out = []
    for c in np.unique(top):
        idx = np.flatnonzero(top == c)
        Zs = []
        for l in levels:
            _, inv = np.unique(l[idx], return_inverse=True)
            Z = np.zeros((len(idx), inv.max() + 1))
            Z[np.arange(len(idx)), inv] = 1.0
            Zs.append(Z @ Z.T)
        out.append((idx, y[idx], Zs))
    return out


def _neg_reml(theta, clusters):
    """-2 x restricted log-likelihood (intercept-only mean) for variances exp(theta)."""
    v = np.exp(theta)
    ld, a, b, c = 0.0, 0.0, 0.0, 0.0
    for _, yc, Zs in clusters:
        V = v[-1] * np.eye(len(yc))
        for k, ZZ in enumerate(Zs):
            V += v[k] * ZZ
        L = np.linalg.cholesky(V)
        ld += 2 * np.log(np.diag(L)).sum()
        w = np.linalg.solve(L, np.column_stack([np.ones(len(yc)), yc]))
        a += w[:, 0] @ w[:, 0]
        b += w[:, 0] @ w[:, 1]
        c += w[:, 1] @ w[:, 1]
    return ld + np.log(a) + c - b * b / a


def reml_nested(y: np.ndarray, levels: list[np.ndarray], init=None, clusters=None, blup_level: int | None = None):
    """REML variance components for nested random intercepts: r = mu + sum_k u_k + e.

    `levels[0]` defines independent top-level clusters; every further level must be nested in it
    (labels are made unique within the parent by the caller).  The likelihood is evaluated exactly,
    cluster by cluster, and maximised over log-variances with L-BFGS-B."""
    from scipy.optimize import minimize
    y = np.asarray(y, float)
    clusters = clusters if clusters is not None else _prepare(y, levels)
    x0 = np.log(np.full(len(levels) + 1, y.var() / (len(levels) + 1))) if init is None else np.log(np.maximum(init, 1e-8))
    res = minimize(_neg_reml, x0, args=(clusters,), method="L-BFGS-B", bounds=[(-14, 3)] * len(x0), options={"maxiter": 60, "ftol": 1e-9})
    v = np.exp(res.x)
    v[v < 2e-6] = 0.0           # at the lower bound: component not distinguishable from zero
    out = {"var": v[:-1], "resid_var": float(v[-1]), "converged": bool(res.success), "n_groups": [len(np.unique(l)) for l in levels]}
    if blup_level is not None:   # BLUPs of one level (e.g. geometry effects)
        a = b = 0.0
        sol = []
        for idx, yc, Zs in clusters:
            V = v[-1] * np.eye(len(yc)) + sum(v[k] * ZZ for k, ZZ in enumerate(Zs))
            Vi1, Viy = np.linalg.solve(V, np.ones(len(yc))), np.linalg.solve(V, yc)
            a, b = a + Vi1.sum(), b + Viy.sum()
            sol.append((idx, Vi1, Viy))
        mu = b / a
        lab = levels[blup_level]
        eff = {}
        for idx, Vi1, Viy in sol:
            alpha = Viy - mu * Vi1
            for g in np.unique(lab[idx]):
                eff[g] = v[blup_level] * alpha[lab[idx] == g].sum()
        out["mu"], out["effects"] = float(mu), eff
    return out


def _boot_one(clusters, n_levels, seed, init):
    rng = np.random.RandomState(seed)
    pick = rng.randint(0, len(clusters), len(clusters))
    f = reml_nested(None if False else np.concatenate([clusters[i][1] for i in pick]), [None] * n_levels, init, [clusters[i] for i in pick])
    return np.concatenate([f["var"], [f["resid_var"]]])


def decompose(y, df: pd.DataFrame, level_cols: list[str], n_boot: int = 100, seed: int = 0, n_jobs: int = 4, blup: bool = False):
    """Variance components with cluster-bootstrap CIs (clusters = top level, resampled whole).
    Lower-level labels are prefixed by their parents so that the hierarchy is nested by construction."""
    levels, prefix = [], None
    for c in level_cols:
        lab = df[c].astype(str).to_numpy()
        prefix = lab if prefix is None else np.char.add(np.char.add(prefix, "/"), lab)
        levels.append(prefix.copy())
    y = np.asarray(y, float)
    clusters = _prepare(y, levels)
    fit = reml_nested(y, levels, clusters=clusters, blup_level=len(levels) - 1 if blup else None)
    est = np.concatenate([fit["var"], [fit["resid_var"]]])
    rows = pd.DataFrame({"component": level_cols + ["within-geometry residual"], "n_units": fit["n_groups"] + [len(y)],
                         "variance": est, "share": est / est.sum(), "sd": np.sqrt(est)})
    if n_boot:
        reps = np.array(Parallel(n_jobs=n_jobs)(delayed(_boot_one)(clusters, len(levels), seed + b, est) for b in range(n_boot)))
        sh = reps / reps.sum(1, keepdims=True)
        rows["variance_lo"], rows["variance_hi"] = np.percentile(reps, [2.5, 97.5], axis=0)
        rows["share_lo"], rows["share_hi"] = np.percentile(sh, [2.5, 97.5], axis=0)
    return rows, fit


def replicate_pairs(df: pd.DataFrame, rel_tol: float = 0.01) -> pd.DataFrame:
    """Near-replicate MEASURED tests: same geometry group, and Hm0, Tm-1,0, h, Rc within rel_tol, same beta.
    The spread of log10 q* between such tests is a model-free repeatability indicator."""
    rows = []
    for _, g in df.groupby("geometry"):
        if len(g) < 2:
            continue
        a = g[["hm0_toe", "tm10_toe", "h", "rc"]].to_numpy(float)
        b = np.nan_to_num(g["beta_deg"].to_numpy(float))
        idx = g.index.to_numpy()
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = np.abs(a[:, None, :] - a[None]) / np.maximum(np.maximum(np.abs(a[:, None, :]), np.abs(a[None])), 1e-12)
        ok = (rel <= rel_tol).all(-1) & (np.abs(b[:, None] - b[None]) < 0.5)
        for i, j in zip(*np.nonzero(np.triu(ok, 1))):
            rows.append((idx[i], idx[j]))
    p = pd.DataFrame(rows, columns=["i", "j"])
    if len(p):
        p["dy"] = df.loc[p["j"], "y"].to_numpy() - df.loc[p["i"], "y"].to_numpy()
        p["rf_max"] = np.maximum(df.loc[p["i"], "RF"].to_numpy(), df.loc[p["j"], "RF"].to_numpy())
        p["cf_max"] = np.maximum(df.loc[p["i"], "CF"].to_numpy(), df.loc[p["j"], "CF"].to_numpy())
        p["geometry"] = df.loc[p["i"], "geometry"].to_numpy()
        p["series"] = df.loc[p["i"], "series"].to_numpy()
    return p
