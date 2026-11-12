"""Hierarchical decomposition of out-of-fold residuals:  r = mu + sum_k (random intercept of level k) + e.

REML estimates by the EM algorithm on Henderson's mixed-model equations (dense; a few thousand random
effects).  Levels are whatever identifiers exist, e.g. laboratory/source unit, test series, geometry.

Terminology.  The residual component e is "within-geometry residual variation".  It contains
measurement-associated uncertainty, natural experimental variability, input uncertainty and remaining
model discrepancy; it is NOT pure measurement error.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def reml_em(y: np.ndarray, levels: list[np.ndarray], n_iter: int = 300, tol: float = 1e-7, init=None):
    """Return dict(mu, var=[sigma2 per level], resid_var, effects=[BLUP arrays], labels)."""
    y = np.asarray(y, float)
    n = len(y)
    codes, uniq = zip(*[np.unique(l, return_inverse=True)[::-1] for l in levels])
    q = [len(u) for u in uniq]
    off = np.concatenate([[0], np.cumsum(q)])
    Q = off[-1]
    Z = np.zeros((n, Q), dtype=np.float32)
    for k, c in enumerate(codes):
        Z[np.arange(n), off[k] + c] = 1.0
    ZtZ = (Z.T @ Z).astype(float)
    Zt1 = Z.sum(0).astype(float)
    Zty = (Z.T @ y).astype(float)
    s2 = np.array(init[:-1] if init is not None else [y.var() / (len(levels) + 1)] * len(levels), float)
    se2 = float(init[-1] if init is not None else y.var() / (len(levels) + 1))
    C = np.empty((Q + 1, Q + 1))
    rhs = np.concatenate([[y.sum()], Zty])
    for _ in range(n_iter):
        C[0, 0], C[0, 1:], C[1:, 0] = n, Zt1, Zt1
        C[1:, 1:] = ZtZ
        d = np.concatenate([np.full(q[k], se2 / max(s2[k], 1e-12)) for k in range(len(q))])
        C[1:, 1:][np.diag_indices(Q)] += d
        Ci = np.linalg.inv(C)
        sol = Ci @ rhs
        mu, u = sol[0], sol[1:]
        new = np.array([(u[off[k]:off[k + 1]] @ u[off[k]:off[k + 1]] + se2 * np.trace(Ci[1 + off[k]:1 + off[k + 1], 1 + off[k]:1 + off[k + 1]])) / q[k]
                        for k in range(len(q))])
        new_e = float(y @ (y - mu - Z @ u) / (n - 1))
        done = max(np.max(np.abs(new - s2)), abs(new_e - se2)) < tol
        s2, se2 = new, max(new_e, 1e-12)
        if done:
            break
    return {"mu": float(mu), "var": s2, "resid_var": se2, "effects": [u[off[k]:off[k + 1]] for k in range(len(q))],
            "labels": list(uniq), "n_groups": q}


def _boot_one(y, levels, top, seed, init):
    rng = np.random.RandomState(seed)
    uniq = np.unique(top)
    members = {u: np.flatnonzero(top == u) for u in uniq}
    idx, tag = [], []
    for b, u in enumerate(rng.choice(uniq, len(uniq))):
        idx.append(members[u])
        tag.append(np.full(len(members[u]), b))
    idx, tag = np.concatenate(idx), np.concatenate(tag)
    lv = [np.char.add(np.char.add(tag.astype(str), "#"), l[idx].astype(str)) for l in levels]
    f = reml_em(y[idx], lv, n_iter=40, init=init)
    return np.concatenate([f["var"], [f["resid_var"]]])


def decompose(y, df: pd.DataFrame, level_cols: list[str], n_boot: int = 100, seed: int = 0, n_jobs: int = 4) -> pd.DataFrame:
    """Variance components with cluster-bootstrap CIs (clusters = top level).  Lower levels are nested by
    construction of their labels (series within lab unit, geometry within series)."""
    levels = [df[c].astype(str).to_numpy() for c in level_cols]
    fit = reml_em(y, levels)
    est = np.concatenate([fit["var"], [fit["resid_var"]]])
    rows = pd.DataFrame({"component": level_cols + ["within-geometry residual"], "n_units": fit["n_groups"] + [len(y)],
                         "variance": est, "share": est / est.sum(), "sd": np.sqrt(est)})
    if n_boot:
        reps = np.array(Parallel(n_jobs=n_jobs)(delayed(_boot_one)(y, levels, levels[0], seed + b, est) for b in range(n_boot)))
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
