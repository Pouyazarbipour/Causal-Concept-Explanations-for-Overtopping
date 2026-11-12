"""Counterfactual physical auditing.

Three audit types are kept strictly apart:
  A. physical input interventions  (this module: `build_views`, `curve_metrics`)
  B. model-representation interventions (computed inside ccw_net.py, reported separately)
  C. empirical matched comparisons of measured tests (`matched_freeboard_pairs`)

Interventions are defined on the RAW engineering variables at fixed water level
and fixed incident waves; the dimensionless predictors are then recomputed, so
dependent quantities stay consistent.  cotaexcl/cotaincl (which depend on Rc) are
not predictors, so no hidden inconsistency is introduced.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from .data_preprocessing import signed_log1p

# name -> (description, expected sign of dY, feasibility rule)
INTERVENTIONS = {
    "crest_up": "Raise the whole crest: Rc and Ac both + delta*Hm0 (water level and waves fixed)",
    "wall_up": "Raise the crown wall only: Rc + delta*Hm0, Ac fixed (only where Rc >= Ac)",
    "crest_wider": "Widen the armour crest: Gc + delta*Hm0 (only where an armour crest exists, Gc > 0)",
    "rougher": "Rougher/more permeable armour: gamma_f - d (only sloping lower profile, gamma_f - d >= 0.35)",
}
GF_STEPS = {0.05: 0.05, 0.1: 0.10, 0.25: 0.15, 0.5: 0.20, 1.0: 0.30}  # delta level -> gamma_f decrement


def features_from_raw(df: pd.DataFrame, feats: list[str], log1p: list[str], g: float = 9.81) -> np.ndarray:
    """Dimensionless predictor matrix (with the fixed signed-log1p transform) from raw variables."""
    H = df["hm0_toe"].to_numpy(float)
    L0 = g * df["tm10_toe"].to_numpy(float) ** 2 / (2 * np.pi)
    m = df["m"].to_numpy(float)
    cols = {"rc_h": df["rc"] / H, "ac_h": df["ac"] / H, "gc_l": df["gc"] / L0, "s0": H / L0, "h_l": df["h"] / L0,
            "ht_h": df["ht"] / H, "bt_l": df["bt"] / L0, "b_l": df["b_berm"] / L0, "hb_h": df["hb"] / H,
            "tanab": df["tanab"], "cotad": df["cotad"], "cotau": df["cotau"], "gf": df["gf_raw"],
            "beta": df["beta_deg"], "fs": np.where(m > 0, 1.0 / np.where(m > 0, m, 1), np.nan)}
    X = np.column_stack([np.asarray(cols[f], float) for f in feats])
    for j, f in enumerate(feats):
        if f in log1p:
            X[:, j] = signed_log1p(X[:, j])
    return X


def build_views(df: pd.DataFrame, feats, log1p, deltas, g: float = 9.81):
    """Return (X_views [n, V, p], view table, feasibility mask [n, V]).  View 0 is the observed record."""
    views = [("base", 0.0)]
    Xs = [features_from_raw(df, feats, log1p, g)]
    feas = [np.ones(len(df), bool)]
    H = df["hm0_toe"]
    slope_lower = (df["cotad"] >= 1).to_numpy()
    for name in INTERVENTIONS:
        for d in deltas:
            cf = df.copy()
            if name == "crest_up":
                cf["rc"], cf["ac"] = df["rc"] + d * H, df["ac"] + d * H
                ok = np.ones(len(df), bool)
            elif name == "wall_up":
                cf["rc"] = df["rc"] + d * H
                ok = (df["rc"] >= df["ac"] - 1e-9).to_numpy()
            elif name == "crest_wider":
                cf["gc"] = df["gc"] + d * H
                ok = (df["gc"] > 0).to_numpy()
            else:
                cf["gf_raw"] = df["gf_raw"] - GF_STEPS[d]
                ok = slope_lower & (cf["gf_raw"] >= 0.35 - 1e-9).to_numpy()
            views.append((name, d))
            Xs.append(features_from_raw(cf, feats, log1p, g))
            feas.append(ok)
    vt = pd.DataFrame(views, columns=["intervention", "delta"])
    return np.stack(Xs, axis=1), vt, np.stack(feas, axis=1)


def support_flags(X_train: np.ndarray, X_views_test: np.ndarray, k: int = 5, q: float = 0.99) -> np.ndarray:
    """Common-support diagnostic for counterfactual points of one fold.

    A point is 'supported' if (i) every predictor lies inside the training range and (ii) its mean
    distance to the k nearest training points (standardised predictors) does not exceed the q-quantile
    of the same distance among training points themselves (leave-one-out).  Missing values are
    median-imputed for the distance only."""
    med = np.nanmedian(X_train, axis=0)
    Xt = np.where(np.isnan(X_train), med, X_train)
    mu, sd = Xt.mean(0), Xt.std(0) + 1e-12
    Zt = (Xt - mu) / sd
    nn = NearestNeighbors(n_neighbors=k + 1).fit(Zt)
    d_tr = nn.kneighbors(Zt)[0][:, 1:].mean(1)
    thr = np.quantile(d_tr, q)
    lo, hi = Xt.min(0), Xt.max(0)
    n, V, p = X_views_test.shape
    Xv = X_views_test.reshape(-1, p)
    Xv = np.where(np.isnan(Xv), med, Xv)
    in_range = ((Xv >= lo - 1e-9) & (Xv <= hi + 1e-9)).all(1)
    d = nn.kneighbors((Xv - mu) / sd, n_neighbors=k)[0].mean(1)
    return (in_range & (d <= thr)).reshape(n, V)


def curve_metrics(P: np.ndarray, vt: pd.DataFrame, ok: np.ndarray, intervention: str,
                  tol: float, local_delta: float) -> pd.DataFrame:
    """Per-record response-curve statistics for one intervention.

    P: predictions [n, V].  ok: [n, V] mask (feasible AND supported).  A step is evaluated only when
    both of its end points are admissible.  Expected direction: Y non-increasing along the curve.
    Curves and steps are different units and are reported separately downstream."""
    cols = np.flatnonzero(vt["intervention"].to_numpy() == intervention)
    deltas = vt["delta"].to_numpy()[cols]
    Y = np.column_stack([P[:, 0], P[:, cols]])
    A = np.column_stack([ok[:, 0], ok[:, cols]])
    dY = np.diff(Y, axis=1)
    step_ok = A[:, 1:] & A[:, :-1]
    up = (dY > tol) & step_ok
    j_loc = int(np.argmin(np.abs(deltas - local_delta))) + 1
    out = pd.DataFrame({
        "n_steps_valid": step_ok.sum(1), "n_steps_up": up.sum(1),
        "max_step_rise": np.where(step_ok, dY, -np.inf).max(1),
        "local_valid": A[:, 0] & A[:, j_loc], "local_delta_y": Y[:, j_loc] - Y[:, 0],
        "global_valid": A[:, 0] & A[:, -1], "global_delta_y": Y[:, -1] - Y[:, 0]})
    out["local_reversal"] = out["local_valid"] & (out["local_delta_y"] > tol)
    out["global_reversal"] = out["global_valid"] & (out["global_delta_y"] > tol)
    out["curve_valid"] = out["n_steps_valid"] > 0
    out["curve_affected"] = out["n_steps_up"] > 0
    out.loc[~out["curve_valid"], "max_step_rise"] = np.nan
    return out


def matched_freeboard_pairs(df: pd.DataFrame, mcfg: dict) -> pd.DataFrame:
    """Type C: pairs of MEASURED tests in the same sub-structure (identical geometry except crest level)
    with the same toe depth, wave height, period and obliquity within tolerances, but different Rc.
    Because h is matched, the contrast is a crest-height contrast, not a water-level contrast."""
    rows = []
    for _, g in df.groupby("substructure"):
        if len(g) < 2 or g["rc"].nunique() < 2:
            continue
        a = g[["hm0_toe", "tm10_toe", "h", "beta_deg", "rc", "ac", "y"]].to_numpy(float)
        idx = g.index.to_numpy()
        H, T, h, b, rc = a[:, 0], a[:, 1], a[:, 2], np.nan_to_num(a[:, 3]), a[:, 4]
        relH = np.abs(H[:, None] - H[None]) / np.minimum(H[:, None], H[None])
        relT = np.abs(T[:, None] - T[None]) / np.minimum(T[:, None], T[None])
        relh = np.abs(h[:, None] - h[None]) / np.maximum(np.minimum(np.abs(h[:, None]), np.abs(h[None])), 1e-9)
        drc = (rc[None] - rc[:, None]) / H[:, None]
        okm = ((relH <= mcfg["rel_tol_wave"]) & (relT <= mcfg["rel_tol_wave"]) & (relh <= mcfg["rel_tol_depth"])
               & (np.abs(b[:, None] - b[None]) <= mcfg["abs_tol_beta"]) & (drc >= mcfg["min_rel_drc"]))
        for i, j in zip(*np.nonzero(okm)):
            rows.append((idx[i], idx[j], drc[i, j], a[j, 6] - a[i, 6], relH[i, j], relT[i, j], relh[i, j],
                         (a[j, 5] - a[i, 5]) / H[i]))
    return pd.DataFrame(rows, columns=["low", "high", "d_rc_h", "d_y", "rel_dH", "rel_dT", "rel_dh", "d_ac_h"])
