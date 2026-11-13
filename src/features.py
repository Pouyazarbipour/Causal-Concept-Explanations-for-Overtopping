"""Dimensionless physical predictors from the raw CLASH variables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data_preprocessing import signed_log1p


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
