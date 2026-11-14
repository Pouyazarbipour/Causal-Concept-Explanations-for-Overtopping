"""Prediction metrics on Y = log10(q*)."""
from __future__ import annotations

import numpy as np


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return {k: np.nan for k in ("n", "rmse", "mae", "r2", "bias", "f10", "fac2", "fac3", "fac10")} | {"n": 0}
    e = p - y
    ss = np.sum((y - y.mean()) ** 2)
    return {"n": int(len(y)), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "r2": float(1 - np.sum(e ** 2) / ss) if ss > 0 else np.nan, "bias": float(np.mean(e)),
            # F10: geometric standard deviation factor of q_pred/q_meas (10^RMSE of the log10 ratio)
            "f10": float(10 ** np.sqrt(np.mean(e ** 2))),
            "fac2": float(np.mean(np.abs(e) <= np.log10(2))), "fac3": float(np.mean(np.abs(e) <= np.log10(3))),
            "fac10": float(np.mean(np.abs(e) <= 1.0))}


METRICS = ("rmse", "mae", "r2", "bias", "f10", "fac2", "fac3", "fac10")
