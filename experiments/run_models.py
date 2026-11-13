"""Nested, geometry-grouped, Optuna-tuned cross-validation of the diagnostic regression models.

Usage: python -m experiments.run_models [--models rf xgb catboost mlp] [--schemes geometry random series]
Writes: results/predictions/<scheme>__<model>.npz   out-of-fold predictions [seeds, n] for every record
        results/models/optuna/<scheme>__<model>__*   best hyper-parameters, objective values, full trial history
Every fold is checkpointed, so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.baseline_models import outer_fold
from src.pipeline import groups_for, load_model_data
from src.utils import ROOT, get_logger, out

log = get_logger("models")
CACHE = ROOT / "results" / "models" / "fold_cache"


def run(name, scheme, D, n_jobs=4):
    cfg = D["cfg"]
    E, M = cfg["experiment"], cfg["model"]
    X, y = D["X"], D["y"]
    fold, groups = D["folds"][scheme].to_numpy(), groups_for(D["df"], scheme)
    seeds = E["nn_seeds"] if name == "mlp" else [0]
    t0 = time.time()
    res = Parallel(n_jobs=n_jobs)(delayed(outer_fold)(name, scheme, int(k), np.flatnonzero(fold != k), np.flatnonzero(fold == k), X, y, groups,
                                                      seeds, M["optuna_trials"][name], E["n_inner_folds"], E["seed"], M["mlp"], CACHE,
                                                      scheme == E["primary_scheme"]) for k in np.unique(fold))
    P = np.full((len(seeds), len(y)), np.nan, dtype=np.float32)
    best, hist = [], []
    for r in res:
        P[:, r["te"]] = r["pred"]
        best.append({"fold": r["fold"], "inner_cv_rmse": r["inner_rmse"], "outer_test_rmse": float(np.sqrt(np.mean((r["pred"].mean(0) - y[r["te"]]) ** 2))),
                     "n_trials": len(r["trials"]), **r["best"]})
        hist.append(r["trials"].assign(fold=r["fold"]))
    assert np.isfinite(P).all(), "every record must receive an out-of-fold prediction"
    np.savez_compressed(out("predictions", f"{scheme}__{name}.npz"), pred=P)
    d = ROOT / "results" / "models" / "optuna"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(best).to_csv(d / f"{scheme}__{name}__best.csv", index=False)
    pd.concat(hist).to_csv(d / f"{scheme}__{name}__trials.csv", index=False)
    if name == "xgb" and res[0]["shap"] is not None:
        S = np.full((len(y), X.shape[1] + 1), np.nan, dtype=np.float32)
        for r in res:
            S[r["te"]] = r["shap"]
        np.savez_compressed(out("diagnostics", f"shap__{scheme}__xgb.npz"), shap=S)
    log.info("%s %s OOF rmse=%.4f (%.0fs)", scheme, name, float(np.sqrt(np.mean((P.mean(0) - y) ** 2))), time.time() - t0)


def main():
    D = load_model_data()
    E = D["cfg"]["experiment"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=E["models"])
    ap.add_argument("--schemes", nargs="*", default=[E["primary_scheme"]] + E["sensitivity_schemes"])
    a = ap.parse_args()
    for scheme in a.schemes:
        for name in a.models:
            if scheme != E["primary_scheme"] and name != E["reference_model"]:
                continue        # sensitivity schemes: reference model only
            run(name, scheme, D)


if __name__ == "__main__":
    main()
