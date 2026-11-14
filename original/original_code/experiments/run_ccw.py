"""CCW-Net (Optuna-tuned per outer fold), loss-function ablations, unaligned wrapper and concept bottleneck.

Usage: python -m experiments.run_ccw
Seed 0 of every fold runs the Optuna search for the wrapper; later seeds reuse the selected configuration.
Full model, unaligned wrapper and concept bottleneck: all seeds; loss ablations: first two seeds.
Every fold-seed job is checkpointed.
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.ccw_net import VARIANTS, run_fold
from src.geometry_grouping import check_no_leakage
from src.pipeline import groups_for, load_model_data
from src.utils import ROOT, get_logger, out

log = get_logger("ccw")
CORE = ["ccw_full", "ccw_unaligned", "cbm"]
CACHE = ROOT / "results" / "models" / "ccw_cache"


def fold_job(scheme, k, seeds, tr, te, X, y, labels, groups, rfcf, mcfg, base_seed):
    CACHE.mkdir(parents=True, exist_ok=True)
    outs, tuned = [], None
    for si, seed in enumerate(seeds):
        f = CACHE / f"{scheme}__fold{k}__seed{seed}.pkl"
        if f.exists():
            r = pickle.load(open(f, "rb"))
        else:
            variants = CORE + ([v for v in VARIANTS if v not in CORE] if si < 2 else [])
            r = run_fold(k, seed, tr, te, X, y, labels, groups, rfcf, mcfg, variants, base_seed, tuned)
            tmp = f.with_suffix(".tmp")
            pickle.dump(r, open(tmp, "wb"))
            tmp.rename(f)
        tuned = r["C"]
        outs.append(r)
    return outs


def main() -> None:
    D = load_model_data()
    cfg = D["cfg"]
    E = cfg["experiment"]
    scheme = E["primary_scheme"]
    df, y, X = D["df"], D["y"], D["X"]
    labels, rfcf = D["labels"].to_numpy(), df[["RF", "CF"]].to_numpy()
    fold, groups = D["folds"][scheme].to_numpy(), groups_for(df, scheme)
    seeds = list(E["nn_seeds"])
    jobs = []
    for k in np.unique(fold):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        check_no_leakage(groups[tr], groups[te])
        jobs.append(delayed(fold_job)(scheme, int(k), seeds, tr, te, X, y, labels, groups, rfcf, cfg["model"], E["seed"]))
    t0 = time.time()
    res = [r for rr in Parallel(n_jobs=4)(jobs) for r in rr]
    log.info("%d fold-seed jobs in %.0fs", len(res), time.time() - t0)
    for n in sorted({n for r in res for n in r["preds"]}):
        P = np.full((len(seeds), len(y)), np.nan, dtype=np.float32)
        for r in res:
            if n in r["preds"]:
                P[seeds.index(r["seed"]), r["te"]] = r["preds"][n]
        P = P[np.isfinite(P).all(1)]
        np.savez_compressed(out("predictions", f"{scheme}__{n}.npz"), pred=P)
    pd.DataFrame([m for r in res for m in r["metrics"]]).to_csv(out("diagnostics", "ccw_metrics.csv"), index=False)
    pd.DataFrame([m for r in res for m in r["shapley"]]).to_csv(out("diagnostics", "ccw_shapley.csv"), index=False)
    pd.DataFrame([m for r in res for m in r["quality"]]).to_csv(out("diagnostics", "ccw_quality_branch.csv"), index=False)
    with open(out("diagnostics", "ccw_per_concept.json"), "w") as fh:
        json.dump([m for r in res for m in r["per_concept"]], fh)
    with open(out("diagnostics", "ccw_history.json"), "w") as fh:
        json.dump([{"fold": r["fold"], "seed": r["seed"], "variant": v, "history": h} for r in res for v, h in r["history"].items()], fh)
    d = ROOT / "results" / "models" / "optuna"
    d.mkdir(parents=True, exist_ok=True)
    tun = [r for r in res if "tuning" in r]
    pd.DataFrame([{"fold": r["fold"], "inner_val_rmse_std_units": r["tuning"]["value"], "n_trials": len(r["tuning"]["trials"]), **r["tuning"]["params"]}
                  for r in tun]).to_csv(d / f"{scheme}__ccw__best.csv", index=False)
    pd.concat([r["tuning"]["trials"].assign(fold=r["fold"]) for r in tun]).to_csv(d / f"{scheme}__ccw__trials.csv", index=False)
    # per-record explanation-quality quantities and concept vectors (seed-wise) for the RF/CF analysis
    for v in ("ccw_full", "ccw_unaligned"):
        keys = list(res[0]["records"][v])
        arrs = {}
        for key in keys:
            shape = (len(seeds), len(y)) + res[0]["records"][v][key].shape[1:]
            A = np.full(shape, np.nan, dtype=np.float32)
            for r in res:
                A[seeds.index(r["seed"]), r["te"]] = r["records"][v][key]
            arrs[key] = A
        np.savez_compressed(out("diagnostics", f"ccw_records__{v}.npz"), **arrs)
    # concept vectors of the full model: dimension may differ between folds, so store per fold (seed 0)
    pickle.dump({r["fold"]: (r["te"], r["reps"]["ccw_full"]) for r in res if r["seed"] == seeds[0]},
                open(out("diagnostics", "ccw_full_test_representations_seed0.pkl"), "wb"))


if __name__ == "__main__":
    main()
