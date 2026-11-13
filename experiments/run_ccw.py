"""Stage E: CCW-Net, its ablations, the unaligned wrapper and the concept-bottleneck baseline.

Usage: python -m experiments.run_ccw [--schemes geometry series]
Outputs: results/predictions/<scheme>__<variant>.npz  (pred [seeds, n, views]),
         results/diagnostics/ccw_<scheme>_{metrics,per_concept,shapley,quality,history}.*
"""
from __future__ import annotations

import argparse
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
SENS = ["ccw_weights_x0.1", "ccw_weights_x10"]
CORE = ["ccw_full", "ccw_unaligned", "cbm"]
CACHE = ROOT / "results" / "diagnostics" / "ccw_cache"


def cached_fold(scheme, k, seed, *args):
    """Checkpoint every fold-seed job so that an interrupted run resumes instead of restarting."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{scheme}__fold{k}__seed{seed}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    r = run_fold(k, seed, *args)
    tmp = f.with_suffix(".tmp")
    pickle.dump(r, open(tmp, "wb"))
    tmp.rename(f)
    return r


def main() -> None:
    D = load_model_data()
    cfg = D["cfg"]
    E = cfg["experiment"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="*", default=E["neural_schemes"])
    ap.add_argument("--n_jobs", type=int, default=4)
    a = ap.parse_args()
    df, y, Xv = D["df"], D["y"], D["X_views"]
    labels, rfcf = D["labels"].to_numpy(), df[["RF", "CF"]].to_numpy()
    for scheme in a.schemes:
        fold, groups = D["folds"][scheme].to_numpy(), groups_for(df, scheme)
        primary = scheme == E["primary_scheme"]
        jobs = []
        for seed in E["nn_seeds"]:
            # full model, unaligned wrapper and CBM: all seeds; loss/architecture ablations: first two seeds;
            # loss-weight sensitivity: first seed (primary scheme only)
            si = E["nn_seeds"].index(seed)
            variants = list(CORE)
            if primary and si < 2:
                variants += [v for v in VARIANTS if v not in CORE and (v not in SENS or si == 0)]
            for k in np.unique(fold):
                tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
                check_no_leakage(groups[tr], groups[te])
                jobs.append(delayed(cached_fold)(scheme, int(k), seed, tr, te, Xv, y, labels, groups, rfcf, cfg["model"], variants, E["seed"],
                                              E["audit"]["meaningful_tol"], primary and seed == E["nn_seeds"][0]))
        t0 = time.time()
        res = Parallel(n_jobs=a.n_jobs)(jobs)
        log.info("%s: %d fold-seed jobs in %.0fs", scheme, len(res), time.time() - t0)
        names = sorted({n for r in res for n in r["preds"]})
        seeds = list(E["nn_seeds"])
        for n in names:
            P = np.full((len(seeds), len(y), Xv.shape[1]), np.nan, dtype=np.float32)
            for r in res:
                if n in r["preds"]:
                    P[seeds.index(r["seed"]), r["te"]] = r["preds"][n]
            np.savez_compressed(out("predictions", f"{scheme}__{n}.npz"), pred=P)
        pd.DataFrame([m for r in res for m in r["metrics"]]).to_csv(out("diagnostics", f"ccw_{scheme}_metrics.csv"), index=False)
        pd.DataFrame([m for r in res for m in r["shapley"]]).to_csv(out("diagnostics", f"ccw_{scheme}_shapley.csv"), index=False)
        pd.DataFrame([m for r in res for m in r["quality"]]).to_csv(out("diagnostics", f"ccw_{scheme}_quality_branch.csv"), index=False)
        with open(out("diagnostics", f"ccw_{scheme}_per_concept.json"), "w") as fh:
            json.dump([m for r in res for m in r["per_concept"]], fh)
        with open(out("diagnostics", f"ccw_{scheme}_history.json"), "w") as fh:
            json.dump([{"fold": r["fold"], "seed": r["seed"], "variant": v, "history": h} for r in res for v, h in r["history"].items()], fh)
        if primary:
            M = np.full((len(y), labels.shape[1], cfg["model"]["ccw"]["concept_dim"]), np.nan, dtype=np.float32)
            for r in res:
                if "reps" in r:
                    M[r["te"]] = r["reps"]
            np.savez_compressed(out("diagnostics", "ccw_geometry_test_representations_seed0.npz"), M=M)


if __name__ == "__main__":
    main()
