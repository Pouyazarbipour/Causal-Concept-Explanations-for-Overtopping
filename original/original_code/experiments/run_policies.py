"""Counterfactual data-selection policies on the primary (geometry-grouped) folds.

Usage: python -m experiments.run_policies
Needs the Optuna results of experiments.run_models (per-fold hyper-parameters tuned on training data).
Writes results/predictions/policy__<learner>.npz with pred/lo/hi [n] per policy.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from src.data_selection import POLICIES, policy_fold
from src.pipeline import groups_for, load_model_data
from src.utils import ROOT, get_logger, out

log = get_logger("policies")
CACHE = ROOT / "results" / "models" / "fold_cache"


def job(learner, k, D, fold, groups):
    f = CACHE / f"policy__{learner}__fold{k}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    E = D["cfg"]["experiment"]
    best = pickle.load(open(CACHE / f"{E['primary_scheme']}__{learner}__fold{k}.pkl", "rb"))["best"]
    df = D["df"]
    r = policy_fold(learner, best, D["X"], D["y"], df["RF"].to_numpy(int), df["CF"].to_numpy(int), df["flag_remark_do_not_use"].to_numpy(bool),
                    groups, np.flatnonzero(fold != k), np.flatnonzero(fold == k), E["seed"] + k, E["uncertainty"]["calib_fraction"],
                    E["uncertainty"]["alpha"])
    pickle.dump(r, open(f, "wb"))
    return r


def main():
    D = load_model_data()
    E = D["cfg"]["experiment"]
    scheme = E["primary_scheme"]
    fold, groups = D["folds"][scheme].to_numpy(), groups_for(D["df"], scheme)
    n = len(D["y"])
    for learner in ("xgb", "rf"):
        res = Parallel(n_jobs=4)(delayed(job)(learner, int(k), D, fold, groups) for k in np.unique(fold))
        arr = {f"{p}__{a}": np.full(n, np.nan) for p in POLICIES for a in ("pred", "lo", "hi")}
        vm, ntr = [], []
        for k, (o, beta) in zip(np.unique(fold), res):
            te = np.flatnonzero(fold == k)
            for p in POLICIES:
                for a in ("pred", "lo", "hi"):
                    arr[f"{p}__{a}"][te] = o[p][a]
                ntr.append({"fold": int(k), "policy": p, "n_train": o[p]["n_train"]})
            vm.append({"fold": int(k), **dict(zip(["const", "RF=2", "RF=3", "RF=4", "CF=2", "CF=3", "CF=4"], np.exp(beta)))})
        np.savez_compressed(out("predictions", f"policy__{learner}.npz"), **arr)
        pd.DataFrame(vm).to_csv(out("diagnostics", f"policy_variance_model__{learner}.csv"), index=False)
        pd.DataFrame(ntr).to_csv(out("diagnostics", f"policy_n_train__{learner}.csv"), index=False)
        log.info("%s done", learner)


if __name__ == "__main__":
    main()
