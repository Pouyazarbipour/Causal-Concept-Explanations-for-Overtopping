"""Phase VIII (revision): counterfactual data-selection policies, version 2.

The original implementation (src/data_selection.py, results/predictions/policy__*.npz) is kept unchanged.
Version 2 differs in three documented respects:

1. Independence of the interval scale (correction).  In v1 the variance model sigma^2(RF, CF) used by
   P6a/P6b/P6c was fitted to inner out-of-fold residuals of the WHOLE outer-training partition, which
   includes the calibration geometries.  In v2 it is fitted to grouped inner out-of-fold residuals of the
   proper-training geometries only, so the conformal score function does not depend on calibration data.
2. Partial pooling.  Besides the 7-parameter main-effects model, a shrinkage estimator of the 16 RF x CF
   cell variances (empirical-Bayes on the log scale, cells pulled towards the main-effects fit in proportion
   to their sampling variance) is evaluated as P6b-shrunk.
3. Two complementary experiments:
   A. fixed hyper-parameters (as in v1: tuned once on the full training partition of the fold);
   B. policy-specific nested tuning: for each policy Optuna (TPE, same space and budget as the original
      XGBoost, 30 trials) on grouped inner folds of the outer-training partition, with the policy applied
      to the inner-training rows and the objective evaluated on the policy-eligible inner-validation rows
      (as a practitioner who adopts the policy would do).  The outer test rows are never used.
All policies are evaluated on identical held-out tests of each outer fold.
"""
from __future__ import annotations

import numpy as np

from .baseline_models import FoldPrep, fit_model, suggest, tree_key, val_split
from .geometry_grouping import check_no_leakage, inner_group_folds
from .uncertainty_modeling import group_quantile

POLICIES = ["P1_retain_all", "P2_excl_RF4", "P3_excl_CF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks",
            "P6a_inverse_variance_weights", "P6b_quality_aware_intervals", "P6b_shrunk_cell_intervals", "P6c_weights_and_intervals"]


def design_q(rf, cf):
    return np.column_stack([np.ones(len(rf))] + [(rf == v).astype(float) for v in (2, 3, 4)] + [(cf == v).astype(float) for v in (2, 3, 4)])


def variance_models(r, rf, cf):
    """Main-effects log-variance model and an empirical-Bayes shrinkage of the 16 cell log-variances."""
    lr = np.log(r ** 2 + 1e-4)
    beta = np.linalg.lstsq(design_q(rf, cf), lr, rcond=None)[0]
    main = lambda a, b: np.exp(design_q(a, b) @ beta)  # noqa: E731
    fit_main = design_q(rf, cf) @ beta
    resid = lr - fit_main
    s2 = np.var(resid)                       # within-cell variance of log r^2 (about pi^2/2 for Gaussian r)
    cell = {}
    dev = []
    for a in (1, 2, 3, 4):
        for b in (1, 2, 3, 4):
            m = (rf == a) & (cf == b)
            if m.sum():
                dev.append((a, b, resid[m].mean(), s2 / m.sum()))
    d = np.array([x[2] for x in dev])
    v = np.array([x[3] for x in dev])
    tau2 = max(np.var(d) - v.mean(), 0.0)    # between-cell variance of deviations (method of moments)
    for (a, b, dm, vv) in dev:
        cell[(a, b)] = dm * tau2 / (tau2 + vv) if tau2 > 0 else 0.0
    corr = np.log(np.mean(np.exp(resid)))     # retransformation so that E[r^2] is matched on average

    def shrunk(a_, b_):
        adj = np.array([cell.get((x, y), 0.0) for x, y in zip(a_, b_)])
        return np.exp(design_q(a_, b_) @ beta + adj + corr)
    return beta, (lambda a, b: main(a, b) * np.exp(corr)), shrunk, {"tau2": tau2, "cells": cell}


def eff_n(w):
    return float(w.sum() ** 2 / (w ** 2).sum()) if w is not None else np.nan


def _inner_oof(learner, params, X, y, groups, rows, seed, n_inner=3):
    oof = np.zeros(len(rows))
    for a, b in inner_group_folds(groups[rows], n_inner, seed):
        prep = FoldPrep(X[rows][a])
        pred, _ = fit_model(learner, params, prep.impute(X[rows][a]), y[rows][a], groups[rows][a], seed)
        oof[b] = pred(prep.impute(X[rows][b]))
    return oof


def policy_masks(rf, cf, dnu, w_emp):
    rf4, cf4 = rf == 4, cf == 4
    clash = ((4 - rf) * (4 - cf)).astype(float)
    one = np.ones(len(rf), bool)
    return {
        "P1_retain_all": (one, None, None),
        "P2_excl_RF4": (~rf4, None, None),
        "P3_excl_CF4": (~cf4, None, None),
        "P4_excl_RF4_or_CF4": (~rf4 & ~cf4, None, None),
        "P5_CLASH_weights": (clash > 0, clash, None),
        "P5b_CLASH_weights_and_remarks": ((clash > 0) & ~dnu, clash, None),
        "P6a_inverse_variance_weights": (one, w_emp, None),
        "P6b_quality_aware_intervals": (one, None, "main"),
        "P6b_shrunk_cell_intervals": (one, None, "shrunk"),
        "P6c_weights_and_intervals": (one, w_emp, "main"),
    }


def tune_policy(params_space_seed, X, y, groups, rows, w, n_trials, seed):
    """Optuna on grouped inner folds of the outer-training partition; policy-eligible rows only; weighted fit."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    Xr, yr, gr = X[rows], y[rows], groups[rows]
    wr = None if w is None else w[rows]
    folds = inner_group_folds(gr, 3, seed)

    def objective(trial):
        p = suggest("xgb", trial)
        se, cnt, trees = 0.0, 0, []
        for a, b in folds:
            prep = FoldPrep(Xr[a])
            A, B = prep.impute(Xr[a]), prep.impute(Xr[b])
            pred, info = fit_model("xgb", p, A, yr[a], gr[a], seed, None if wr is None else wr[a], eval_set=(B, yr[b]))
            e = (pred(B) - yr[b]) ** 2
            se += float(e.sum())
            cnt += len(b)
            trees.append(info["n_trees"])
        trial.set_user_attr("n_trees", int(np.ceil(1.1 * np.mean(trees))))
        return float(np.sqrt(se / cnt))
    st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    st.optimize(objective, n_trials=n_trials)
    best = suggest("xgb", optuna.trial.FixedTrial(st.best_params))
    best[tree_key("xgb")] = st.best_trial.user_attrs["n_trees"]
    return best, float(st.best_value)


def policy_fold_v2(params, X, y, rf, cf, dnu, groups, tr, te, seed, calib_frac=0.25, alpha=0.1, tuned=False, n_trials=30):
    """All policies on one outer fold (XGBoost).  Returns per-policy predictions, intervals, weights, ESS."""
    check_no_leakage(groups[tr], groups[te])
    is_fit, is_cal = val_split(groups[tr], seed, calib_frac)      # identical calibration geometries for every policy
    fit_rows = np.flatnonzero(is_fit)
    # variance model from proper-training geometries only (v2 correction)
    oof = _inner_oof("xgb", params, X[tr], y[tr], groups[tr], fit_rows, seed)
    beta, s2_main, s2_shrunk, info = variance_models(y[tr][fit_rows] - oof, rf[tr][fit_rows], cf[tr][fit_rows])
    w_emp = 1.0 / s2_main(rf[tr], cf[tr])
    w_emp /= w_emp.mean()
    masks = policy_masks(rf[tr], cf[tr], dnu[tr], w_emp)
    out, cache = {}, {}
    for name, (m, w, scale) in masks.items():
        key = (m.tobytes(), None if w is None else w.tobytes())
        if key not in cache:
            p_use, inner_val = params, np.nan
            if tuned:
                p_use, inner_val = tune_policy(None, X[tr], y[tr], groups[tr], np.flatnonzero(m), w, n_trials, seed)

            def fit(rows):
                prep = FoldPrep(X[tr][rows])
                pred, _ = fit_model("xgb", p_use, prep.impute(X[tr][rows]), y[tr][rows], groups[tr][rows], seed,
                                    None if w is None else w[rows])
                return lambda Z: pred(prep.impute(Z))
            full = fit(m)
            part = fit(m & is_fit)
            cal = m & is_cal
            cache[key] = (full(X[te]), part(X[te]), np.abs(y[tr][cal] - part(X[tr][cal])), cal, p_use, inner_val)
        p_full, p_part, res_cal, cal, p_use, inner_val = cache[key]
        if scale is None:
            half = np.full(len(te), group_quantile(res_cal, groups[tr][cal], 1 - alpha))
        else:
            sfun = s2_main if scale == "main" else s2_shrunk
            s_cal, s_te = np.sqrt(sfun(rf[tr][cal], cf[tr][cal])), np.sqrt(sfun(rf[te], cf[te]))
            half = group_quantile(res_cal / s_cal, groups[tr][cal], 1 - alpha) * s_te
        ww = np.ones(int(m.sum())) if w is None else w[m]
        out[name] = {"pred": p_full, "lo": p_part - half, "hi": p_part + half, "n_train": int(m.sum()), "ess": eff_n(ww),
                     "w_min": float(ww.min()), "w_max": float(ww.max()), "w_cv": float(ww.std() / ww.mean()),
                     "inner_rmse": inner_val, "params": p_use}
    return out, {"beta": beta, **info}
