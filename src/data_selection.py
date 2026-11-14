"""Counterfactual auditing of data-selection policies.

The question is counterfactual in the TRAINING DATA, not in a physical input: "for the same held-out
tests, what would the prediction and its interval have been had the model been developed under another
retention / weighting policy?"  Everything else is held fixed: outer folds, learner, hyper-parameters
(tuned on the full training partition of the fold), random seeds and the calibration split.

Policies
  P1  retain all records (RF 1-4, CF 1-4), equal weights
  P2  exclude RF = 4
  P3  exclude CF = 4
  P4  exclude RF = 4 or CF = 4
  P5  conventional CLASH weighting WF = (4-RF)(4-CF)  (zero weight for category 4)
  P5b conventional practice: P5 and removal of records remarked "do not use for neural network"
  P6a retain all; empirical inverse-variance weights 1/sigma^2(RF, CF)
  P6b retain all, equal weights; RF/CF-aware interval scale sigma(RF, CF)   (mean model identical to P1)
  P6c P6a weights and P6b interval scale
sigma^2(RF, CF) is a main-effects log-linear model of grouped inner out-of-fold squared residuals, fitted
on the training partition of each outer fold only.
"""
from __future__ import annotations

import numpy as np

from .baseline_models import FoldPrep, fit_model, val_split
from .geometry_grouping import check_no_leakage, inner_group_folds
from .uncertainty_modeling import group_quantile

POLICIES = {
    "P1_retain_all": "Retain all records, equal weights",
    "P2_excl_RF4": "Exclude RF = 4",
    "P3_excl_CF4": "Exclude CF = 4",
    "P4_excl_RF4_or_CF4": "Exclude RF = 4 or CF = 4",
    "P5_CLASH_weights": "Conventional weights (4-RF)(4-CF)",
    "P5b_CLASH_weights_and_remarks": "Conventional weights and removal of 'do not use' records",
    "P6a_inverse_variance_weights": "Retain all, empirical weights 1/sigma^2(RF,CF)",
    "P6b_quality_aware_intervals": "Retain all, equal weights, RF/CF-aware interval width",
    "P6c_weights_and_intervals": "Retain all, empirical weights and RF/CF-aware interval width",
}


def design_q(rf, cf):
    return np.column_stack([np.ones(len(rf))] + [(rf == v).astype(float) for v in (2, 3, 4)] + [(cf == v).astype(float) for v in (2, 3, 4)])


def fit_variance_model(r, rf, cf):
    """log sigma^2 = a + RF category + CF category  (least squares on log squared residuals)."""
    beta = np.linalg.lstsq(design_q(rf, cf), np.log(r ** 2 + 1e-4), rcond=None)[0]
    return beta


def sigma2(beta, rf, cf):
    return np.exp(design_q(rf, cf) @ beta)


def policy_fold(learner, params, X, y, rf, cf, dnu, groups, tr, te, seed, calib_frac, alpha=0.1):
    """All policies on one outer fold.  Returns {policy: {"pred", "lo", "hi", "n_train"}} and the variance model."""
    check_no_leakage(groups[tr], groups[te])
    # grouped inner out-of-fold residuals of the full training partition (basis of the empirical policy)
    oof = np.zeros(len(tr))
    for a, b in inner_group_folds(groups[tr], 3, seed):
        prep = FoldPrep(X[tr][a])
        pred, _ = fit_model(learner, params, prep.impute(X[tr][a]), y[tr][a], groups[tr][a], seed)
        oof[b] = pred(prep.impute(X[tr][b]))
    beta = fit_variance_model(y[tr] - oof, rf[tr], cf[tr])
    w_emp = 1.0 / sigma2(beta, rf[tr], cf[tr])
    w_emp /= w_emp.mean()
    rf4, cf4 = rf[tr] == 4, cf[tr] == 4
    clash = ((4 - rf[tr]) * (4 - cf[tr])).astype(float)
    spec = {  # policy -> (row mask on the training partition, weights or None, quality-aware scale?)
        "P1_retain_all": (np.ones(len(tr), bool), None, False),
        "P2_excl_RF4": (~rf4, None, False),
        "P3_excl_CF4": (~cf4, None, False),
        "P4_excl_RF4_or_CF4": (~rf4 & ~cf4, None, False),
        "P5_CLASH_weights": (clash > 0, clash, False),
        "P5b_CLASH_weights_and_remarks": ((clash > 0) & ~dnu[tr], clash, False),
        "P6a_inverse_variance_weights": (np.ones(len(tr), bool), w_emp, False),
        "P6b_quality_aware_intervals": (np.ones(len(tr), bool), None, True),
        "P6c_weights_and_intervals": (np.ones(len(tr), bool), w_emp, True),
    }
    is_fit, is_cal = val_split(groups[tr], seed, calib_frac)   # the same calibration geometries for every policy
    s_te = np.sqrt(sigma2(beta, rf[te], cf[te]))
    s_tr = np.sqrt(sigma2(beta, rf[tr], cf[tr]))
    out, cache = {}, {}
    for name, (m, w, scaled) in spec.items():
        key = (m.tobytes(), None if w is None else w.tobytes())
        if key not in cache:
            def fit(rows):
                prep = FoldPrep(X[tr][rows])
                pred, _ = fit_model(learner, params, prep.impute(X[tr][rows]), y[tr][rows], groups[tr][rows], seed,
                                    None if w is None else w[rows])
                return lambda Z: pred(prep.impute(Z))
            full = fit(m)                       # accuracy: model developed on all policy-eligible training rows
            part = fit(m & is_fit)              # intervals: proper-training part, calibrated on the policy's own
            cal = m & is_cal                    # calibration rows (an excluded category is absent from calibration)
            cache[key] = (full(X[te]), part(X[te]), np.abs(y[tr][cal] - part(X[tr][cal])), cal)
        p_full, p_part, res_cal, cal = cache[key]
        if scaled:
            q = group_quantile(res_cal / s_tr[cal], groups[tr][cal], 1 - alpha)
            half = q * s_te
        else:
            half = np.full(len(te), group_quantile(res_cal, groups[tr][cal], 1 - alpha))
        out[name] = {"pred": p_full, "lo": p_part - half, "hi": p_part + half, "n_train": int(m.sum())}
    return out, beta
