"""Revised (v2) model optimisation: wider search spaces, larger budgets, pruning, additional learners.

This module ADDS to `src/baseline_models.py`, which is left unchanged so that the original
results remain reproducible.  The evaluation protocol is identical to the original one:

* the same pre-specified outer geometry-grouped folds (data/processed/fold_assignments.csv);
* all 9 333 positive-discharge records, RF and CF 1-4 retained, equal weights;
* Optuna (TPE) on grouped inner folds of the outer-training partition only, with median pruning
  after each inner fold;
* imputation fitted on training rows only; boosted-tree size by early stopping on inner folds;
* no outer-test information in any choice.  The outer RMSE is computed once, after the fact.

Physically motivated features (`phys` feature set) are deterministic functions of the documented
predictors (no target information): the Iribarren number xi0 = 1/(cot(alpha) sqrt(s0)) with cot(alpha) the mean of the lower and upper slopes, the
roughness-scaled freeboard Rc/(Hm0 gamma_f), the EurOtop-type breaking-wave freeboard
Rc/(Hm0 xi0 gamma_f) and the obliquity factor gamma_beta = 1 - 0.0033 min(|beta|, 80).
"""
from __future__ import annotations

import numpy as np

from .baseline_models import FoldPrep, fit_model as fit_model_v1
from .geometry_grouping import check_no_leakage, inner_group_folds

BUDGET = {"xgb": 100, "xgb_phys": 100, "catboost": 40, "lgbm": 80, "hgb": 40, "et": 30, "rf": 40, "mlp": 24}
TREE_KEY = {"xgb": "n_estimators", "xgb_phys": "n_estimators", "catboost": "iterations", "lgbm": "n_estimators"}
MAX_TREES = {"xgb": 4000, "xgb_phys": 4000, "catboost": 3000, "lgbm": 4000}
ES = 100


def phys_features(df) -> np.ndarray:
    """Additional deterministic physical predictors (raw, untransformed ratios)."""
    H = df["hm0_toe"].to_numpy(float)
    L0 = 9.81 * df["tm10_toe"].to_numpy(float) ** 2 / (2 * np.pi)
    s0 = H / L0
    # tan(alpha_B) in CLASH is the berm slope (0 for 97 % of records), so the Iribarren number uses the
    # mean of the lower and upper structure slopes; undefined for vertical structures (cot = 0)
    cot = 0.5 * (df["cotad"].to_numpy(float) + df["cotau"].to_numpy(float))
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = np.where((cot > 0) & (s0 > 0), 1.0 / (cot * np.sqrt(s0)), np.nan)
    gf = df["gf_raw"].to_numpy(float)
    rch = df["rc"].to_numpy(float) / H
    gb = 1.0 - 0.0033 * np.minimum(np.abs(df["beta_deg"].to_numpy(float)), 80.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cols = [np.log(xi), rch / gf, rch / (xi * gf), gb, rch / (gf * gb)]
    X = np.column_stack(cols)
    X[~np.isfinite(X)] = np.nan
    return X


def suggest(name, trial):
    if name in ("xgb", "xgb_phys"):
        return {"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 50.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "colsample_bynode": trial.suggest_float("colsample_bynode", 0.3, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 50.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0),
                "max_bin": trial.suggest_categorical("max_bin", [128, 256, 512]),
                "objective": trial.suggest_categorical("objective", ["reg:squarederror", "reg:pseudohubererror"])}
    if name == "catboost":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "depth": trial.suggest_int("depth", 3, 10),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.5, 50.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
                "rsm": trial.suggest_float("rsm", 0.4, 1.0),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
                "border_count": trial.suggest_categorical("border_count", [128, 254])}
    if name == "lgbm":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 14),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0), "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)}
    if name == "hgb":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 100, log=True),
                "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
                "max_features": trial.suggest_float("max_features", 0.4, 1.0)}
    if name in ("et", "rf"):
        return {"n_estimators": 500, "max_features": trial.suggest_float("max_features", 0.2, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                "max_depth": trial.suggest_int("max_depth", 6, 40)}
    if name == "mlp":
        return {"width": trial.suggest_categorical("width", [64, 128, 256, 512]), "depth": trial.suggest_int("depth", 2, 6),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3),
                "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}
    raise ValueError(name)


def fit(name, params, Xtr, ytr, gtr, seed, eval_set=None, mcfg=None):
    if name in ("xgb", "xgb_phys"):
        import xgboost as xgb
        kw = dict(params)
        n = kw.pop("n_estimators", MAX_TREES["xgb"])
        if eval_set is not None:
            m = xgb.XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", n_estimators=n, early_stopping_rounds=ES, **kw)
            m.fit(Xtr, ytr, eval_set=[eval_set], verbose=False)
            return m.predict, {"n_trees": int(m.best_iteration) + 1}
        m = xgb.XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", n_estimators=n, **kw).fit(Xtr, ytr)
        return m.predict, {}
    if name == "catboost":
        from catboost import CatBoostRegressor
        kw = dict(params)
        n = kw.pop("iterations", MAX_TREES["catboost"])
        m = CatBoostRegressor(random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False, iterations=n, **kw)
        if eval_set is not None:
            m.fit(Xtr, ytr, eval_set=eval_set, early_stopping_rounds=ES)
            return m.predict, {"n_trees": int(m.get_best_iteration() or n - 1) + 1}
        m.fit(Xtr, ytr)
        return m.predict, {}
    if name == "lgbm":
        import lightgbm as lgb
        kw = dict(params)
        n = kw.pop("n_estimators", MAX_TREES["lgbm"])
        m = lgb.LGBMRegressor(random_state=seed, n_jobs=1, n_estimators=n, verbose=-1, **kw)
        if eval_set is not None:
            m.fit(Xtr, ytr, eval_set=[eval_set], callbacks=[lgb.early_stopping(ES, verbose=False)])
            return m.predict, {"n_trees": int(m.best_iteration_ or n)}
        m.fit(Xtr, ytr)
        return m.predict, {}
    if name == "hgb":       # internal early stopping on a random 10 % of the (inner) training rows
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=2000, early_stopping=True, validation_fraction=0.1, n_iter_no_change=50,
                                          random_state=seed, **params).fit(Xtr, ytr)
        return m.predict, {}
    if name == "et":
        from sklearn.ensemble import ExtraTreesRegressor
        m = ExtraTreesRegressor(random_state=seed, n_jobs=1, **params).fit(Xtr, ytr)
        return m.predict, {}
    if name in ("rf", "mlp"):
        return fit_model_v1(name, params, Xtr, ytr, gtr, seed, mcfg=mcfg)
    raise ValueError(name)


def tune(name, Xtr, ytr, gtr, n_trials, n_inner, seed, mcfg=None):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = inner_group_folds(gtr, n_inner, seed)
    tk = TREE_KEY.get(name)

    def objective(trial):
        params = suggest(name, trial)
        se, cnt, trees = 0.0, 0, []
        for step, (itr, iva) in enumerate(folds):
            check_no_leakage(gtr[itr], gtr[iva])
            prep = FoldPrep(Xtr[itr])
            A, B = prep.impute(Xtr[itr]), prep.impute(Xtr[iva])
            pred, info = fit(name, params, A, ytr[itr], gtr[itr], seed, eval_set=(B, ytr[iva]) if tk else None, mcfg=mcfg)
            se += float(((pred(B) - ytr[iva]) ** 2).sum())
            cnt += len(iva)
            trees.append(info.get("n_trees", 0))
            trial.report(float(np.sqrt(se / cnt)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if tk:
            trial.set_user_attr("n_trees", int(np.ceil(1.1 * np.mean(trees))))
        return float(np.sqrt(se / cnt))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0))
    study.optimize(objective, n_trials=n_trials)
    return study


def params_of(name, trial):
    p = suggest(name, __import__("optuna").trial.FixedTrial(trial.params))
    if TREE_KEY.get(name):
        p[TREE_KEY[name]] = trial.user_attrs["n_trees"]
    return p


def best_within(study, n):
    """Best completed trial among the first n (budget-sensitivity analysis; inner objective only)."""
    done = [t for t in study.trials if t.number < n and t.value is not None and t.state.name == "COMPLETE"]
    return min(done, key=lambda t: t.value)


def outer_fold(name, k, tr, te, X, y, groups, seeds, n_trials, n_inner, base_seed, mcfg, cache_dir, budgets=()):
    """Tune on outer-training rows, refit with each seed, predict held-out rows; budget checkpoints refit too."""
    import pickle
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{name}__fold{k}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    check_no_leakage(groups[tr], groups[te])
    study = tune(name, X[tr], y[tr], groups[tr], n_trials, n_inner, base_seed + int(k), mcfg)
    prep = FoldPrep(X[tr])
    A, B = prep.impute(X[tr]), prep.impute(X[te])
    best = params_of(name, study.best_trial)
    preds = [fit(name, best, A, y[tr], groups[tr], base_seed + s, mcfg=mcfg)[0](B) for s in seeds]
    budget_pred = {}
    for b in budgets:          # refit the best configuration found within the first b trials (seed 0 only)
        if b < n_trials:
            t = best_within(study, b)
            budget_pred[b] = (t.value, fit(name, params_of(name, t), A, y[tr], groups[tr], base_seed, mcfg=mcfg)[0](B))
    df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs", "duration"))
    res = {"fold": int(k), "te": te, "pred": np.stack(preds), "best": best, "inner_rmse": float(study.best_value),
           "trials": df, "budget_pred": budget_pred}
    tmp = f.with_suffix(".tmp")
    pickle.dump(res, open(tmp, "wb"))
    tmp.rename(f)
    return res
