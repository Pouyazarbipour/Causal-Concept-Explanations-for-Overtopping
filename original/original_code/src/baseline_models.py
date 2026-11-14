"""Baseline learners and the leakage-safe nested cross-validation runner.

The regression models are diagnostic tools for the data-quality study, not its contribution.
All models are evaluated on identical outer folds.  Hyper-parameters are selected by Optuna (TPE) on
grouped inner folds of the outer-training partition only; boosted-tree size is set by early stopping;
imputation and scaling are fitted on training rows only.
"""
from __future__ import annotations

import numpy as np

from .geometry_grouping import check_no_leakage, inner_group_folds


class FoldPrep:
    """Median imputation + standardisation fitted on training rows only."""

    def __init__(self, X_train: np.ndarray):
        self.med = np.nanmedian(X_train, axis=0)
        Xi = np.where(np.isnan(X_train), self.med, X_train)
        self.mu, self.sd = Xi.mean(0), Xi.std(0)
        self.sd[self.sd < 1e-12] = 1.0

    def impute(self, X):
        return np.where(np.isnan(X), self.med, X)

    def scale(self, X):
        return (self.impute(X) - self.mu) / self.sd


# --------------------------------------------------------------------------- torch MLP
def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


def make_mlp(p: int, width: int, depth: int, dropout: float, latent: int | None = None):
    torch = _torch()
    nn = torch.nn
    layers, d = [], p
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU(), nn.Dropout(dropout)]
        d = width
    if latent:
        layers += [nn.Linear(d, latent), nn.SiLU()]
        d = latent

    class TabMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.body = nn.Sequential(*layers)
            self.out = nn.Linear(d, 1)
            self.latent_dim = d

        def latent(self, x):
            return self.body(x)

        def forward(self, x):
            return self.out(self.body(x)).squeeze(-1)

    return TabMLP()


def val_split(groups: np.ndarray, seed: int, frac: float = 0.2):
    """Grouped train/validation split for early stopping (training partition only)."""
    uniq = np.unique(groups)
    rng = np.random.RandomState(seed)
    val_g = set(rng.choice(uniq, max(1, int(round(frac * len(uniq)))), replace=False))
    is_val = np.array([g in val_g for g in groups])
    return ~is_val, is_val


def train_mlp(Xtr, ytr, groups, params, seed, epochs=150, patience=20, batch_size=256, weight=None, latent=None):
    """Train a tabular MLP on standardised inputs/target.  Returns (model, history)."""
    torch = _torch()
    torch.manual_seed(seed)
    tr, va = val_split(groups, seed)
    model = make_mlp(Xtr.shape[1], params["width"], params["depth"], params["dropout"], latent)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    X, y = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    w = torch.ones(len(y)) if weight is None else torch.tensor(weight, dtype=torch.float32)
    itr, iva = torch.tensor(np.flatnonzero(tr)), torch.tensor(np.flatnonzero(va))
    best, best_state, bad, hist = np.inf, None, 0, []
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train()
        perm = itr[torch.randperm(len(itr), generator=g)]
        for s in range(0, len(perm), batch_size):
            b = perm[s:s + batch_size]
            loss = (w[b] * (model(X[b]) - y[b]) ** 2).sum() / w[b].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = float(((model(X[iva]) - y[iva]) ** 2).mean())
        hist.append(v)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


# --------------------------------------------------------------------------- Optuna search spaces
def suggest(name: str, trial) -> dict:
    """Justified search spaces (9 000 records, 15 predictors, noisy target, grouped generalisation)."""
    if name == "rf":      # variance reduction through feature sub-sampling and leaf size; depth left free or capped
        return {"n_estimators": 300, "max_features": trial.suggest_float("max_features", 0.2, 1.0),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_depth": trial.suggest_int("max_depth", 6, 30)}
    if name == "xgb":     # number of trees is set by early stopping, not searched
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0)}
    if name == "catboost":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 9),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
                "rsm": trial.suggest_float("rsm", 0.5, 1.0)}
    if name == "mlp":
        return {"width": trial.suggest_categorical("width", [64, 128, 256]), "depth": trial.suggest_int("depth", 2, 5),
                "dropout": trial.suggest_float("dropout", 0.0, 0.2),
                "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),
                "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)}
    raise ValueError(name)


MAX_TREES = {"xgb": 2000, "catboost": 1500}
ES_ROUNDS = 50


# --------------------------------------------------------------------------- fit/predict
def fit_model(name, params, Xtr, ytr, gtr, seed, weight=None, eval_set=None, mcfg=None):
    """Fit one model on already-imputed training data.  Returns (predict_fn, info).

    Boosted trees: with `eval_set` the number of trees is chosen by early stopping on that set and
    returned in info["n_trees"]; without it, params must contain the number of trees."""
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(random_state=seed, n_jobs=1, **params).fit(Xtr, ytr, sample_weight=weight)
        return m.predict, {}
    if name == "xgb":
        import xgboost as xgb
        kw = dict(params)
        n = kw.pop("n_estimators", MAX_TREES["xgb"])
        if eval_set is not None:
            m = xgb.XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", n_estimators=n, early_stopping_rounds=ES_ROUNDS, **kw)
            m.fit(Xtr, ytr, sample_weight=weight, eval_set=[eval_set], verbose=False)
            return m.predict, {"n_trees": int(m.best_iteration) + 1}
        m = xgb.XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", n_estimators=n, **kw).fit(Xtr, ytr, sample_weight=weight)
        return m.predict, {"model": m}
    if name == "catboost":
        from catboost import CatBoostRegressor
        kw = dict(params)
        n = kw.pop("iterations", MAX_TREES["catboost"])
        m = CatBoostRegressor(random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False, iterations=n, **kw)
        if eval_set is not None:
            m.fit(Xtr, ytr, sample_weight=weight, eval_set=eval_set, early_stopping_rounds=ES_ROUNDS)
            return m.predict, {"n_trees": int(m.get_best_iteration() or n - 1) + 1}
        m.fit(Xtr, ytr, sample_weight=weight)
        return m.predict, {}
    if name == "mlp":      # early stopping on a grouped validation split of the training data (inside train_mlp)
        torch = _torch()
        mu, sd = Xtr.mean(0), Xtr.std(0)
        sd[sd < 1e-12] = 1.0
        ymu, ysd = ytr.mean(), ytr.std()
        model, hist = train_mlp((Xtr - mu) / sd, (ytr - ymu) / ysd, gtr, params, seed, mcfg["epochs"], mcfg["patience"],
                                mcfg["batch_size"], weight)

        def predict(X):
            with torch.no_grad():
                return model(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy() * ysd + ymu
        return predict, {"epochs": len(hist)}
    raise ValueError(name)


def tree_key(name):
    return {"xgb": "n_estimators", "catboost": "iterations"}.get(name)


def tune_fold(name, Xtr, ytr, gtr, n_trials, n_inner, seed, mcfg=None):
    """Optuna (TPE) on grouped inner folds of the training partition only.  Returns (best params incl.
    number of trees, trials dataframe)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = inner_group_folds(gtr, n_inner, seed)

    def objective(trial):
        params = suggest(name, trial)
        se, cnt, trees = 0.0, 0, []
        for itr, iva in folds:
            check_no_leakage(gtr[itr], gtr[iva])
            prep = FoldPrep(Xtr[itr])
            A, B = prep.impute(Xtr[itr]), prep.impute(Xtr[iva])
            es = (B, ytr[iva]) if tree_key(name) else None
            pred, info = fit_model(name, params, A, ytr[itr], gtr[itr], seed, eval_set=es, mcfg=mcfg)
            se += float(((pred(B) - ytr[iva]) ** 2).sum())
            cnt += len(iva)
            trees.append(info.get("n_trees", 0))
        if tree_key(name):
            trial.set_user_attr("n_trees", int(np.ceil(1.1 * np.mean(trees))))
        return float(np.sqrt(se / cnt))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = suggest(name, optuna.trial.FixedTrial(study.best_params))
    if tree_key(name):
        best[tree_key(name)] = study.best_trial.user_attrs["n_trees"]
    df = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "duration"))
    return best, df, float(study.best_value)


def outer_fold(name, scheme, k, tr, te, X, y, groups, seeds, n_trials, n_inner, base_seed, mcfg, cache_dir, want_shap=False):
    """One outer fold: tune on training rows, refit, predict held-out rows.  Checkpointed on disk."""
    import pickle
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{scheme}__{name}__fold{k}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    check_no_leakage(groups[tr], groups[te])
    best, trials, val = tune_fold(name, X[tr], y[tr], groups[tr], n_trials, n_inner, base_seed + int(k), mcfg)
    prep = FoldPrep(X[tr])
    A, B = prep.impute(X[tr]), prep.impute(X[te])
    preds, shap = [], None
    for s in seeds:
        pred, info = fit_model(name, best, A, y[tr], groups[tr], base_seed + s, mcfg=mcfg)
        preds.append(pred(B))
        if want_shap and name == "xgb" and shap is None:   # TreeSHAP of the held-out records (comparator only)
            import xgboost as xgb
            shap = info["model"].get_booster().predict(xgb.DMatrix(B), pred_contribs=True)
    out_ = {"fold": int(k), "te": te, "pred": np.stack(preds), "best": best, "inner_rmse": val, "trials": trials, "shap": shap}
    tmp = f.with_suffix(".tmp")
    pickle.dump(out_, open(tmp, "wb"))
    tmp.rename(f)
    return out_
