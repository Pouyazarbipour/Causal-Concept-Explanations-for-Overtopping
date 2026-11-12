"""Baseline learners and the leakage-safe nested cross-validation runner.

All models are evaluated on identical outer folds.  Hyper-parameters are selected on grouped inner
folds of the outer-training partition only; imputation and scaling are fitted on training rows only.
Every model returns predictions for the observed test record AND for all its counterfactual views.
"""
from __future__ import annotations

import json

import numpy as np
from joblib import Parallel, delayed

from .geometry_grouping import check_no_leakage, inner_group_folds
from .utils import out


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


# --------------------------------------------------------------------------- fit/predict
def _fit_predict(name, params, Xtr, ytr, gtr, Xte_views, seed, weight=None, monotone=None, mcfg=None):
    """Fit one model on the training partition, predict all test views [n, V]."""
    n, V, p = Xte_views.shape
    prep = FoldPrep(Xtr)
    flat = Xte_views.reshape(-1, p)
    if name == "rf":
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(random_state=seed, n_jobs=1, **params).fit(prep.impute(Xtr), ytr, sample_weight=weight)
        return m.predict(prep.impute(flat)).reshape(n, V)
    if name in ("xgb", "xgb_mono"):
        import xgboost as xgb
        kw = dict(params)
        if name == "xgb_mono":
            kw["monotone_constraints"] = tuple(int(v) for v in monotone)
        m = xgb.XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist", **kw).fit(prep.impute(Xtr), ytr, sample_weight=weight)
        return m.predict(prep.impute(flat)).reshape(n, V)
    if name == "mlp":
        torch = _torch()
        ymu, ysd = ytr.mean(), ytr.std()
        model, _ = train_mlp(prep.scale(Xtr), (ytr - ymu) / ysd, gtr, params, seed, mcfg["epochs"], mcfg["patience"],
                             mcfg["batch_size"], weight)
        with torch.no_grad():
            pz = model(torch.tensor(prep.scale(flat), dtype=torch.float32)).numpy()
        return (pz * ysd + ymu).reshape(n, V)
    raise ValueError(name)


def _outer_fold(name, grid, k, tr, te, X_views, y, groups, seeds, n_inner, base_seed, weight, monotone, mcfg):
    Xtr, ytr, gtr = X_views[tr, 0], y[tr], groups[tr]
    wtr = None if weight is None else weight[tr]
    scores = []
    if len(grid) > 1:
        for params in grid:  # inner grouped CV on the training partition only
            se, cnt = 0.0, 0
            for itr, iva in inner_group_folds(gtr, n_inner, base_seed + k):
                check_no_leakage(gtr[itr], gtr[iva])
                pv = _fit_predict(name, params, Xtr[itr], ytr[itr], gtr[itr], Xtr[iva][:, None, :], base_seed,
                                  None if wtr is None else wtr[itr], monotone, mcfg)[:, 0]
                se += float(((pv - ytr[iva]) ** 2).sum())
                cnt += len(iva)
            scores.append(np.sqrt(se / cnt))
        best = int(np.argmin(scores))
    else:
        best = 0
    preds = np.stack([_fit_predict(name, grid[best], Xtr, ytr, gtr, X_views[te], base_seed + s, wtr, monotone, mcfg)
                      for s in seeds])
    return k, te, preds, {"fold": int(k), "params": grid[best], "inner_rmse": [float(s) for s in scores]}


def run_cv(name, grid, X_views, y, fold, groups, *, seeds=(0,), n_inner=3, base_seed=0, train_mask=None,
           weight=None, monotone=None, mcfg=None, n_jobs=4, tag=None, save=True):
    """Nested CV.  Returns predictions [S, n, V] (NaN where a record is never in a test fold)."""
    n, V, _ = X_views.shape
    train_mask = np.ones(n, bool) if train_mask is None else train_mask
    tasks = []
    for k in np.unique(fold):
        te = np.flatnonzero(fold == k)
        tr = np.flatnonzero((fold != k) & train_mask)
        check_no_leakage(groups[tr], groups[te])
        tasks.append((k, tr, te))
    res = Parallel(n_jobs=n_jobs)(delayed(_outer_fold)(name, grid, k, tr, te, X_views, y, groups, seeds, n_inner,
                                                       base_seed, weight, monotone, mcfg) for k, tr, te in tasks)
    P = np.full((len(seeds), n, V), np.nan, dtype=np.float32)
    info = []
    for k, te, preds, inf in res:
        P[:, te] = preds
        info.append(inf)
    if save and tag:
        np.savez_compressed(out("predictions", f"{tag}.npz"), pred=P)
        with open(out("models", f"{tag}__hyperparameters.json"), "w") as fh:
            json.dump(info, fh, indent=1)
    return P, info
