"""Cluster bootstrap, multiple-comparison correction, ordinal trend and agreement statistics."""
from __future__ import annotations

import numpy as np
from scipy import stats


def cluster_bootstrap(stat_fn, clusters, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    """Percentile CI for `stat_fn(idx)` resampling whole clusters with replacement.

    `stat_fn` receives an integer index array (with repeats) and returns a scalar or 1-D array.
    """
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.RandomState(seed)
    point = np.atleast_1d(np.asarray(stat_fn(np.arange(len(clusters))), float))
    reps = np.full((n_boot, len(point)), np.nan)
    for b in range(n_boot):
        pick = rng.randint(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        try:
            reps[b] = np.atleast_1d(stat_fn(idx))
        except Exception:  # noqa: BLE001  (a resample may lack a category)
            pass
    lo = np.nanpercentile(reps, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(reps, 100 * (1 - alpha / 2), axis=0)
    return point, lo, hi, reps


def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def jonckheere(values, groups) -> tuple[float, float]:
    """Jonckheere-Terpstra trend statistic (standardised) for ordered groups; normal approximation
    without tie correction.  NOTE: the p-value assumes independent observations and is reported only
    as a descriptive companion to the cluster-bootstrap intervals."""
    values, groups = np.asarray(values, float), np.asarray(groups)
    levels = np.sort(np.unique(groups))
    samples = [values[groups == g] for g in levels]
    J = 0.0
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            u = stats.mannwhitneyu(samples[j], samples[i], alternative="two-sided").statistic
            J += u
    n = np.array([len(s) for s in samples], float)
    N = n.sum()
    mean = (N ** 2 - np.sum(n ** 2)) / 4
    var = (N ** 2 * (2 * N + 3) - np.sum(n ** 2 * (2 * n + 3))) / 72
    z = (J - mean) / np.sqrt(var)
    return float(z), float(2 * stats.norm.sf(abs(z)))


def weighted_kappa(a, b, categories, weights: str = "quadratic") -> float:
    """Cohen's weighted kappa for two paired ordinal ratings."""
    cats = list(categories)
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    i, j = np.indices((k, k))
    W = (np.abs(i - j) / (k - 1)) ** (2 if weights == "quadratic" else 1)
    den = (W * E).sum()
    return float(1 - (W * O).sum() / den) if den > 0 else np.nan


def weighted_quantile(values, weights, q: float) -> float:
    """Smallest v with cumulative weight >= q (weights need not be normalised; may include +inf value)."""
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values)
    cw = np.cumsum(weights[order]) / weights.sum()
    return float(values[order][min(np.searchsorted(cw, q, side="left"), len(values) - 1)])
