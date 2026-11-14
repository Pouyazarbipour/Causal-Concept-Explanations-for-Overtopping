"""Phase V (revision): stability of the reconstructed geometries and of the results that depend on them.

The primary key rounds the water-level-invariant descriptors to 3 significant digits within a series.
Here the key is rebuilt with 2 and 4 significant digits (and the existing coarse key is included).
For each definition: number of geometries, agreement with the primary partition (adjusted Rand index,
share of primary geometries reproduced exactly), out-of-fold XGBoost RMSE on folds built from that
definition (per-fold hyper-parameters of the original run, fixed), REML variance shares, and the RF = 4
variance ratio.  Outputs: revision/tables/G*.csv
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GroupKFold

from .baseline_models import FoldPrep
from .error_decomposition import decompose
from .geometry_grouping import add_groups, check_no_leakage
from .pipeline import load_model_data
from .rf_cf_validation import variance_model
from .utils import ROOT

TAB = ROOT / "revision" / "tables"


def folds_for(groups, seed, n=5):
    uniq = np.unique(groups)
    perm = dict(zip(uniq, np.random.RandomState(seed).permutation(len(uniq))))
    gi = np.array([perm[v] for v in groups])
    f = np.full(len(groups), -1)
    for k, (_, te) in enumerate(GroupKFold(n).split(gi, groups=gi)):
        f[te] = k
    return f


def oof_xgb(X, y, fold, groups, seed):
    import xgboost as xgb
    pred = np.full(len(y), np.nan)
    for k in range(5):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        check_no_leakage(groups[tr], groups[te])
        best = pickle.load(open(ROOT / "results/models/fold_cache" / f"geometry__xgb__fold{k}.pkl", "rb"))["best"]
        prep = FoldPrep(X[tr])
        pred[te] = xgb.XGBRegressor(n_jobs=4, tree_method="hist", random_state=seed, **best).fit(prep.impute(X[tr]), y[tr]).predict(prep.impute(X[te]))
    return pred


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    D = load_model_data()
    df, X, y = D["df"], D["X"], D["y"]
    seed = D["cfg"]["experiment"]["seed"]
    base = df["geometry"].to_numpy()
    rows = []
    for lab, key in (("3 significant digits (primary)", None), ("2 significant digits", 2), ("4 significant digits", 4), ("coarse key (slopes, roughness, berm/toe presence)", "coarse")):
        if key is None:
            g = base
        elif key == "coarse":
            g = df["geometry_coarse"].to_numpy()
        else:
            g = add_groups(D["master"], key).loc[D["master"].use_regression, "geometry"].to_numpy()
        same = pd.Series(base).groupby(base).apply(lambda s: len(np.unique(g[s.index])) == 1)
        exact = pd.Series(g).groupby(g).apply(lambda s: len(np.unique(base[s.index])) == 1)
        fold = D["folds"]["geometry"].to_numpy() if key is None else folds_for(g, seed)
        pred = oof_xgb(X, y, fold, g, seed)
        r = y - pred
        d = df.assign(geometry=g)
        dec, _ = decompose(r, d, ["series", "geometry"], 0, seed)
        vm = variance_model(d, r, "series", 300, seed, interaction=False).set_index("term")
        rows.append({"definition": lab, "n_geometries": len(np.unique(g)), "adjusted_rand_index_vs_primary": adjusted_rand_score(base, g),
                     "pct_primary_geometries_not_split": 100 * float(same.mean()), "pct_geometries_not_merging_primary": 100 * float(exact.mean()),
                     "oof_rmse_xgb_fixed_params": float(np.sqrt(np.mean(r ** 2))),
                     "share_series": float(dec.share.iloc[0]), "share_geometry": float(dec.share.iloc[1]), "share_within": float(dec.share.iloc[2]),
                     "RF4_variance_ratio": vm.loc["RF=4", "variance_ratio"], "RF4_lo": vm.loc["RF=4", "ratio_lo"], "RF4_hi": vm.loc["RF=4", "ratio_hi"],
                     "CF3_variance_ratio": vm.loc["CF=3", "variance_ratio"], "CF4_variance_ratio": vm.loc["CF=4", "variance_ratio"]})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(TAB / "G1_geometry_definition_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
