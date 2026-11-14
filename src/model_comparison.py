"""Phase III (revision): baseline vs optimised models on identical outer folds and records.

Reads the original OOF predictions (results/predictions) and the v2 predictions (revision/outputs/models).
Reports RMSE, MAE, R2, bias, F10 (pooled OOF; per-seed mean and seed ensemble), geometry-level RMSE,
RMSE by RF and CF, and the paired geometry-cluster bootstrap difference to the baseline XGBoost.
Budget sensitivity (XGBoost v2): outer RMSE of the best configuration found within the first b trials.
Repeated grouped CV: baseline and v2 XGBoost on two further geometry partitions; v2 on series folds.
Outputs: revision/tables/M*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import load_model_data, load_pred
from .statistical_analysis import cluster_bootstrap
from .utils import ROOT

TAB = ROOT / "revision" / "tables"
V2 = ROOT / "revision" / "outputs" / "models"
NB = 1000
SEED = 20261112


def metrics(p, y):
    e = p - y
    rm = float(np.sqrt(np.mean(e ** 2)))
    return {"rmse": rm, "mae": float(np.mean(np.abs(e))), "r2": float(1 - np.mean(e ** 2) / np.var(y)), "bias": float(np.mean(e)), "f10": 10 ** rm}


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    D = load_model_data()
    df, y = D["df"], D["y"]
    geo = df["geometry"].to_numpy()
    rf, cf = df.RF.to_numpy(), df.CF.to_numpy()
    models = [("Random Forest (baseline)", load_pred("geometry__rf")), ("XGBoost (baseline)", load_pred("geometry__xgb")),
              ("CatBoost (baseline)", load_pred("geometry__catboost")), ("MLP (baseline)", load_pred("geometry__mlp")),
              ("CCW-Net (baseline)", load_pred("geometry__ccw_full"))]
    names = {"xgb": "XGBoost v2", "xgb_phys": "XGBoost v2 + physical features", "lgbm": "LightGBM v2", "hgb": "HistGradientBoosting v2",
             "et": "ExtraTrees v2", "catboost": "CatBoost v2", "rf": "Random Forest v2", "mlp": "MLP v2"}
    for k, lab in names.items():
        f = V2 / f"geometry__{k}.npz"
        if f.exists():
            models.append((lab, np.load(f)["pred"]))
    base = load_pred("geometry__xgb")[0]
    eb = (base - y) ** 2
    rows, cat = [], []
    for lab, P in models:
        ens = P.mean(0)
        per_seed = [metrics(p, y) for p in P]
        row = {"model": lab, "n_seeds": len(P), **{f"{k}": np.mean([m[k] for m in per_seed]) for k in per_seed[0]},
               "rmse_seed_sd": float(np.std([m["rmse"] for m in per_seed])), **{f"ensemble_{k}": v for k, v in metrics(ens, y).items()}}
        e1 = (ens - y) ** 2
        pt, lo, hi, _ = cluster_bootstrap(lambda i: [np.sqrt(e1[i].mean()), np.sqrt(e1[i].mean()) - np.sqrt(eb[i].mean())], geo, NB, SEED)
        row.update({"ensemble_rmse_lo": lo[0], "ensemble_rmse_hi": hi[0], "d_rmse_vs_xgb_baseline": pt[1], "d_lo": lo[1], "d_hi": hi[1],
                    "improvement_significant": bool(hi[1] < 0)})
        gm = pd.Series(ens - y).groupby(geo).mean()
        row["geometry_level_rmse_of_mean_residual"] = float(np.sqrt((gm ** 2).mean()))
        rows.append(row)
        for fac, arr in (("RF", rf), ("CF", cf)):
            for v in (1, 2, 3, 4):
                mk = arr == v
                cat.append({"model": lab, "factor": fac, "category": v, "n": int(mk.sum()), "rmse": float(np.sqrt(np.mean((ens[mk] - y[mk]) ** 2))),
                            "bias": float(np.mean(ens[mk] - y[mk]))})
    pd.DataFrame(rows).to_csv(TAB / "M1_model_comparison.csv", index=False)
    pd.DataFrame(cat).to_csv(TAB / "M2_model_rmse_by_rf_cf.csv", index=False)
    # budget sensitivity
    b = V2 / "geometry__xgb__budget.csv"
    if b.exists():
        B = pd.read_csv(b)
        P = np.load(V2 / "geometry__xgb.npz")["pred"][0]
        rr = []
        for bud, g in B.groupby("budget"):
            p = np.full(len(y), np.nan)
            p[g.idx.to_numpy()] = g.pred.to_numpy()
            rr.append({"budget_trials": bud, "inner_rmse_mean": g.groupby("fold").inner_rmse.first().mean(), "outer_rmse": float(np.sqrt(np.mean((p - y) ** 2)))})
        best = pd.read_csv(V2 / "geometry__xgb__best.csv")
        rr.append({"budget_trials": 100, "inner_rmse_mean": best.inner_cv_rmse.mean(), "outer_rmse": float(np.sqrt(np.mean((P - y) ** 2)))})
        o = pd.read_csv(ROOT / "results/models/optuna/geometry__xgb__best.csv")
        rr.append({"budget_trials": "30 (baseline space)", "inner_rmse_mean": o.inner_cv_rmse.mean(), "outer_rmse": float(np.sqrt(np.mean((base - y) ** 2)))})
        pd.DataFrame(rr).to_csv(TAB / "M3_xgb_budget_sensitivity.csv", index=False)
    # repeated partitions / series
    rr = [{"partition": "primary geometry partition", "model": "XGBoost baseline", **metrics(base, y)}]
    if (V2 / "geometry__xgb.npz").exists():
        rr.append({"partition": "primary geometry partition", "model": "XGBoost v2 (seed 0)", **metrics(np.load(V2 / "geometry__xgb.npz")["pred"][0], y)})
    for s in (1, 2):
        for ver in (1, 2):
            f = V2 / f"geometry_alt{s}__xgb_v{ver}.npz"
            if f.exists():
                rr.append({"partition": f"alternative geometry partition {s}", "model": f"XGBoost {'baseline' if ver == 1 else 'v2'}", **metrics(np.load(f)["pred"][0], y)})
    rr.append({"partition": "series held out", "model": "XGBoost baseline", **metrics(load_pred("series__xgb")[0], y)})
    if (V2 / "series__xgb.npz").exists():
        rr.append({"partition": "series held out", "model": "XGBoost v2 (seed 0)", **metrics(np.load(V2 / "series__xgb.npz")["pred"][0], y)})
    pd.DataFrame(rr).to_csv(TAB / "M4_repeated_partitions.csv", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows)[["model", "rmse", "ensemble_rmse", "r2", "d_rmse_vs_xgb_baseline", "d_lo", "d_hi"]].round(4).to_string())


if __name__ == "__main__":
    main()
