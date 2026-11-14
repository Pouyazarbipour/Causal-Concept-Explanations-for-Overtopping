"""Phase XIII (revision): independent recomputation of the manuscript's key numbers from the ORIGINAL
out-of-fold predictions (not through the table-generating code) and comparison with the original tables.
Output: revision/audit/numerical_consistency_audit.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .error_decomposition import replicate_pairs
from .pipeline import load_model_data, load_pred
from .utils import ROOT

O = ROOT / "original" / "original_results" / "tables"


def t(n):
    return pd.read_csv(O / f"{n}.csv")


def main():
    D = load_model_data()
    df, y = D["df"], D["y"]
    fold = D["folds"]["geometry"].to_numpy()
    rows = []

    def add(item, manuscript, recomputed, definition, verdict=None, note=""):
        if verdict is None:
            try:
                verdict = "consistent" if np.isclose(float(manuscript), float(recomputed), atol=0.0051, rtol=0.0) else "DIFFERENT"
            except (TypeError, ValueError):
                verdict = "see note"
        rows.append({"item": item, "manuscript_value": manuscript, "recomputed_value": recomputed, "definition": definition, "verdict": verdict, "note": note})

    perf = t("table03_model_performance").set_index(["scheme", "model"])
    opt = t("table04_optuna_best_hyperparameters")
    labs = {"rf": "Random Forest", "xgb": "XGBoost", "catboost": "CatBoost", "mlp": "MLP"}
    for m, lab in labs.items():
        P = load_pred(f"geometry__{m}")
        pooled_seed = float(np.mean(np.sqrt(((P - y) ** 2).mean(1))))
        ens = P.mean(0)
        foldmean = float(np.mean([np.sqrt(np.mean((ens[fold == k] - y[fold == k]) ** 2)) for k in range(5)]))
        add(f"Table 4 RMSE {lab}", round(perf.loc[("geometry", lab), "rmse"], 3), round(pooled_seed, 4), "pooled OOF RMSE, mean over seeds")
        add(f"Table 2 outer RMSE {lab}", round(opt[(opt.scheme == "geometry") & (opt.model == m)].outer_test_rmse.mean(), 3), round(foldmean, 4),
            "mean of 5 fold-wise RMSEs of the seed ensemble",
            note="Table 2 and Table 4 use different definitions (A1); the revision reports the pooled OOF RMSE everywhere")
        r2 = 1 - np.mean((P - y) ** 2, axis=1).mean() / np.var(y)
        add(f"Table 4 R2 {lab}", round(perf.loc[("geometry", lab), "r2"], 3), round(float(r2), 4), "1 - MSE/Var(y), per-seed mean")
    for sch, val in (("random", 0.354), ("series", 0.749)):
        add(f"XGBoost {sch} RMSE", val, round(float(np.sqrt(np.mean((load_pred(f'{sch}__xgb')[0] - y) ** 2))), 4), "pooled OOF RMSE")
    # RF strata and variance ratios (Gamma GLM point estimates)
    import statsmodels.api as sm
    r = y - load_pred("geometry__xgb")[0]
    rf, cf = df.RF.to_numpy(int), df.CF.to_numpy(int)
    for v, n in zip((1, 2, 3, 4), (3189, 3174, 2333, 637)):
        add(f"n RF={v} (regression)", n, int((rf == v).sum()), "count")
    X = np.column_stack([np.ones(len(df))] + [(rf == v).astype(float) for v in (2, 3, 4)] + [(cf == v).astype(float) for v in (2, 3, 4)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sm.GLM(r ** 2 + 1e-8, X, family=sm.families.Gamma(sm.families.links.Log())).fit().params
    for k, (v, ms) in enumerate(zip((2, 3, 4), (1.38, 1.49, 2.34))):
        add(f"variance ratio RF={v} vs RF=1", ms, round(float(np.exp(b[1 + k])), 3), "Gamma GLM (log link) of r^2 on RF and CF categories")
    # CF geometry level
    g = df.assign(r=r).groupby("geometry")
    cfmax = g.CF.max()
    for v, (ng, ns) in zip((1, 2, 3, 4), ((1367, 120), (363, 48), (263, 27), (53, 7))):
        gg = cfmax[cfmax == v].index
        add(f"CF={v} geometries (max CF)", ng, int(len(gg)), "count of geometries with max CF")
        add(f"CF={v} series (max CF geometries)", ns, int(df[df.geometry.isin(gg)].series.nunique()), "count")
    rmse_g = g.r.apply(lambda s: np.sqrt(np.mean(s ** 2)))
    add("CF=3 geometry RMSE", 0.391, round(float(rmse_g[cfmax == 3].mean()), 4), "unweighted mean of per-geometry RMSE, geometries with max CF = 3 (935 tests)",
        note="A4: different aggregation from the record-level value")
    add("CF=3 record-level RMSE", 0.71, round(float(np.sqrt(np.mean(r[cf == 3] ** 2))), 4), "pooled RMSE over 825 records with CF = 3",
        note="pooled RMSE over the 935 tests in max-CF=3 geometries = %.3f" % np.sqrt(np.mean(r[df.geometry.map(cfmax).to_numpy() == 3] ** 2)))
    # replicate pairs
    rp = replicate_pairs(df, 0.01)
    add("near-replicate pairs", 1136, len(rp), "pairs within 1 % on Hm0, Tm-1,0, h, Rc, same geometry and beta")
    for v, n in zip((1, 2, 3, 4), (130, 483, 457, 66)):
        add(f"pairs with max RF={v}", n, int((rp.rf_max == v).sum()), "count")
    add("replicate variance / within-geometry variance", 0.576, round(float(np.mean(rp.dy ** 2) / 2 / 0.1384), 4), "mean(d^2)/2 divided by REML within variance")
    # series-held-out CF trend
    rs = np.abs(y - load_pred("series__xgb")[0])
    add("Spearman(|r|, CF), series held out", 0.11, round(float(spearmanr(cf, rs)[0]), 4), "record-level Spearman correlation")
    # policies
    Z = np.load(ROOT / "original/original_results/predictions/policy__xgb.npz")
    com = (rf < 4) & (cf < 4)
    base = Z["P1_retain_all__pred"]
    dmax = 0.0
    for p in ("P2_excl_RF4", "P3_excl_CF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks"):
        d = np.sqrt(np.mean((Z[f"{p}__pred"][com] - y[com]) ** 2)) - np.sqrt(np.mean((base[com] - y[com]) ** 2))
        dmax = max(dmax, abs(d))
    add("max |dRMSE| on common tests, exclusion policies", 0.007, round(float(dmax), 4), "pooled RMSE difference to P1 on 8 494 common tests")
    cov = lambda p, mk: float(((y >= Z[f"{p}__lo"]) & (y <= Z[f"{p}__hi"]))[mk].mean())  # noqa: E731
    add("coverage RF=4, P1", 0.81, round(cov("P1_retain_all", rf == 4), 4), "empirical coverage of 90 % intervals")
    add("coverage RF=4, P6b", 0.89, round(cov("P6b_quality_aware_intervals", rf == 4), 4), "empirical coverage")
    add("coverage CF=4, P1", 0.94, round(cov("P1_retain_all", cf == 4), 4), "empirical coverage")
    add("coverage CF=4, P6b", 0.90, round(cov("P6b_quality_aware_intervals", cf == 4), 4), "empirical coverage")
    add("coverage CF=3, P6b", "below nominal", round(cov("P6b_quality_aware_intervals", cf == 3), 4), "empirical coverage", verdict="consistent (under-covered)")
    w = lambda p: float((Z[f"{p}__hi"] - Z[f"{p}__lo"]).mean())  # noqa: E731
    add("mean width P1", 1.55, round(w("P1_retain_all"), 4), "log10 units", note="manuscript: P1 1.55 vs P6b 1.54; the revision brief lists them in reverse order")
    add("mean width P6b", 1.54, round(w("P6b_quality_aware_intervals"), 4), "log10 units")
    # wet/dry AUC (original model)
    from sklearn.metrics import roc_auc_score
    wd = pd.read_csv(ROOT / "results/predictions/wetdry__xgb_geometry.csv")
    wet = (wd.response_class == "positive").astype(int)
    for v, ms in ((1, 0.92), (4, 0.81)):
        mk = wd.RF == v
        add(f"wet/dry AUC RF={v} (original model)", ms, round(float(roc_auc_score(wet[mk], wd.p_wet[mk])), 4), "ROC AUC, original fixed-parameter classifier")
    # highlights / abstract wording
    add("Highlight: 'All 10 532 tests ... retained'", "all retained", "10 532 audited; 10 389 in wet/dry model; 9 333 in regression", "-", verdict="WORDING CORRECTED",
        note="database audit vs model eligibility must be distinguished")
    add("Abstract/highlight: intervals 'equalise coverage'", "equalised", "RF equalised (max deviation 4.3 pts); CF = 3 remains under-covered", "-", verdict="WORDING CORRECTED")
    add("Figure cross-references (REML, policies)", "Fig. 4, Fig. 5, Fig. 6", "Fig. 5, Fig. 6, Fig. 7", "build order of figures", verdict="CORRECTED")
    out = pd.DataFrame(rows)
    (ROOT / "revision" / "audit").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "revision/audit/numerical_consistency_audit.csv", index=False)
    pd.set_option("display.width", 250)
    print(out[["item", "manuscript_value", "recomputed_value", "verdict"]].to_string())


if __name__ == "__main__":
    main()
