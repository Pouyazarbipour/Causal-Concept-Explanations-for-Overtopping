"""Tables for the data-quality validation study.

Usage: python -m experiments.run_analysis [--only perf rfcf decomp ccw policy unc] [--n_boot N]
Every experiment only needs saved out-of-fold predictions, so each can be re-run independently.
"""
from __future__ import annotations

import argparse
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from src import rf_cf_validation as V
from src import uncertainty_modeling as U
from src.concept_ontology import CONCEPT_NAMES, CONCEPTS
from src.data_selection import POLICIES
from src.error_decomposition import decompose, replicate_pairs
from src.evaluation import METRICS, regression_metrics
from src.pipeline import groups_for, have, load_model_data, load_pred
from src.statistical_analysis import cluster_bootstrap, holm
from src.utils import ROOT, get_logger, out, save_table

log = get_logger("analysis")
warnings.filterwarnings("ignore", category=RuntimeWarning)
MODELS = ["rf", "xgb", "catboost", "mlp", "ccw_full"]
EXTRA = ["backbone", "cbm", "ccw_unaligned"]
LABEL = {"rf": "Random Forest", "xgb": "XGBoost", "catboost": "CatBoost", "mlp": "MLP", "backbone": "CCW backbone (tabular MLP)",
         "cbm": "Concept bottleneck", "ccw_unaligned": "Concept wrapper (unaligned)", "ccw_full": "CCW-Net"}
REF_MODEL = "xgb"  # pre-specified reference model for residual-based RF/CF analyses
# predictor -> concept group, for the comparison with conventional feature attribution (TreeSHAP)
FEATURE_CONCEPT = {"rc_h": "relative_freeboard", "s0": "wave_steepness", "h_l": "relative_depth", "beta": "wave_obliquity",
                   "cotad": "profile_type", "cotau": "profile_type", "b_l": "berm", "hb_h": "berm", "tanab": "berm",
                   "gf": "roughness_permeability", "bt_l": "toe_berm", "ht_h": "toe_berm", "ac_h": "crest_configuration",
                   "gc_l": "crest_configuration", "fs": "foreshore"}


def seed_metrics(P, y, mask=None):
    """Mean over seeds of per-seed metrics."""
    mask = np.ones(len(y), bool) if mask is None else mask
    ms = [regression_metrics(y[mask], p[mask]) for p in P]
    return {k: float(np.mean([m[k] for m in ms])) for k in ("n",) + METRICS} | {"rmse_seed_sd": float(np.std([m["rmse"] for m in ms]))}


def rmse_ci(P, y, clusters, n_boot, seed, mask=None):
    se = ((P - y) ** 2).mean(0)
    idx0 = np.flatnonzero(np.ones(len(y), bool) if mask is None else mask)
    _, lo, hi, _ = cluster_bootstrap(lambda i: [np.sqrt(se[idx0][i].mean())], clusters[idx0], n_boot, seed)
    return float(lo[0]), float(hi[0])


# ============================================================================== model performance
def exp_perf(D, nb, seed):
    df, y, E = D["df"], D["y"], D["cfg"]["experiment"]
    prim = E["primary_scheme"]
    rows, strat = [], []
    for scheme in [prim] + E["sensitivity_schemes"]:
        cl = df["geometry"].to_numpy() if scheme == "random" else groups_for(df, scheme)
        for m in MODELS + EXTRA:
            if not have(f"{scheme}__{m}"):
                continue
            P = load_pred(f"{scheme}__{m}")
            lo, hi = rmse_ci(P, y, cl, nb, seed)
            rows.append({"scheme": scheme, "model": LABEL[m], "n_seeds": len(P), **seed_metrics(P, y), "rmse_lo": lo, "rmse_hi": hi})
            if scheme == prim:
                for fac in ("RF", "CF"):
                    for v in (1, 2, 3, 4):
                        mk = (df[fac] == v).to_numpy()
                        l2, h2 = rmse_ci(P, y, cl, min(nb, 500), seed, mk)
                        strat.append({"model": LABEL[m], "factor": fac, "category": v, **seed_metrics(P, y, mk), "rmse_lo": l2, "rmse_hi": h2})
    t = pd.DataFrame(rows)
    save_table(t, "table03_model_performance")
    save_table(pd.DataFrame(strat), "table03b_performance_by_rf_cf")
    # Optuna summary: selected hyper-parameters, inner objective and outer test error per fold
    d = ROOT / "results" / "models" / "optuna"
    opt = []
    for f in sorted(d.glob("*__best.csv")):
        scheme, model = f.stem.split("__")[:2]
        b = pd.read_csv(f)
        tr = pd.read_csv(d / f"{scheme}__{model}__trials.csv")
        b.insert(0, "model", model)
        b.insert(0, "scheme", scheme)
        b["objective_first_trial"] = tr.groupby("fold")["value"].first().reindex(b["fold"]).to_numpy()
        b["objective_median_trial"] = tr.groupby("fold")["value"].median().reindex(b["fold"]).to_numpy()
        opt.append(b)
    if opt:
        save_table(pd.concat(opt), "table04_optuna_best_hyperparameters")
    wd = ROOT / "results/predictions/wetdry__xgb_geometry.csv"
    if wd.exists():
        from sklearn.metrics import brier_score_loss, roc_auc_score
        w = pd.read_csv(wd)
        w["wet"] = (w.response_class == "positive").astype(int)
        rr = [{"subset": "all", "n": len(w), "n_zero": int((w.wet == 0).sum()), "auc": roc_auc_score(w.wet, w.p_wet), "brier": brier_score_loss(w.wet, w.p_wet)}]
        for fac in ("RF", "CF"):
            for v in (1, 2, 3, 4):
                sub = w[w[fac] == v]
                rr.append({"subset": f"{fac}={v}", "n": len(sub), "n_zero": int((sub.wet == 0).sum()),
                           "auc": roc_auc_score(sub.wet, sub.p_wet) if sub.wet.nunique() > 1 else np.nan, "brier": brier_score_loss(sub.wet, sub.p_wet)})
        save_table(pd.DataFrame(rr), "tableS_wet_dry_model")


# ============================================================================== Experiments 1, 5, 6, 7
def exp_rfcf(D, nb, seed):
    df, y = D["df"], D["y"]
    scheme = D["cfg"]["experiment"]["primary_scheme"]
    r = y - load_pred(f"{scheme}__{REF_MODEL}")[0]
    r_rand = y - load_pred(f"random__{REF_MODEL}")[0]
    strata = pd.concat([V.strata_table(df, r, f, "series", nb, seed) for f in ("RF", "CF")])
    save_table(strata, "table05a_residuals_by_rf_cf")
    trends = [V.trend_tests(df, r, f, "series", nb, seed) | {"residuals": f"{REF_MODEL}, geometry-grouped"} for f in ("RF", "CF")]
    for m in ["rf", "catboost", "mlp", "ccw_full"]:   # robustness across models
        if have(f"{scheme}__{m}"):
            rm = y - load_pred(f"{scheme}__{m}").mean(0)
            trends += [V.trend_tests(df, rm, f, "series", min(nb, 300), seed) | {"residuals": f"{m}, geometry-grouped"} for f in ("RF", "CF")]
    trends += [V.trend_tests(df, r_rand, f, "series", min(nb, 300), seed) | {"residuals": f"{REF_MODEL}, random CV"} for f in ("RF", "CF")]
    if have(f"series__{REF_MODEL}"):
        r_ser = y - load_pred(f"series__{REF_MODEL}")[0]
        trends += [V.trend_tests(df, r_ser, f, "series", min(nb, 300), seed) | {"residuals": f"{REF_MODEL}, series held out"} for f in ("RF", "CF")]
    t = pd.DataFrame(trends)
    t["boot_p_holm"] = holm(t["boot_p_spearman"])
    save_table(t, "table05b_ordinal_trend_tests")
    # ordinal (proportional-odds) models -------------------------------------------------------
    d2 = df.assign(abs_resid=np.abs(r), log_scale=np.log10(df["hm0_toe"]), ref_missing=df["flag_reference_missing"].astype(float),
                   deep_missing=df["flag_deep_missing"].astype(float), tm10_default=df["flag_tm10_default_ratio"].astype(float))
    gm = pd.Series(r).groupby(df["geometry"].to_numpy()).transform("mean").to_numpy()
    d2["abs_within_geometry_resid"], d2["abs_geometry_mean_resid"] = np.abs(r - gm), np.abs(gm)
    nb2 = min(nb, 200)
    ol = [V.ordinal_logit(d2, "RF", ["abs_resid"], "series", nb2, seed).assign(model="RF ~ |r|"),
          V.ordinal_logit(d2, "RF", ["abs_within_geometry_resid", "abs_geometry_mean_resid"], "series", nb2, seed).assign(model="RF ~ within + between geometry"),
          V.ordinal_logit(d2, "RF", ["abs_resid", "tm10_default", "deep_missing", "ref_missing", "log_scale"], "series", nb2, seed).assign(model="RF ~ |r| + completeness indicators + scale"),
          V.ordinal_logit(d2, "CF", ["abs_resid"], "series", nb2, seed).assign(model="CF ~ |r| (record level, descriptive)")]
    G = V.geometry_level(df, r, r_rand)
    G.to_csv(out("diagnostics", "geometry_level_residuals.csv"), index=False)
    g2 = G.assign(CFg=G.cf_max, log_n=np.log10(G.n))
    ol.append(V.ordinal_logit(g2.rename(columns={"CFg": "CF"}), "CF", ["abs_mean_resid", "log_n"], "series", nb2, seed).assign(model="CF ~ |geometry mean residual| (geometry level)"))
    save_table(pd.concat(ol), "table05c_ordinal_validation_models")
    save_table(V.variance_model(df, r, "series", nb2, seed), "table05e_variance_model_with_interaction")
    save_table(V.variance_model(df, r, "series", nb2, seed, interaction=False), "table05d_variance_model_main_effects")
    save_table(V.cf_geometry_table(G, nb, seed), "table06a_cf_geometry_level")
    cfg_ = []
    cfv, gcl = G.cf_max.to_numpy(float), G.series.to_numpy()
    for col in ("abs_mean_resid", "rmse_grouped", "generalisation_gap"):
        x = G[col].to_numpy(float)
        pt, lo, hi, reps = cluster_bootstrap(lambda i: [stats.spearmanr(x[i], cfv[i], nan_policy="omit").statistic], gcl, nb, seed)
        cfg_.append({"geometry_level_quantity": col, "spearman_with_CF": pt[0], "lo": lo[0], "hi": hi[0],
                     "boot_p": max(2 * min((reps[:, 0] <= 0).mean(), (reps[:, 0] >= 0).mean()), 1 / nb), "n_geometries": len(G)})
    save_table(pd.DataFrame(cfg_), "table06b_cf_geometry_level_trends")
    save_table(V.joint_table(df, r, "series", min(nb, 500), seed), "table07_rf_cf_joint")
    # construct-related evidence: RF vs documented/inferred completeness indicators (model-free) -------
    m = D["master"]
    cons = []
    for flag in ("flag_tm10_default_ratio", "flag_tm_default_ratio", "flag_deep_missing", "flag_reference_missing", "flag_toe_equals_deep",
                 "flag_remark_do_not_use", "flag_depth_limited_gt1", "n_estimated_indicators"):
        for fac in ("RF", "CF"):
            x, f_, c_ = m[flag].to_numpy(float), m[fac].to_numpy(float), m["series"].to_numpy()
            pt, lo, hi, _ = cluster_bootstrap(lambda i: [stats.spearmanr(x[i], f_[i]).statistic], c_, min(nb, 500), seed)
            cons.append({"indicator": flag, "factor": fac, "spearman": pt[0], "lo": lo[0], "hi": hi[0],
                         **{f"mean_cat{v}": float(x[f_ == v].mean()) for v in (1, 2, 3, 4)}})
    save_table(pd.DataFrame(cons), "table05f_rf_cf_vs_completeness_indicators")
    # zero-overtopping share by RF/CF (all wet/dry records)
    w = m[m.use_wetdry]
    z = [{"factor": fac, "category": v, "n": int((w[fac] == v).sum()), "pct_zero": 100 * (w.loc[w[fac] == v, "response_class"] == "zero").mean()}
         for fac in ("RF", "CF") for v in (1, 2, 3, 4)]
    save_table(pd.DataFrame(z), "table05g_zero_share_by_rf_cf")
    # repeatability from near-replicate measured tests (model-free) ---------------------------------
    rp = replicate_pairs(df, D["cfg"]["experiment"]["replicates"]["rel_tol"])
    rp.to_csv(out("diagnostics", "near_replicate_pairs.csv"), index=False)
    rr = []
    for key, sub in [("all", rp)] + [(f"max RF={v}", rp[rp.rf_max == v]) for v in (1, 2, 3, 4)] + [(f"max CF={v}", rp[rp.cf_max == v]) for v in (1, 2, 3, 4)]:
        row = {"subset": key, "n_pairs": len(sub), "n_geometries": sub.geometry.nunique() if len(sub) else 0, "n_series": sub.series.nunique() if len(sub) else 0}
        if len(sub) >= 20 and sub.series.nunique() >= 3:
            d = sub.dy.to_numpy()
            pt, lo, hi, _ = cluster_bootstrap(lambda i: [np.sqrt(np.mean(d[i] ** 2) / 2), np.median(np.abs(d[i]))], sub.series.to_numpy(), nb, seed)
            row.update({"repeat_sd_log10": pt[0], "sd_lo": lo[0], "sd_hi": hi[0], "median_abs_diff": pt[1]})
        rr.append(row)
    save_table(pd.DataFrame(rr), "table05h_repeatability_near_replicates")


# ============================================================================== Experiment 8
def exp_decomp(D, nb, seed):
    df, y = D["df"], D["y"]
    scheme = D["cfg"]["experiment"]["primary_scheme"]
    nbd = min(nb, 100)
    rows = []

    def add(label, r, sub, levels, n_boot=nbd):
        t, fit = decompose(r, sub.reset_index(drop=True), levels, n_boot, seed)
        t.insert(0, "analysis", label)
        t.insert(1, "n_records", len(sub))
        rows.append(t)
        return fit
    r = y - load_pred(f"{scheme}__{REF_MODEL}")[0]
    t0, fit = decompose(r, df, ["series", "geometry"], nbd, seed, blup=True)
    t0.insert(0, "analysis", "PRIMARY: XGBoost, geometry-grouped OOF residuals; series / geometry")
    t0.insert(1, "n_records", len(df))
    rows.append(t0)
    pd.DataFrame({"geometry_key": list(fit["effects"]), "u_geometry": list(fit["effects"].values())}).to_csv(
        out("diagnostics", "geometry_random_effects.csv"), index=False)
    add("grouping sensitivity: geometry_coarse", r, df, ["series", "geometry_coarse"], 0)
    add("grouping sensitivity: substructure", r, df, ["series", "substructure"], 0)
    add("grouping sensitivity: source (reference) / series / geometry", r, df, ["source", "series", "geometry"], 0)
    known = (df["lab"] != "unknown").to_numpy()
    add("records with an INFERRED laboratory only: lab / series / geometry", r[known], df[known], ["lab", "series", "geometry"], nbd)
    rr = y - load_pred(f"random__{REF_MODEL}")[0]
    add("random-CV residuals (optimistic reference)", rr, df, ["series", "geometry"], 0)
    if have(f"series__{REF_MODEL}"):
        add("series-held-out residuals", y - load_pred(f"series__{REF_MODEL}")[0], df, ["series", "geometry"], 0)
    for m in ("catboost", "mlp", "ccw_full", "rf"):
        if have(f"{scheme}__{m}"):
            add(f"model sensitivity: {m}", y - load_pred(f"{scheme}__{m}").mean(0), df, ["series", "geometry"], 0)
    for v in (1, 2, 3, 4):
        mk = (df["CF"] == v).to_numpy()
        add(f"stratum CF={v}", r[mk], df[mk], ["series", "geometry"], nbd // 2)
    for v in (1, 2, 3, 4):
        mk = (df["RF"] == v).to_numpy()
        add(f"stratum RF={v}", r[mk], df[mk], ["series", "geometry"], nbd // 2)
    save_table(pd.concat(rows), "table08_hierarchical_error_decomposition")
    # model-induced part: total OOF MSE vs. model-free repeatability variance
    rp = pd.read_csv(ROOT / "results/diagnostics/near_replicate_pairs.csv") if (ROOT / "results/diagnostics/near_replicate_pairs.csv").exists() else replicate_pairs(df)
    rep_var = float(np.mean(rp.dy ** 2) / 2) if len(rp) else np.nan
    prim = rows[0]
    summ = pd.DataFrame([{"quantity": "total OOF residual variance (geometry-grouped)", "value": float(np.var(r))},
                         {"quantity": "total OOF residual variance (random CV)", "value": float(np.var(rr))},
                         {"quantity": "within-geometry residual variance (REML)", "value": float(prim.variance.iloc[-1])},
                         {"quantity": "near-replicate repeatability variance (model-free, pooled)", "value": rep_var},
                         {"quantity": "n near-replicate pairs", "value": len(rp)},
                         {"quantity": "ratio: near-replicate variance / within-geometry residual variance", "value": rep_var / float(prim.variance.iloc[-1])}])
    save_table(summ, "table08b_repeatability_vs_residual")


# ============================================================================== Experiment 9
def exp_unc(D, nb, seed):
    df, y = D["df"], D["y"]
    cfg = D["cfg"]
    scheme = cfg["experiment"]["primary_scheme"]
    fold, groups = D["folds"][scheme].to_numpy(), groups_for(df, scheme)
    X = D["X"]
    rf, cf = df["RF"].to_numpy(int), df["CF"].to_numpy(int)
    best = {int(k): pickle.load(open(ROOT / "results/models/fold_cache" / f"{scheme}__xgb__fold{int(k)}.pkl", "rb"))["best"] for k in np.unique(fold)}
    alphas = [0.5, 0.2, 0.1, 0.05]
    reps = 3
    jobs = [delayed(U.run_fold)(X, y, rf, cf, groups, np.flatnonzero(fold != k), np.flatnonzero(fold == k), best[int(k)], alphas,
                                seed + 17 * rep, cfg["experiment"]["uncertainty"]["calib_fraction"]) for rep in range(reps) for k in np.unique(fold)]
    res = Parallel(n_jobs=4)(jobs)
    iv = {a: {m: (np.full((reps, len(y)), np.nan), np.full((reps, len(y)), np.nan)) for m in U.METHODS} for a in alphas}
    i = 0
    for rep in range(reps):
        for k in np.unique(fold):
            te = np.flatnonzero(fold == k)
            for m in U.METHODS:
                for a in alphas:
                    iv[a][m][0][rep, te], iv[a][m][1][rep, te] = res[i][m][a]
            i += 1
    al = cfg["experiment"]["uncertainty"]["alpha"]
    t, strat = U.coverage_tables(df, y, iv[al], al, nb, seed)
    cal = {m: np.mean([abs(((y >= iv[a][m][0]) & (y <= iv[a][m][1])).mean() - (1 - a)) for a in alphas]) for m in U.METHODS}
    t["calibration_error_over_levels"] = t["method"].map(cal)
    for a in alphas:
        t[f"coverage@{1 - a:.2f}"] = [((y >= iv[a][m][0]) & (y <= iv[a][m][1])).mean() for m in t["method"]]
    save_table(t, "tableS_conformal_methods_by_quality")
    save_table(strat, "tableS_coverage_by_rf_cf_cell")
    np.savez_compressed(out("predictions", "conformal_intervals_alpha0.1.npz"), **{f"{m}__lo": iv[al][m][0] for m in U.METHODS},
                        **{f"{m}__hi": iv[al][m][1] for m in U.METHODS})
    # width ratio by category: do quality-aware intervals widen where errors are larger?
    wr = []
    for m in ("split_abs", "norm_physical", "norm_physical+quality", "norm_quality_only", "cqr_physical", "cqr_physical+quality"):
        w = (iv[al][m][1] - iv[al][m][0]).mean(0)
        for fac, arr in (("RF", rf), ("CF", cf)):
            for v in (1, 2, 3, 4):
                ww = w[arr == v]
                wr.append({"method": m, "factor": fac, "category": v, "mean_width": float(ww[np.isfinite(ww)].mean()),
                           "coverage": float(((y >= iv[al][m][0]) & (y <= iv[al][m][1])).mean(0)[arr == v].mean())})
    save_table(pd.DataFrame(wr), "tableS_interval_width_by_rf_cf")


# ============================================================================== causal concept analysis
def exp_ccw(D, nb, seed):
    df, y = D["df"], D["y"]
    scheme = D["cfg"]["experiment"]["primary_scheme"]
    seeds = D["cfg"]["experiment"]["nn_seeds"]
    fold = D["folds"][scheme].to_numpy()
    m = pd.read_csv(ROOT / "results/diagnostics/ccw_metrics.csv")
    rm = []
    for v in m["variant"].unique():
        P = load_pred(f"{scheme}__{v}")
        for si in range(len(P)):
            for k in np.unique(fold):
                rm.append({"variant": v, "seed": seeds[si], "fold": k, "rmse": float(np.sqrt(np.mean((P[si, fold == k] - y[fold == k]) ** 2)))})
    m = m.merge(pd.DataFrame(rm), on=["variant", "seed", "fold"], how="left")
    cols = [c for c in ["rmse", "concept_acc", "concept_macro_f1", "concept_ordinal_mae", "concept_ece", "nec_abs_effect", "suf_rmse_vs_target",
                        "suf_rmse_vs_output", "ind_rmse", "interaction_share", "rho_inactive", "rho_inactive_per_null_slot",
                        "rho_active_per_active_slot", "rho_changed_matched", "epochs_run"] if c in m]
    g = m.groupby("variant")[cols]
    t = g.mean()
    t.insert(0, "n_runs(fold x seed)", g.size())
    for c in cols:
        t[c + "_sd"] = g.std()[c]
    full = m[m.variant == "ccw_full"].set_index(["fold", "seed"])
    for v in t.index:   # paired differences over identical fold-seed runs (runs are not independent: descriptive)
        o = m[m.variant == v].set_index(["fold", "seed"])
        j = full.index.intersection(o.index)
        for c in ("rmse", "rho_inactive_per_null_slot", "nec_abs_effect", "suf_rmse_vs_output", "ind_rmse", "rho_changed_matched"):
            if c in o and len(j) > 1 and o.loc[j, c].notna().all():
                d = (o.loc[j, c] - full.loc[j, c]).to_numpy()
                t.loc[v, f"d_{c}_vs_full"] = d.mean()
                t.loc[v, f"d_{c}_t95"] = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    save_table(t.reset_index(), "table09_ccw_alignment_and_ablation")
    pc = json.load(open(ROOT / "results/diagnostics/ccw_per_concept.json"))
    sh_all = pd.read_csv(ROOT / "results/diagnostics/ccw_shapley.csv")
    # ---- conventional feature attribution (TreeSHAP of the reference model), grouped by concept ----------
    S = np.abs(np.load(ROOT / f"results/diagnostics/shap__{scheme}__xgb.npz")["shap"][:, :-1])
    feats = D["features"]
    G = np.column_stack([S[:, [i for i, f in enumerate(feats) if FEATURE_CONCEPT[f] == c]].sum(1) for c in CONCEPT_NAMES])
    L = D["labels"].to_numpy()
    nullcap = np.array([c.has_null for c in CONCEPTS])
    null = (L == 0) & nullcap
    share = G / G.sum(1, keepdims=True)
    hn = null.any(1)
    shap_null_share = float(((share * null).sum(1) / np.maximum(null.sum(1), 1))[hn].mean())
    shap_active_share = float(((share * ~null).sum(1) / (~null).sum(1)).mean())
    for var in ("ccw_full", "ccw_unaligned"):
        sel = [r for r in pc if r["variant"] == var]
        A = lambda k: np.nanmean([r[k] for r in sel], axis=0)  # noqa: E731
        sh = sh_all[sh_all.variant == var]
        rho = sh.groupby("concept").apply(lambda q: np.average(q.rho_changed, weights=q.n), include_groups=False)
        t6 = pd.DataFrame({"concept": CONCEPT_NAMES, "null_capable": nullcap, "ccw_slot_ablation_effect_log10": A("abl_effect"),
                           "sd_over_runs": np.nanstd([r["abl_effect"] for r in sel], axis=0),
                           "ccw_effect_when_active": A("abl_effect_active"), "ccw_effect_when_null": A("abl_effect_null"),
                           "concept_accuracy": A("concept_acc"), "macro_f1": A("concept_f1"), "ordinal_mae": A("concept_ordinal_mae"), "ece": A("concept_ece"),
                           "rho_changed_matched": [rho.get(j, np.nan) for j in range(len(CONCEPTS))],
                           "xgb_mean_abs_shap_log10": G.mean(0),
                           "xgb_shap_when_active": [G[~null[:, j], j].mean() for j in range(len(CONCEPTS))],
                           "xgb_shap_when_null": [G[null[:, j], j].mean() if null[:, j].any() else 0.0 for j in range(len(CONCEPTS))]})
        save_table(t6, f"table10_concept_attribution_{var}")
        Pm = A("pair_abs")
        pr = [{"concept_a": CONCEPT_NAMES[i], "concept_b": CONCEPT_NAMES[j], "mean_abs_pair_term_log10": Pm[i][j]}
              for i in range(len(CONCEPTS)) for j in range(i + 1, len(CONCEPTS))]
        save_table(pd.DataFrame(pr).sort_values("mean_abs_pair_term_log10", ascending=False), f"tableS_interaction_pairs_{var}")
    t6 = pd.read_csv(ROOT / "results/tables/table10_concept_attribution_ccw_full.csv")
    tb = t.reset_index().set_index("variant")
    cmp_ = pd.DataFrame([
        {"quantity": "rank agreement of concept importance: CCW slot ablation vs grouped |SHAP| (Spearman, 10 concepts)",
         "value": stats.spearmanr(t6.ccw_slot_ablation_effect_log10, t6.xgb_mean_abs_shap_log10).statistic},
        {"quantity": "attribution share per absent (null-state) concept: grouped TreeSHAP, XGBoost", "value": shap_null_share},
        {"quantity": "attribution share per absent (null-state) concept: CCW-Net", "value": tb.loc["ccw_full", "rho_inactive_per_null_slot"]},
        {"quantity": "attribution share per absent (null-state) concept: unaligned wrapper", "value": tb.loc["ccw_unaligned", "rho_inactive_per_null_slot"]},
        {"quantity": "attribution share per absent (null-state) concept: concept bottleneck", "value": tb.loc["cbm", "rho_inactive_per_null_slot"]},
        {"quantity": "attribution share per active concept: grouped TreeSHAP, XGBoost", "value": shap_active_share},
        {"quantity": "attribution share per active concept: CCW-Net", "value": tb.loc["ccw_full", "rho_active_per_active_slot"]},
        {"quantity": "share routed through the concept whose label changed (matched estimator): CCW-Net (0.10 = indiscriminate)", "value": tb.loc["ccw_full", "rho_changed_matched"]},
        {"quantity": "out-of-geometry RMSE: XGBoost (explained by SHAP)", "value": float(np.sqrt(np.mean((load_pred(f"{scheme}__xgb")[0] - y) ** 2)))},
        {"quantity": "out-of-geometry RMSE: CCW-Net (explained by concept slots)", "value": float(np.sqrt(np.mean((load_pred(f"{scheme}__ccw_full") - y) ** 2)))}])
    save_table(cmp_, "table11_concept_vs_feature_attribution")
    q = pd.read_csv(ROOT / "results/diagnostics/ccw_quality_branch.csv")
    if len(q):
        qq = q.groupby("head")[["test_nll", "cov90", "mean_sigma"]].agg(["mean", "std"])
        qq.columns = ["_".join(c) for c in qq.columns]
        save_table(qq.reset_index(), "tableS_ccw_quality_layer")
    # ---- explanation quality versus RF / CF ------------------------------------------------------------
    cl = df["series"].to_numpy()
    rows, trend = [], []
    for var in ("ccw_full", "ccw_unaligned"):
        R = np.load(ROOT / f"results/diagnostics/ccw_records__{var}.npz")
        err = np.abs(load_pred(f"{scheme}__{var}") - y).mean(0)
        Q = {"abs_error": err, "necessity_effect": np.nanmean(R["nec_effect"], 0), "sufficiency_gap": R["suf_gap"].mean(0),
             "independence_residual": R["ind_resid"].mean(0), "interaction_abs": R["interaction_abs"].mean(0),
             "null_slot_effect_share": np.nanmean(R["rho_inactive"], 0), "concept_readout_accuracy": R["concept_correct"].mean(0)}
        for fac in ("RF", "CF"):
            f_ = df[fac].to_numpy(float)
            for name, x in Q.items():
                ok = np.isfinite(x)
                for v in (1, 2, 3, 4):
                    mk = ok & (f_ == v)
                    i0 = np.flatnonzero(mk)
                    pt, lo, hi, _ = cluster_bootstrap(lambda i: [x[i0][i].mean()], cl[i0], min(nb, 400), seed)
                    rows.append({"model": LABEL[var], "quantity": name, "factor": fac, "category": v, "n": int(mk.sum()), "mean": pt[0], "lo": lo[0], "hi": hi[0]})
                i1 = np.flatnonzero(ok)
                pt, lo, hi, reps = cluster_bootstrap(lambda i: [stats.spearmanr(x[i1][i], f_[i1][i]).statistic], cl[i1], min(nb, 400), seed)
                trend.append({"model": LABEL[var], "quantity": name, "factor": fac, "spearman": pt[0], "lo": lo[0], "hi": hi[0],
                              "boot_p": max(2 * min((reps[:, 0] <= 0).mean(), (reps[:, 0] >= 0).mean()), 1 / min(nb, 400))})
    tt = pd.DataFrame(trend)
    tt["boot_p_holm"] = holm(tt["boot_p"])
    save_table(pd.DataFrame(rows), "table12a_explanation_quality_by_rf_cf")
    save_table(tt, "table12b_explanation_quality_trends")
    # ---- is quality information recoverable from the learned PHYSICAL concept vectors? -------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    reps = pickle.load(open(ROOT / "results/diagnostics/ccw_full_test_representations_seed0.pkl", "rb"))
    Xraw = np.nan_to_num(D["X"])
    targets = {"RF>=3": (df["RF"] >= 3).to_numpy(), "RF=4": (df["RF"] == 4).to_numpy(), "CF>=2": (df["CF"] >= 2).to_numpy(), "CF>=3": (df["CF"] >= 3).to_numpy()}
    pr = []
    for tname, tv in targets.items():
        for src in ("concept vectors (CCW-Net)", "raw physical predictors"):
            for gname in ("geometry", "series"):
                aucs = []
                for k, (te, Mk) in reps.items():        # probes are fitted and scored inside each outer test fold
                    A = Mk if src.startswith("concept") else Xraw[te]
                    grp = df[gname].to_numpy()[te]
                    p = np.full(len(te), np.nan)
                    for a, b in GroupKFold(3).split(A, groups=grp):
                        if tv[te][a].min() == tv[te][a].max():
                            continue
                        p[b] = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500)).fit(A[a], tv[te][a]).predict_proba(A[b])[:, 1]
                    okp = np.isfinite(p)
                    if okp.sum() and tv[te][okp].min() != tv[te][okp].max():
                        aucs.append(roc_auc_score(tv[te][okp], p[okp]))
                pr.append({"target": tname, "probe_input": src, "probe_validation_grouped_by": gname, "auc_mean_over_folds": float(np.mean(aucs)),
                           "auc_sd": float(np.std(aucs)), "prevalence": float(tv.mean())})
    save_table(pd.DataFrame(pr), "table12c_quality_probe_from_concepts")


# ============================================================================== data-selection policies
def exp_policy(D, nb, seed):
    df, y = D["df"], D["y"]
    al = D["cfg"]["experiment"]["uncertainty"]["alpha"]
    cl = df["geometry"].to_numpy()
    rf, cf, fam = df["RF"].to_numpy(), df["CF"].to_numpy(), df["family"].to_numpy()
    subsets = {"all": np.ones(len(y), bool), "common (RF<4 and CF<4)": (rf < 4) & (cf < 4), "RF=4": rf == 4, "CF=4": cf == 4}
    nbp = min(nb, 500)
    rows, fam_rows, cal_rows = [], [], []
    for learner in ("xgb", "rf"):
        f = ROOT / f"results/predictions/policy__{learner}.npz"
        if not f.exists():
            continue
        Z = np.load(f)
        ntr = pd.read_csv(ROOT / f"results/diagnostics/policy_n_train__{learner}.csv").groupby("policy")["n_train"].sum() / 4  # each record trains in 4 of 5 folds
        base = Z["P1_retain_all__pred"]
        gsize = pd.Series(cl).map(pd.Series(cl).value_counts()).to_numpy()
        for p, desc in POLICIES.items():
            pred, lo_, hi_ = Z[f"{p}__pred"], Z[f"{p}__lo"], Z[f"{p}__hi"]
            row = {"learner": learner, "policy": p, "description": desc, "n_training_records_used": float(ntr[p])}
            cov, width = ((y >= lo_) & (y <= hi_)).astype(float), hi_ - lo_
            for sname, mk in subsets.items():
                i0 = np.flatnonzero(mk)
                e1, e0 = (pred[i0] - y[i0]) ** 2, (base[i0] - y[i0]) ** 2
                pt, lo, hi, _ = cluster_bootstrap(lambda i: [np.sqrt(e1[i].mean()), np.sqrt(e1[i].mean()) - np.sqrt(e0[i].mean()), cov[i0][i].mean()], cl[i0], nbp, seed)
                tag = sname.split(" ")[0]
                row.update({f"rmse_{tag}": pt[0], f"d_rmse_{tag}": pt[1], f"d_rmse_{tag}_lo": lo[1], f"d_rmse_{tag}_hi": hi[1]})
                cal_rows.append({"learner": learner, "policy": p, "test_subset": sname, "n": int(mk.sum()), "nominal": 1 - al, "coverage": pt[2],
                                 "cov_lo": lo[2], "cov_hi": hi[2], "mean_width": float(width[i0].mean()), "mae": float(np.abs(pred[i0] - y[i0]).mean()),
                                 "bias_pred_minus_meas": float((pred[i0] - y[i0]).mean())})
            for fac, arr in (("RF", rf), ("CF", cf)):
                for v in (1, 2, 3, 4):
                    mk = arr == v
                    cal_rows.append({"learner": learner, "policy": p, "test_subset": f"{fac}={v}", "n": int(mk.sum()), "nominal": 1 - al,
                                     "coverage": float(cov[mk].mean()), "mean_width": float(width[mk].mean()), "mae": float(np.abs(pred[mk] - y[mk]).mean()),
                                     "bias_pred_minus_meas": float((pred[mk] - y[mk]).mean())})
            # geometry-level generalisation: bias of whole held-out geometries (geometries with >= 2 tests)
            gm = pd.Series(pred - y).groupby(cl).mean()
            gn = pd.Series(cl).value_counts().reindex(gm.index)
            row["geometry_level_abs_bias"] = float(gm[gn >= 2].abs().mean())
            row["geometry_level_rmse"] = float(np.sqrt((gm[gn >= 2] ** 2).mean()))
            row["coverage_all"] = float(cov.mean())
            row["coverage_geometry_mean"] = float(pd.Series(cov).groupby(cl).mean().mean())
            row["coverage_RF4"], row["coverage_CF4"] = float(cov[rf == 4].mean()), float(cov[cf == 4].mean())
            row["max_abs_coverage_deviation_over_RF"] = float(max(abs(cov[rf == v].mean() - (1 - al)) for v in (1, 2, 3, 4)))
            row["mean_width"] = float(width.mean())
            fb = {f_: float((pred - y)[fam == f_].mean()) for f_ in np.unique(fam)}
            row["max_abs_family_bias"] = float(max(abs(v) for v in fb.values()))
            for f_, b in fb.items():
                fam_rows.append({"learner": learner, "policy": p, "family": f_, "n": int((fam == f_).sum()), "bias_pred_minus_meas": b,
                                 "rmse": float(np.sqrt(np.mean((pred - y)[fam == f_] ** 2))), "coverage": float(cov[fam == f_].mean())})
            # stability of predictions relative to the retain-all policy (same folds, learner, seeds)
            d = pred - base
            com, c4 = subsets["common (RF<4 and CF<4)"], (rf == 4) | (cf == 4)
            row.update({"rms_change_vs_P1_common": float(np.sqrt(np.mean(d[com] ** 2))), "rms_change_vs_P1_category4": float(np.sqrt(np.mean(d[c4] ** 2))),
                        "pct_predictions_changed_by_factor2_common": float(100 * np.mean(np.abs(d[com]) > np.log10(2))),
                        "pct_predictions_changed_by_factor2_category4": float(100 * np.mean(np.abs(d[c4]) > np.log10(2))),
                        "pct_predictions_changed_by_factor10_category4": float(100 * np.mean(np.abs(d[c4]) > 1.0))})
            rows.append(row)
    save_table(pd.DataFrame(rows), "table13_data_selection_policies")
    save_table(pd.DataFrame(cal_rows), "table14_policy_calibration_by_quality")
    save_table(pd.DataFrame(fam_rows), "table15_policy_bias_by_structure_family")
    vm = [pd.read_csv(f).assign(learner=f.stem.split("__")[1]) for f in sorted((ROOT / "results/diagnostics").glob("policy_variance_model__*.csv"))]
    if vm:
        save_table(pd.concat(vm), "tableS_policy_training_variance_model")


EXPS = {"perf": exp_perf, "rfcf": exp_rfcf, "ccw": exp_ccw, "policy": exp_policy, "unc": exp_unc, "decomp": exp_decomp}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(EXPS))
    ap.add_argument("--n_boot", type=int, default=None)
    a = ap.parse_args()
    D = load_model_data()
    nb = a.n_boot or D["cfg"]["experiment"]["n_boot"]
    for k in a.only:
        log.info("experiment %s", k)
        EXPS[k](D, nb, D["cfg"]["experiment"]["seed"])


if __name__ == "__main__":
    main()
