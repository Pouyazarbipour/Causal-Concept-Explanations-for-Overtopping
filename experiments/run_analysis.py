"""Stages F-H (tables): performance, CCW alignment, counterfactual audit, RF/CF validity, error
decomposition, uncertainty and filtering sensitivity.

Usage: python -m experiments.run_analysis [--only perf ccw audit rfcf decomp unc filter]
Every experiment only needs the saved out-of-fold predictions, so each can be re-run independently.
"""
from __future__ import annotations

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from src import rf_cf_validation as V
from src import uncertainty_modeling as U
from src.concept_ontology import CONCEPT_NAMES, CONCEPTS
from src.counterfactual_audit import INTERVENTIONS, curve_metrics, matched_freeboard_pairs
from src.error_decomposition import decompose, replicate_pairs
from src.evaluation import METRICS, regression_metrics
from src.pipeline import groups_for, load_model_data, load_pred
from src.statistical_analysis import cluster_bootstrap, holm
from src.utils import ROOT, get_logger, out, save_table

log = get_logger("analysis")
warnings.filterwarnings("ignore", category=RuntimeWarning)
PRIMARY_MODELS = ["rf", "xgb", "xgb_mono", "mlp", "cbm", "ccw_unaligned", "ccw_full"]
LABEL = {"rf": "Random Forest", "xgb": "XGBoost", "xgb_mono": "Monotonic XGBoost", "mlp": "MLP", "backbone": "CCW backbone (tabular MLP)",
         "cbm": "Concept bottleneck", "ccw_unaligned": "Concept wrapper (unaligned)", "ccw_full": "CCW-Net (full)"}
REF_MODEL = "xgb"  # pre-specified reference model for residual-based RF/CF analyses


def have(tag):
    return (ROOT / "results" / "predictions" / f"{tag}.npz").exists()


def seed_metrics(P, y, mask=None):
    """Mean over seeds of per-seed metrics on records with predictions."""
    mask = np.ones(len(y), bool) if mask is None else mask
    ms = [regression_metrics(y[mask], p[mask, 0]) for p in P]
    return {k: float(np.mean([m[k] for m in ms])) for k in ("n",) + METRICS} | {"rmse_seed_sd": float(np.std([m["rmse"] for m in ms]))}


def rmse_ci(P, y, clusters, n_boot, seed, mask=None):
    se = ((P[:, :, 0] - y) ** 2).mean(0)
    mask = np.isfinite(se) if mask is None else mask & np.isfinite(se)
    idx0 = np.flatnonzero(mask)
    _, lo, hi, _ = cluster_bootstrap(lambda i: [np.sqrt(se[idx0][i].mean())], clusters[idx0], n_boot, seed)
    return float(lo[0]), float(hi[0])


# ============================================================================== Experiment 2
def exp_perf(D, nb, seed):
    df, y, E = D["df"], D["y"], D["cfg"]["experiment"]
    rows, strat = [], []
    for scheme in E["cv_schemes"]:
        cl = df["geometry"].to_numpy() if scheme == "random" else groups_for(df, scheme)
        for m in ["rf", "xgb", "xgb_mono", "mlp", "backbone", "cbm", "ccw_unaligned", "ccw_full"]:
            if not have(f"{scheme}__{m}"):
                continue
            P = load_pred(f"{scheme}__{m}")
            lo, hi = rmse_ci(P, y, cl, nb, seed)
            rows.append({"scheme": scheme, "model": LABEL[m], "n_seeds": len(P), **seed_metrics(P, y), "rmse_lo": lo, "rmse_hi": hi})
            if scheme == E["primary_scheme"]:
                for fac in ("RF", "CF"):
                    for v in (1, 2, 3, 4):
                        mk = (df[fac] == v).to_numpy()
                        l2, h2 = rmse_ci(P, y, cl, nb, seed, mk)
                        strat.append({"model": LABEL[m], "factor": fac, "category": v, **seed_metrics(P, y, mk), "rmse_lo": l2, "rmse_hi": h2})
            if scheme == "lofo_family":
                for fam in sorted(df["family"].unique()):
                    mk = (df["family"] == fam).to_numpy()
                    strat.append({"model": LABEL[m], "factor": "held-out family", "category": fam, **seed_metrics(P, y, mk)})
    t = pd.DataFrame(rows)
    save_table(t, "table03_predictive_performance_all_schemes")
    save_table(t[t.scheme == E["primary_scheme"]].drop(columns="scheme"), "table03a_predictive_performance_geometry_grouped")
    save_table(pd.DataFrame(strat), "table03b_performance_by_rf_cf_and_family")
    if have("geometry__xgb_rfcf_secondary"):
        cl = df["geometry"].to_numpy()
        a, b = load_pred("geometry__xgb"), load_pred("geometry__xgb_rfcf_secondary")
        ea, eb = (a[0, :, 0] - y) ** 2, (b[0, :, 0] - y) ** 2
        pt, lo, hi, _ = cluster_bootstrap(lambda i: [np.sqrt(eb[i].mean()) - np.sqrt(ea[i].mean())], cl, nb, seed)
        save_table(pd.DataFrame([{"model": "XGBoost, physical predictors (PRIMARY)", **regression_metrics(y, a[0, :, 0])},
                                 {"model": "XGBoost + RF, CF as inputs (SECONDARY ablation)", **regression_metrics(y, b[0, :, 0]),
                                  "rmse_diff_vs_primary": pt[0], "diff_lo": lo[0], "diff_hi": hi[0]}]), "table03c_secondary_rfcf_as_mean_inputs")
    wd = ROOT / "results/predictions/wetdry__xgb_geometry.csv"
    if wd.exists():
        from sklearn.metrics import brier_score_loss, roc_auc_score
        w = pd.read_csv(wd)
        w["wet"] = (w.response_class == "positive").astype(int)
        rr = [{"subset": "all", "n": len(w), "n_zero": int((w.wet == 0).sum()), "auc": roc_auc_score(w.wet, w.p_wet), "brier": brier_score_loss(w.wet, w.p_wet)}]
        for fac in ("RF", "CF"):
            for v in (1, 2, 3, 4):
                s = w[w[fac] == v]
                rr.append({"subset": f"{fac}={v}", "n": len(s), "n_zero": int((s.wet == 0).sum()),
                           "auc": roc_auc_score(s.wet, s.p_wet) if s.wet.nunique() > 1 else np.nan, "brier": brier_score_loss(s.wet, s.p_wet)})
        save_table(pd.DataFrame(rr), "table03d_wet_dry_model")


# ============================================================================== Experiment 3
def exp_ccw(D, nb, seed):
    y = D["y"]
    for scheme in D["cfg"]["experiment"]["neural_schemes"]:
        f = ROOT / f"results/diagnostics/ccw_{scheme}_metrics.csv"
        if not f.exists():
            continue
        m = pd.read_csv(f)
        fold = D["folds"][scheme].to_numpy()
        seeds = D["cfg"]["experiment"]["nn_seeds"]
        rm = []
        for v in m["variant"].unique():       # per fold-seed RMSE from saved predictions
            P = load_pred(f"{scheme}__{v}")
            for si, s in enumerate(seeds):
                for k in np.unique(fold):
                    e = P[si, fold == k, 0] - y[fold == k]
                    if np.isfinite(e).all():
                        rm.append({"variant": v, "seed": s, "fold": k, "rmse": float(np.sqrt(np.mean(e ** 2)))})
        m = m.merge(pd.DataFrame(rm), on=["variant", "seed", "fold"], how="left")
        cols = ["rmse", "concept_acc", "concept_macro_f1", "concept_ordinal_mae", "concept_ece", "nec_abs_effect", "suf_rmse_vs_target",
                "suf_rmse_vs_output", "ind_rmse", "interaction_share", "rho_inactive", "rho_inactive_per_null_slot",
                "rho_active_per_active_slot", "rho_changed_matched", "repr_freeboard_reversal_rate", "repr_freeboard_median_dy",
                "repr_roughness_reversal_rate", "repr_obliquity_reversal_rate", "epochs_run"]
        cols = [c for c in cols if c in m]
        g = m.groupby("variant")[cols]
        mean, sd, n = g.mean(), g.std(), g.size()
        t = mean.copy()
        t.insert(0, "n_runs(fold x seed)", n)
        for c in cols:
            t[c + "_sd"] = sd[c]
        # paired difference to the full model over identical fold-seed runs (runs are not independent: descriptive)
        if "ccw_full" in mean.index:
            full = m[m.variant == "ccw_full"].set_index(["fold", "seed"])
            for v in mean.index:
                o = m[m.variant == v].set_index(["fold", "seed"])
                j = full.index.intersection(o.index)
                for c in ("rmse", "rho_inactive", "nec_abs_effect", "suf_rmse_vs_output", "ind_rmse", "rho_changed_matched"):
                    if c in o and len(j) > 1 and o.loc[j, c].notna().all():
                        d = (o.loc[j, c] - full.loc[j, c]).to_numpy()
                        t.loc[v, f"d_{c}_vs_full"] = d.mean()
                        t.loc[v, f"d_{c}_t95"] = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        save_table(t.reset_index(), f"table05_ccw_alignment_ablation_{scheme}")
        # ---- Table 6: concept-level attribution, full model --------------------------------------
        pc = [r for r in json.load(open(ROOT / f"results/diagnostics/ccw_{scheme}_per_concept.json"))]
        for var in ("ccw_full", "ccw_unaligned"):
            sel = [r for r in pc if r["variant"] == var]
            if not sel:
                continue
            A = lambda k: np.nanmean([r[k] for r in sel], axis=0)  # noqa: E731
            S = lambda k: np.nanstd([r[k] for r in sel], axis=0)   # noqa: E731
            sh = pd.read_csv(ROOT / f"results/diagnostics/ccw_{scheme}_shapley.csv")
            sh = sh[sh.variant == var]
            rho = sh.groupby("concept").apply(lambda s: np.average(s.rho_changed, weights=s.n), include_groups=False)
            nn_ = sh.groupby("concept")["n"].sum() / max(len(sel), 1)
            t6 = pd.DataFrame({"concept": CONCEPT_NAMES, "null_capable": [c.has_null for c in CONCEPTS],
                               "zero_ablation_effect_log10": A("abl_effect"), "sd_over_runs": S("abl_effect"),
                               "effect_when_active": A("abl_effect_active"), "effect_when_null": A("abl_effect_null"),
                               "single_slot_contribution_|D0|": A("d0_abs"), "concept_accuracy": A("concept_acc"),
                               "macro_f1": A("concept_f1"), "ordinal_mae": A("concept_ordinal_mae"), "ece": A("concept_ece"),
                               "rho_changed_matched": [rho.get(j, np.nan) for j in range(len(CONCEPTS))],
                               "matched_inputs_per_run": [nn_.get(j, 0) for j in range(len(CONCEPTS))]})
            save_table(t6, f"table06_concept_attribution_{var}_{scheme}")
            Pm = A("pair_abs")
            pr = [{"concept_a": CONCEPT_NAMES[i], "concept_b": CONCEPT_NAMES[j], "mean_abs_pair_term_log10": Pm[i][j]}
                  for i in range(len(CONCEPTS)) for j in range(i + 1, len(CONCEPTS))]
            save_table(pd.DataFrame(pr).sort_values("mean_abs_pair_term_log10", ascending=False), f"table06b_interaction_pairs_{var}_{scheme}")
        q = pd.read_csv(ROOT / f"results/diagnostics/ccw_{scheme}_quality_branch.csv")
        if len(q):
            qq = q.groupby("head")[["test_nll", "cov90", "mean_sigma"]].agg(["mean", "std"])
            qq.columns = ["_".join(c) for c in qq.columns]
            save_table(qq.reset_index(), f"table05b_ccw_quality_layer_{scheme}")


# ============================================================================== Experiment 4
def audit_frames(D, scheme, model):
    """Curve metrics per (seed, record, intervention) for one model."""
    A = D["cfg"]["experiment"]["audit"]
    P = load_pred(f"{scheme}__{model}")
    sup = np.load(ROOT / f"results/diagnostics/support__{scheme}.npz")["support"]
    ok = sup & D["feasible"]
    fr = []
    for s in range(len(P)):
        for iv in INTERVENTIONS:
            c = curve_metrics(P[s], D["views"], ok, iv, A["meaningful_tol"], A["local_delta"])
            c["intervention"], c["seed"], c["rec"] = iv, s, np.arange(len(c))
            fr.append(c)
    return pd.concat(fr, ignore_index=True), ok


def rate_ci(c, num, den, clusters, nb, seed):
    a, b, cl = c[num].to_numpy(float), c[den].to_numpy(float), clusters[c["rec"].to_numpy()]
    if b.sum() == 0:
        return np.nan, np.nan, np.nan
    pt, lo, hi, _ = cluster_bootstrap(lambda i: [a[i].sum() / max(b[i].sum(), 1)], cl, nb, seed)
    return float(pt[0]), float(lo[0]), float(hi[0])


def exp_audit(D, nb, seed):
    df, y = D["df"], D["y"]
    scheme = D["cfg"]["experiment"]["primary_scheme"]
    cl = df["geometry"].to_numpy()
    rows, strat, assoc = [], [], []
    feas = D["feasible"]
    for m in PRIMARY_MODELS:
        if not have(f"{scheme}__{m}"):
            continue
        C, ok = audit_frames(D, scheme, m)
        C.to_parquet(out("diagnostics", f"audit_curves__{scheme}__{m}.parquet")) if False else None
        for iv in INTERVENTIONS:
            c = C[C.intervention == iv]
            cols = np.flatnonzero(D["views"]["intervention"].to_numpy() == iv)
            loc = rate_ci(c, "local_reversal", "local_valid", cl, nb, seed)
            glo = rate_ci(c, "global_reversal", "global_valid", cl, nb, seed)
            aff = rate_ci(c, "curve_affected", "curve_valid", cl, nb, seed)
            stp = rate_ci(c, "n_steps_up", "n_steps_valid", cl, nb, seed)
            rise = c.loc[c.curve_affected, "max_step_rise"]
            rows.append({"model": LABEL[m], "intervention": iv, "n_records_feasible": int(feas[:, cols[0]].sum()),
                         "pct_cf_points_outside_support": 100 * (1 - ok[:, cols][feas[:, cols]].mean()),
                         "n_curves_valid": int(c.curve_valid.sum() / c.seed.nunique()), "n_curves_affected": c.curve_affected.sum() / c.seed.nunique(),
                         "pct_curves_affected": 100 * aff[0], "curves_lo": 100 * aff[1], "curves_hi": 100 * aff[2],
                         "n_steps_valid": int(c.n_steps_valid.sum() / c.seed.nunique()), "n_steps_up": c.n_steps_up.sum() / c.seed.nunique(),
                         "pct_steps_up": 100 * stp[0], "steps_lo": 100 * stp[1], "steps_hi": 100 * stp[2],
                         "mean_up_steps_per_affected_curve": c.loc[c.curve_affected, "n_steps_up"].mean(),
                         "local_reversal_pct": 100 * loc[0], "local_lo": 100 * loc[1], "local_hi": 100 * loc[2],
                         "global_reversal_pct": 100 * glo[0], "global_lo": 100 * glo[1], "global_hi": 100 * glo[2],
                         "median_reversal_log10": rise.median(), "max_reversal_log10": rise.max()})
            if iv != "crest_up":
                continue
            c = c.assign(RF=df["RF"].to_numpy()[c.rec], CF=df["CF"].to_numpy()[c.rec], family=df["family"].to_numpy()[c.rec])
            c["RFxCF"] = "RF" + c.RF.astype(int).astype(str) + "/CF" + c.CF.astype(int).astype(str)
            for key in ("RF", "CF", "family", "RFxCF"):
                for v, s in c.groupby(key):
                    if s.curve_valid.sum() / s.seed.nunique() < 30:
                        continue
                    a_ = rate_ci(s, "curve_affected", "curve_valid", cl, min(nb, 300), seed)
                    l_ = rate_ci(s, "local_reversal", "local_valid", cl, min(nb, 300), seed)
                    g_ = rate_ci(s, "global_reversal", "global_valid", cl, min(nb, 300), seed)
                    strat.append({"model": LABEL[m], "stratum": key, "value": v, "n_curves_valid": int(s.curve_valid.sum() / s.seed.nunique()),
                                  "pct_curves_affected": 100 * a_[0], "aff_lo": 100 * a_[1], "aff_hi": 100 * a_[2],
                                  "local_reversal_pct": 100 * l_[0], "local_lo": 100 * l_[1], "local_hi": 100 * l_[2],
                                  "global_reversal_pct": 100 * g_[0], "global_lo": 100 * g_[1], "global_hi": 100 * g_[2]})
            # association of "curve affected" with RF and CF (rate difference category 4 vs 1; Spearman at record level)
            v = c[c.curve_valid]
            for key in ("RF", "CF"):
                aff_, k_, cl_ = v.curve_affected.to_numpy(float), v[key].to_numpy(float), cl[v.rec.to_numpy()]

                def st(i, aff_=aff_, k_=k_):
                    return [aff_[i][k_[i] == 4].mean() - aff_[i][k_[i] == 1].mean(), stats.spearmanr(aff_[i], k_[i]).statistic]
                pt, lo, hi, reps = cluster_bootstrap(st, cl_, min(nb, 500), seed)
                p = max(2 * min((reps[:, 1] <= 0).mean(), (reps[:, 1] >= 0).mean()), 1 / min(nb, 500))
                assoc.append({"model": LABEL[m], "factor": key, "rate_diff_cat4_minus_cat1_pct": 100 * pt[0], "diff_lo": 100 * lo[0],
                              "diff_hi": 100 * hi[0], "spearman": pt[1], "sp_lo": lo[1], "sp_hi": hi[1], "boot_p": p})
    save_table(pd.DataFrame(rows), "table07_counterfactual_inconsistency")
    save_table(pd.DataFrame(strat), "table07b_reversals_by_rf_cf_family")
    a = pd.DataFrame(assoc)
    if len(a):
        a["boot_p_holm"] = holm(a["boot_p"])
        save_table(a, "table07c_reversal_association_with_rf_cf")
    # ---- Type C: measured matched pairs ----------------------------------------------------------
    mp = matched_freeboard_pairs(df, D["cfg"]["experiment"]["matching"])
    mp["rf_max"] = np.maximum(df.loc[mp.low, "RF"].to_numpy(), df.loc[mp.high, "RF"].to_numpy())
    mp["cf"] = df.loc[mp.low, "CF"].to_numpy()
    mp["geometry"] = df.loc[mp.low, "substructure"].to_numpy()
    tol = D["cfg"]["experiment"]["audit"]["meaningful_tol"]
    mp["measured_reversal"] = mp.d_y > tol
    for m in PRIMARY_MODELS:
        if have(f"{scheme}__{m}"):
            p = load_pred(f"{scheme}__{m}").mean(0)[:, 0]
            mp[f"pred_dy_{m}"] = p[mp.high] - p[mp.low]
    mp.to_csv(out("diagnostics", "matched_freeboard_pairs.csv"), index=False)
    rr = []
    for key, sub in [("all", mp)] + [(f"max RF={v}", mp[mp.rf_max == v]) for v in (1, 2, 3, 4)] + [(f"CF={v}", mp[mp.cf == v]) for v in (1, 2, 3, 4)]:
        if len(sub) < 20:
            rr.append({"subset": key, "n_pairs": len(sub)})
            continue
        x, c_ = sub.measured_reversal.to_numpy(float), sub.geometry.to_numpy()
        pt, lo, hi, _ = cluster_bootstrap(lambda i: [x[i].mean()], c_, nb, seed)
        row = {"subset": key, "n_pairs": len(sub), "n_substructures": sub.geometry.nunique(), "median_d_rc_h": sub.d_rc_h.median(),
               "median_measured_dy": sub.d_y.median(), "measured_reversal_pct": 100 * pt[0], "lo": 100 * lo[0], "hi": 100 * hi[0]}
        for m in PRIMARY_MODELS:
            if f"pred_dy_{m}" in sub:
                row[f"pred_reversal_pct_{m}"] = 100 * (sub[f"pred_dy_{m}"] > tol).mean()
                row[f"pair_slope_rmse_{m}"] = float(np.sqrt(np.mean((sub[f"pred_dy_{m}"] - sub.d_y) ** 2)))
        rr.append(row)
    save_table(pd.DataFrame(rr), "table07d_matched_measured_freeboard_pairs")


# ============================================================================== Experiments 1, 5, 6, 7
def exp_rfcf(D, nb, seed):
    df, y = D["df"], D["y"]
    scheme = D["cfg"]["experiment"]["primary_scheme"]
    r = y - load_pred(f"{scheme}__{REF_MODEL}")[0, :, 0]
    r_rand = y - load_pred(f"random__{REF_MODEL}")[0, :, 0]
    strata = pd.concat([V.strata_table(df, r, f, "series", nb, seed) for f in ("RF", "CF")])
    save_table(strata, "table08a_residuals_by_rf_cf")
    trends = [V.trend_tests(df, r, f, "series", nb, seed) | {"residuals": f"{REF_MODEL}, geometry-grouped"} for f in ("RF", "CF")]
    for m in ["rf", "xgb_mono", "mlp", "ccw_full"]:   # robustness across models
        if have(f"{scheme}__{m}"):
            rm = y - load_pred(f"{scheme}__{m}").mean(0)[:, 0]
            trends += [V.trend_tests(df, rm, f, "series", min(nb, 300), seed) | {"residuals": f"{m}, geometry-grouped"} for f in ("RF", "CF")]
    trends += [V.trend_tests(df, r_rand, f, "series", min(nb, 300), seed) | {"residuals": f"{REF_MODEL}, random CV"} for f in ("RF", "CF")]
    t = pd.DataFrame(trends)
    t["boot_p_holm"] = holm(t["boot_p_spearman"])
    save_table(t, "table08b_ordinal_trend_tests")
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
    save_table(pd.concat(ol), "table08_ordinal_validation_models")
    save_table(V.variance_model(df, r, "series", nb2, seed), "table08c_heteroscedastic_variance_model")
    save_table(V.variance_model(df, r, "series", nb2, seed, interaction=False), "table08c2_variance_model_main_effects")
    save_table(V.cf_geometry_table(G, nb, seed), "table08d_cf_geometry_level")
    cfg_ = []
    cfv, gcl = G.cf_max.to_numpy(float), G.series.to_numpy()
    for col in ("abs_mean_resid", "rmse_grouped", "generalisation_gap"):
        x = G[col].to_numpy(float)
        pt, lo, hi, reps = cluster_bootstrap(lambda i: [stats.spearmanr(x[i], cfv[i], nan_policy="omit").statistic], gcl, nb, seed)
        cfg_.append({"geometry_level_quantity": col, "spearman_with_CF": pt[0], "lo": lo[0], "hi": hi[0],
                     "boot_p": max(2 * min((reps[:, 0] <= 0).mean(), (reps[:, 0] >= 0).mean()), 1 / nb), "n_geometries": len(G)})
    save_table(pd.DataFrame(cfg_), "table08e_cf_geometry_level_trends")
    save_table(V.joint_table(df, r, "series", min(nb, 500), seed), "table10_rf_cf_joint")
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
    save_table(pd.DataFrame(cons), "table08f_rf_cf_vs_completeness_indicators")
    # zero-overtopping share by RF/CF (all wet/dry records)
    w = m[m.use_wetdry]
    z = [{"factor": fac, "category": v, "n": int((w[fac] == v).sum()), "pct_zero": 100 * (w.loc[w[fac] == v, "response_class"] == "zero").mean()}
         for fac in ("RF", "CF") for v in (1, 2, 3, 4)]
    save_table(pd.DataFrame(z), "table08g_zero_share_by_rf_cf")
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
    save_table(pd.DataFrame(rr), "table08h_repeatability_near_replicates")


# ============================================================================== Experiment 8
def exp_decomp(D, nb, seed):
    df, y = D["df"], D["y"]
    scheme = D["cfg"]["experiment"]["primary_scheme"]
    nbd = min(nb, 100)
    rows = []

    def add(label, r, sub, levels, n_boot=nbd):
        t, fit = decompose(r, sub, levels, n_boot, seed)
        t.insert(0, "analysis", label)
        t.insert(1, "n_records", len(sub))
        rows.append(t)
        return fit
    r = y - load_pred(f"{scheme}__{REF_MODEL}")[0, :, 0]
    fit = add("PRIMARY: XGBoost, geometry-grouped OOF residuals; lab-unit / series / geometry", r, df, ["lab_unit", "series", "geometry"])
    eff = pd.DataFrame({"geometry": fit["labels"][2], "u_geometry": fit["effects"][2]})
    eff.to_csv(out("diagnostics", "geometry_random_effects.csv"), index=False)
    add("grouping sensitivity: geometry_coarse", r, df, ["lab_unit", "series", "geometry_coarse"], 0)
    add("grouping sensitivity: substructure", r, df, ["lab_unit", "series", "substructure"], 0)
    add("grouping sensitivity: series / geometry only", r, df, ["series", "geometry"], 0)
    known = (df["lab"] != "unknown").to_numpy()
    add("records with an inferred laboratory only: lab / series / geometry", r[known], df[known], ["lab", "series", "geometry"], 0)
    rr = y - load_pred(f"random__{REF_MODEL}")[0, :, 0]
    add("random-CV residuals (optimistic reference)", rr, df, ["lab_unit", "series", "geometry"], 0)
    for m in ("mlp", "ccw_full", "rf"):
        if have(f"{scheme}__{m}"):
            add(f"model sensitivity: {m}", y - load_pred(f"{scheme}__{m}").mean(0)[:, 0], df, ["lab_unit", "series", "geometry"], 0)
    for v in (1, 2, 3, 4):
        mk = (df["CF"] == v).to_numpy()
        add(f"stratum CF={v}", r[mk], df[mk], ["series", "geometry"], nbd)
    for v in (1, 2, 3, 4):
        mk = (df["RF"] == v).to_numpy()
        add(f"stratum RF={v}", r[mk], df[mk], ["series", "geometry"], nbd)
    save_table(pd.concat(rows), "table09_hierarchical_error_decomposition")
    # model-induced part: total OOF MSE vs. model-free repeatability variance
    rp = pd.read_csv(ROOT / "results/diagnostics/near_replicate_pairs.csv") if (ROOT / "results/diagnostics/near_replicate_pairs.csv").exists() else replicate_pairs(df)
    rep_var = float(np.mean(rp.dy ** 2) / 2) if len(rp) else np.nan
    prim = rows[0]
    summ = pd.DataFrame([{"quantity": "total OOF residual variance (geometry-grouped)", "value": float(np.var(r))},
                         {"quantity": "total OOF residual variance (random CV)", "value": float(np.var(rr))},
                         {"quantity": "within-geometry residual variance (REML)", "value": float(prim.variance.iloc[-1])},
                         {"quantity": "near-replicate repeatability variance (model-free, pooled)", "value": rep_var},
                         {"quantity": "n near-replicate pairs", "value": len(rp)},
                         {"quantity": "share of within-geometry variance attributable to repeatability", "value": rep_var / float(prim.variance.iloc[-1])}])
    save_table(summ, "table09b_repeatability_vs_residual")


# ============================================================================== Experiment 9
def exp_unc(D, nb, seed):
    df, y = D["df"], D["y"]
    cfg = D["cfg"]
    scheme = cfg["experiment"]["primary_scheme"]
    fold, groups = D["folds"][scheme].to_numpy(), groups_for(df, scheme)
    X = D["X_views"][:, 0]
    rf, cf = df["RF"].to_numpy(int), df["CF"].to_numpy(int)
    params = cfg["model"]["xgboost"]["grid"][1]   # pre-specified, not tuned on test folds
    alphas = [0.5, 0.2, 0.1, 0.05]
    reps = 3
    jobs = [delayed(U.run_fold)(X, y, rf, cf, groups, np.flatnonzero(fold != k), np.flatnonzero(fold == k), params, alphas,
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
    save_table(t, "table11_quality_aware_uncertainty")
    save_table(strat, "table11b_coverage_by_rf_cf_cell")
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
    save_table(pd.DataFrame(wr), "table11c_interval_width_by_rf_cf")


# ============================================================================== Experiment 10
def exp_filter(D, nb, seed):
    df, y = D["df"], D["y"]
    cl = df["geometry"].to_numpy()
    rf, cf = df["RF"].to_numpy(), df["CF"].to_numpy()
    subsets = {"all records": np.ones(len(y), bool), "RF<4 and CF<4": (rf < 4) & (cf < 4), "RF=4": rf == 4, "CF=4": cf == 4,
               "RF=4 or CF=4": (rf == 4) | (cf == 4)}
    rows = []
    for model in ("xgb", "rf"):
        if not have(f"filter__{model}__full"):
            continue
        base = load_pred(f"filter__{model}__full")[0, :, 0]
        for v in ["full", "excl_rf4", "excl_cf4", "excl_rf4_or_cf4", "clash_weight_(4-RF)(4-CF)", "soft_weight_(5-RF)(5-CF)", "excl_remark_do_not_use"]:
            if not have(f"filter__{model}__{v}"):
                continue
            p = load_pred(f"filter__{model}__{v}")[0, :, 0]
            ntrain = {"full": len(y), "excl_rf4": int((rf != 4).sum()), "excl_cf4": int((cf != 4).sum()),
                      "excl_remark_do_not_use": int((~df.flag_remark_do_not_use).sum())}.get(v, int(((rf != 4) & (cf != 4)).sum()) if "excl" in v or "clash" in v else len(y))
            for sname, mk in subsets.items():
                i0 = np.flatnonzero(mk)
                e1, e0 = (p[i0] - y[i0]) ** 2, (base[i0] - y[i0]) ** 2
                pt, lo, hi, _ = cluster_bootstrap(lambda i: [np.sqrt(e1[i].mean()), np.sqrt(e1[i].mean()) - np.sqrt(e0[i].mean())], cl[i0], min(nb, 500), seed)
                rows.append({"learner": model, "training_set": v, "n_train_eligible": ntrain, "test_subset": sname, "n_test": int(mk.sum()),
                             **{k: regression_metrics(y[i0], p[i0])[k] for k in ("rmse", "mae", "r2", "bias")}, "rmse_lo": lo[0], "rmse_hi": hi[0],
                             "rmse_diff_vs_full": pt[1], "diff_lo": lo[1], "diff_hi": hi[1]})
    save_table(pd.DataFrame(rows), "table12_category4_filtering_sensitivity")


EXPS = {"perf": exp_perf, "ccw": exp_ccw, "audit": exp_audit, "rfcf": exp_rfcf, "decomp": exp_decomp, "unc": exp_unc, "filter": exp_filter}


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
