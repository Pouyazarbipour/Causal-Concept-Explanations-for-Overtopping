"""Figures of the revision (Times New Roman / Liberation Serif, Elsevier column widths).

Usage: python -m src.plotting_revision        -> revision/figures/{main,supplementary}/*.png|pdf
Each figure is drawn only from revision/tables (and original tables where stated); no value is edited.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import plotting as P0   # style, palette and helpers of the original figures (rcParams are set on import)
from .utils import ROOT

T = ROOT / "revision" / "tables"
FIG = ROOT / "revision" / "figures"
CAT, SEQ, INK, INK2, MUTED = P0.CAT, P0.SEQ, P0.INK, P0.INK2, P0.MUTED
panel, err = P0.panel, P0.err


def tab(n):
    return pd.read_csv(T / f"{n}.csv")


def save(fig, name, where="main"):
    d = FIG / where
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(d / f"{name}.pdf")
    fig.savefig(d / f"{name}.png", dpi=400)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- R1 data accounting
def fig_accounting():
    ex = tab("S_accounting_exclusive")
    ex = ex[ex.category != "TOTAL"].sort_values("n")
    m = tab("S_missingness_by_rf_cf")
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axs[0]
    y = np.arange(len(ex))
    ax.barh(y, ex.n, color=CAT[0], height=0.6, label="all records")
    ax.plot(ex.RF4.clip(lower=0.8), y, "D", color=CAT[1], ms=4, label="of which RF = 4")
    for i, (n, r4) in enumerate(zip(ex.n, ex.RF4)):
        ax.text(n * 1.25, i, f"{int(n):,}".replace(",", " ") + f" (RF=4: {int(r4)})", va="center", fontsize=7)
    ax.set_xscale("log"); ax.set_xlim(0.7, 3e6)
    ax.set_yticks(y); ax.set_yticklabels([c.replace(" (wet/dry + regression)", "\n(wet/dry + regression)").replace(" (wet/dry only)", "\n(wet/dry only)")
                                          for c in ex.category], fontsize=7.5)
    ax.set_xlabel("Number of records (log scale)")
    ax.legend(loc="lower right")
    panel(ax, "a", "Record accounting (n = 10 532)")
    ax = axs[1]
    for k, (fac, col) in enumerate((("RF", CAT[1]), ("CF", CAT[0]))):
        d = m[m.factor == fac]
        ax.bar(d.category + (k - 0.5) * 0.38, d.pct_unusable, width=0.36, color=col, label=fac)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("Category"); ax.set_ylabel("Records with unusable inputs [%]")
    ax.legend()
    panel(ax, "b", "Unusable inputs by rating")
    fig.tight_layout()
    save(fig, "Fig01_data_accounting")


# --------------------------------------------------------------------------------------------- R2 model comparison
def fig_models():
    M = tab("M1_model_comparison")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    M = M.sort_values("ensemble_rmse", ascending=False).reset_index(drop=True)
    y = np.arange(len(M))
    col = [CAT[1] if "baseline" in m else CAT[0] for m in M.model]
    for i, r in M.iterrows():
        err(ax, [r.ensemble_rmse], [i], None, None, col[i]) if False else None
        ax.errorbar(r.ensemble_rmse, i, xerr=[[r.ensemble_rmse - r.ensemble_rmse_lo], [r.ensemble_rmse_hi - r.ensemble_rmse]], fmt="o", color=col[i], ms=4, capsize=2)
    ax.set_yticks(y); ax.set_yticklabels(M.model, fontsize=7.5); ax.set_xlabel("Out-of-fold RMSE [log10 units], held-out geometries")
    panel(ax, "a", "Baseline (orange) and optimised (blue) models")
    ax = axs[1]
    for i, r in M.iterrows():
        ax.errorbar(r.d_rmse_vs_xgb_baseline, i, xerr=[[r.d_rmse_vs_xgb_baseline - r.d_lo], [r.d_hi - r.d_rmse_vs_xgb_baseline]], fmt="o", color=col[i], ms=4, capsize=2)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([]); ax.set_xlabel("ΔRMSE vs baseline XGBoost (paired)")
    panel(ax, "b", "Paired difference, 95 % CI")
    fig.tight_layout()
    save(fig, "Fig03_model_comparison")
    b = T / "M3_xgb_budget_sensitivity.csv"
    if b.exists():
        B = pd.read_csv(b)
        B = B[B.budget_trials.astype(str).str.isnumeric()].astype({"budget_trials": int}).sort_values("budget_trials")
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        ax.plot(B.budget_trials, B.inner_rmse_mean, "o-", color=MUTED, label="inner CV objective")
        ax.plot(B.budget_trials, B.outer_rmse, "o-", color=CAT[0], label="outer test RMSE")
        ax.set_xlabel("Optuna trials per outer fold"); ax.set_ylabel("RMSE [log10 units]"); ax.legend()
        panel(ax, "a", "XGBoost v2: search budget")
        fig.tight_layout()
        save(fig, "FigS02_budget_sensitivity", "supplementary")


# --------------------------------------------------------------------------------------------- R3 robustness of RF/CF
def _logx(ax, ticks=(0.25, 0.5, 1, 2, 4, 8)):
    from matplotlib.ticker import FixedLocator, NullFormatter, FixedFormatter
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def fig_robust():
    R1 = tab("R1_variance_ratios_by_learner_and_scheme")
    G1 = tab("G1_geometry_definition_sensitivity") if (T / "G1_geometry_definition_sensitivity.csv").exists() else None
    R3 = tab("R3_repeatability_tolerance_sensitivity")
    R4 = tab("R4_leave_one_series_out_CF")
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 3.3))
    ax = axs[0]
    labs = list(R1.residuals.unique())
    for j, (term, col) in enumerate((("RF=4", CAT[1]), ("CF=3", CAT[0]), ("CF=4", CAT[2]))):
        d = R1[R1.term == term].set_index("residuals").reindex(labs)
        yy = np.arange(len(labs)) + (j - 1) * 0.22
        ax.errorbar(d.variance_ratio, yy, xerr=[d.variance_ratio - d.ratio_lo, d.ratio_hi - d.variance_ratio], fmt="o", ms=3.5, capsize=1.5, color=col, label=term)
    ax.axvline(1, color=INK, lw=0.8); _logx(ax)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels([s.replace(" (original)", "").replace(", geometry-grouped", " (geom.)") for s in labs], fontsize=6.8)
    ax.set_xlabel("Residual-variance ratio vs category 1"); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    panel(ax, "a", "Learners and schemes")
    ax = axs[1]
    d = R3[(R3.pairs == "all pairs") & R3.subset.str.startswith("max RF")]
    for j, tol in enumerate(sorted(d.rel_tol.unique())):
        s = d[d.rel_tol == tol]
        xx = np.arange(1, 5) + (j - 1.5) * 0.15
        ax.errorbar(xx, s.repeat_sd, yerr=[s.repeat_sd - s.lo, s.hi - s.repeat_sd], fmt="o", ms=3, capsize=1.5, color=SEQ[j], label=f"{100 * tol:g} %")
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("max RF of the pair"); ax.set_ylabel("Repeat SD [log10]")
    ax.legend(title="tolerance", fontsize=7, title_fontsize=7)
    panel(ax, "b", "Repeatability vs tolerance")
    ax = axs[2]
    rows = []
    for cat in (4, 3):
        s_ = R4[R4.CF_category == cat].copy()
        base = s_[s_.left_out_series == "none"].variance_ratio_vs_CF1.iloc[0]
        s_["infl"] = (np.log(s_.variance_ratio_vs_CF1) - np.log(base)).abs()
        keep = pd.concat([s_[s_.left_out_series == "none"], s_[s_.left_out_series != "none"].nlargest(4, "infl")])
        rows.append((cat, keep))
    yy, lab = 0, []
    for cat, keep in rows:
        col = CAT[2] if cat == 4 else CAT[0]
        for _, r in keep.iterrows():
            ax.errorbar(r.variance_ratio_vs_CF1, yy, xerr=[[r.variance_ratio_vs_CF1 - r.lo], [r.hi - r.variance_ratio_vs_CF1]], fmt="o", ms=3.5, capsize=1.5, color=col)
            lab.append(f"CF{cat}: " + ("all series" if r.left_out_series == "none" else f"without {r.left_out_series}"))
            yy += 1
        yy += 0.6
        lab.append("")
    ticks = [i for i, l in enumerate([x for x in lab]) if l]
    pos, t = [], 0
    for l in lab:
        if l:
            pos.append(t)
            t += 1
        else:
            t += 0.6
    ax.set_yticks(pos); ax.set_yticklabels([l for l in lab if l], fontsize=7); ax.invert_yaxis()
    ax.axvline(1, color=INK, lw=0.8); _logx(ax); ax.set_xlabel("Variance ratio vs CF = 1")
    panel(ax, "c", "Leave one series out")
    fig.tight_layout()
    save(fig, "Fig05_rf_cf_robustness")


# --------------------------------------------------------------------------------------------- R4 policies fixed vs tuned
def fig_policies():
    P = tab("P1_policies_v2")
    pols = ["P2_excl_RF4", "P3_excl_CF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks", "P6a_inverse_variance_weights", "P6c_weights_and_intervals"]
    labels = ["P2 excl. RF=4", "P3 excl. CF=4", "P4 excl. RF/CF=4", "P5 (4−RF)(4−CF)", "P5b P5 + remarks", "P6a 1/σ² weights", "P6c P6a + P6b"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    for ax, (s, title), letter in zip(axs, (("common", "Common tests (RF<4, CF<4)"), ("RF4", "RF = 4 tests"), ("CF4", "CF = 4 tests")), "abc"):
        for j, (mode, col) in enumerate((("fixed", CAT[0]), ("tuned", CAT[1]))):
            d = P[P["mode"] == mode].set_index("policy").reindex(pols)
            yy = np.arange(len(pols)) + (j - 0.5) * 0.3
            ax.errorbar(d[f"d_rmse_{s}"], yy, xerr=[d[f"d_rmse_{s}"] - d[f"d_rmse_{s}_lo"], d[f"d_rmse_{s}_hi"] - d[f"d_rmse_{s}"]], fmt="o", ms=3.5,
                        capsize=1.5, color=col, label="fixed hyper-parameters" if mode == "fixed" else "policy-specific tuning")
        ax.axvline(0, color=INK, lw=0.8)
        ax.set_xlabel("ΔRMSE vs P1 [log10]")
        panel(ax, letter, title)
    axs[0].set_yticks(range(len(pols))); axs[0].set_yticklabels(labels, fontsize=7.5); axs[0].invert_yaxis()
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=2, fontsize=7.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "Fig07_policies_fixed_vs_tuned")


# --------------------------------------------------------------------------------------------- R5 conformal
def fig_conformal():
    C1, C2 = tab("C1_conformal_sensitivity"), tab("C2_conformal_coverage_by_category")
    scales = ["constant", "RF+CF main effects", "RFxCF shrunk cells", "CF only", "physical only", "physical+RF+CF"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    for ax, fac, letter in ((axs[0], "RF", "a"), (axs[1], "CF", "b")):
        for j, s in enumerate(scales):
            d = C2[(C2.scale == s) & C2.subset.str.startswith(fac)]
            xx = np.arange(1, 5) + (j - 2.5) * 0.11
            ax.errorbar(xx, d.coverage, yerr=[d.coverage - d.lo, d.hi - d.coverage], fmt="o", ms=2.8, capsize=1, color=CAT[j], label=s)
        ax.axhline(0.9, color=INK, lw=0.8, ls="--")
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel(fac); ax.set_ylabel("Coverage of 90 % intervals"); ax.set_ylim(0.55, 1.0)
        panel(ax, letter, f"Coverage by {fac} (c = 0.25)")
    axs[0].legend(fontsize=6, loc="lower left")
    ax = axs[2]
    for j, s in enumerate(scales):
        d = C1[C1.scale == s].sort_values("calib_fraction")
        ax.plot(d.calib_fraction, d.max_abs_dev_RF * 100, "o-", ms=3, color=CAT[j])
    ax.set_xlabel("Calibration fraction"); ax.set_ylabel("Max coverage deviation over RF [pts]")
    panel(ax, "c", "Sensitivity")
    fig.tight_layout()
    save(fig, "Fig08_conformal_quality_aware")


# --------------------------------------------------------------------------------------------- R6 two-stage
def fig_two_stage():
    from sklearn.metrics import roc_curve
    o = pd.read_csv(ROOT / "revision/outputs/wet_dry/two_stage_oof.csv")
    W1, W2 = tab("W1_stageA_classification"), tab("W2_stageA_reliability")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax = axs[0]
    for v in (1, 2, 3, 4):
        d = o[o.RF == v]
        f, t_, _ = roc_curve(d.wet, d.p_wet)
        auc = W1.set_index("subset").loc[f"RF={v}", "auc"]
        ax.plot(f, t_, color=SEQ[v - 1], label=f"RF={v}  AUC {auc:.2f}")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls="--"); ax.set_xlabel("False-wet rate"); ax.set_ylabel("True-wet rate"); ax.legend(fontsize=6.8)
    panel(ax, "a", "Occurrence (stage A)")
    ax = axs[1]
    ax.plot(W2.mean_pred, W2.obs_wet, "o-", color=CAT[0]); ax.plot([0, 1], [0, 1], color=MUTED, lw=0.7, ls="--")
    ax.set_xlabel("Predicted P(wet)"); ax.set_ylabel("Observed wet fraction")
    panel(ax, "b", "Reliability")
    ax = axs[2]
    d = W1[W1.subset.str.startswith(("RF", "CF"))]
    xx = np.arange(len(d))
    ax.errorbar(xx, d.auc, yerr=[d.auc - d.auc_lo, d.auc_hi - d.auc], fmt="o", ms=3.5, capsize=1.5, color=CAT[0])
    ax.set_xticks(xx); ax.set_xticklabels(d.subset, rotation=60, fontsize=7); ax.set_ylabel("ROC AUC [95 % CI]")
    panel(ax, "c", "AUC by rating")
    fig.tight_layout()
    save(fig, "Fig04_two_stage_wet_dry")


# --------------------------------------------------------------------------------------------- R7 engineering
def fig_engineering():
    E1, E2, E4 = tab("E1_eurotop_benchmark"), tab("E2_exceedance_decisions_by_policy"), tab("E4_regimes_of_category4_tests")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.2, 1.2, 1.1]})
    ax = axs[0]
    d = E1[E1.subset == "all RF/CF"]
    doms = list(d.domain.unique())
    mods = [m for m in d.model.unique() if "Random" not in m]
    for j, m in enumerate(mods):
        s = d[d.model == m].set_index("domain").reindex(doms)
        ax.barh(np.arange(len(doms)) + (j - 1) * 0.25, s.rmse, height=0.24, color=CAT[j], label=m)
    ax.set_yticks(range(len(doms))); ax.set_yticklabels([f"{x}\n(n={int(d[d.domain == x].n.iloc[0])})" for x in doms], fontsize=6.8)
    ax.set_xlabel("RMSE [log10 q*]")
    h, l_ = ax.get_legend_handles_labels()
    fig.legend(h, [x.replace(" (original OOF)", "") for x in l_], loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.3, -0.04))
    panel(ax, "a", "EurOtop domain")
    ax = axs[1]
    e = E2[(E2.Hm0_prototype_m == 2.0) & (E2.subset == "RF4")]
    for j, p in enumerate(["P1_retain_all", "P2_excl_RF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P6a_inverse_variance_weights"]):
        s = e[e.policy == p].sort_values("q_allow_l_per_s_per_m")
        ax.plot(s.q_allow_l_per_s_per_m, s.false_safe_rate_among_exceed, "o-", ms=3, color=CAT[j], label=p.split("_")[0])
    ax.set_xscale("log"); ax.set_xlabel("q_allow [l/s/m] at Hm0 = 2 m"); ax.set_ylabel("False-safe rate among exceedances [%]")
    ax.legend(fontsize=6.8)
    panel(ax, "b", "RF = 4 tests: missed exceedance")
    ax = axs[2]
    r = E4.sort_values("pct_regime_lost_by_P4")
    ax.barh(range(len(r)), r.pct_regime_lost_by_P4, color=CAT[1])
    ax.set_yticks(range(len(r))); ax.set_yticklabels(r.regime, fontsize=6.5); ax.set_xlabel("% of regime removed by P4")
    panel(ax, "c", "Regimes lost by exclusion")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "Fig09_engineering_decisions")


def make_all():
    done, failed = [], []
    for name, fn in (("accounting", fig_accounting), ("models", fig_models), ("robust", fig_robust), ("policies", fig_policies),
                     ("conformal", fig_conformal), ("two_stage", fig_two_stage), ("engineering", fig_engineering)):
        try:
            fn()
            done.append(name)
        except FileNotFoundError as e:
            failed.append((name, f"missing input: {e.filename}"))
        except Exception as e:  # noqa: BLE001
            failed.append((name, repr(e)))
    return done, failed


if __name__ == "__main__":
    print(make_all())
