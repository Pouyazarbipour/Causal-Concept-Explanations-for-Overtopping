"""Publication figures (vector PDF + 300 dpi PNG).  All figures read saved tables/predictions only.

Colour roles: ordinal RF/CF categories -> one-hue sequential ramp (light = 1, dark = 4);
models/methods -> fixed categorical order; signed quantities -> blue/orange diverging with grey midpoint.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .utils import ROOT, out

SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]            # categories 1..4
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f2a184", "#c94a1c"])
MODEL_COLOR = {"Random Forest": CAT[5], "XGBoost": CAT[0], "Monotonic XGBoost": CAT[2], "MLP": CAT[3], "Concept bottleneck": CAT[4],
               "Concept wrapper (unaligned)": CAT[6], "CCW-Net (full)": CAT[1], "CCW backbone (tabular MLP)": MUTED}
T = ROOT / "results" / "tables"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.edgecolor": AXIS,
                     "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7,
                     "figure.facecolor": "white", "savefig.bbox": "tight", "pdf.fonttype": 42, "mathtext.default": "regular"})


def tab(name):
    return pd.read_csv(T / f"{name}.csv")


def save(fig, name):
    fig.savefig(out("figures", name + ".pdf"))
    fig.savefig(out("figures", name + ".png"), dpi=300)
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold")


def err(ax, x, y, lo, hi, color, **kw):
    ax.errorbar(x, y, yerr=[np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)], fmt="o", color=color, ms=4.5, lw=1.2,
                capsize=2.5, **kw)


def _box(ax, x, y, w, h, text, fc="#f3f7fd", ec="#1c5cab", fs=7, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=INK)


def _arrow(ax, a, b, color=INK2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8, lw=0.9, color=color, linestyle=ls, shrinkA=1, shrinkB=1))


# ----------------------------------------------------------------------------------------- Fig 1
def fig01():
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.02, 0.80, 0.22, 0.16, "Original CLASH database\n10 532 tests, RF and CF 1-4\n(no record removed)", bold=True)
    _box(ax, 0.30, 0.80, 0.30, 0.16, "Audit, flags, response classes\nq* = q / (g H$_{m0}^3$)$^{1/2}$,  Y = log$_{10}$ q*\ngeometry reconstruction, folds")
    _box(ax, 0.66, 0.80, 0.32, 0.16, "Quality metadata kept apart\nRF (test level), CF (structure level)\nnever inputs of the mean model", fc="#fdf1ec", ec="#c94a1c")
    _box(ax, 0.02, 0.50, 0.28, 0.18, "Geometry-grouped nested CV\nRF, XGBoost, monotone XGBoost, MLP\nout-of-fold predictions and residuals")
    _box(ax, 0.36, 0.50, 0.28, 0.18, "Concept ontology (10 groups)\nCCW-Net: vector concepts, mediator flow,\nnecessity / sufficiency / independence")
    _box(ax, 0.70, 0.50, 0.28, 0.18, "Counterfactual physical audit\nA input interventions  B representation\ninterventions  C matched measured pairs")
    _box(ax, 0.02, 0.18, 0.28, 0.20, "RF / CF predictive validity\nordinal trends, variance models,\nrepeatability, geometry-level error", fc="#fdf1ec", ec="#c94a1c")
    _box(ax, 0.36, 0.18, 0.28, 0.20, "Hierarchical error decomposition\nsource/laboratory, series, geometry,\nwithin-geometry residual", fc="#fdf1ec", ec="#c94a1c")
    _box(ax, 0.70, 0.18, 0.28, 0.20, "Quality-aware uncertainty\ngroup-calibrated conformal intervals,\ncategory-4 filtering sensitivity", fc="#fdf1ec", ec="#c94a1c")
    for a, b in (((0.24, 0.88), (0.30, 0.88)), ((0.45, 0.80), (0.16, 0.68)), ((0.45, 0.80), (0.50, 0.68)), ((0.30, 0.59), (0.36, 0.59)),
                 ((0.64, 0.59), (0.70, 0.59)), ((0.16, 0.50), (0.16, 0.38)), ((0.16, 0.50), (0.48, 0.38)), ((0.30, 0.28), (0.36, 0.28)),
                 ((0.64, 0.28), (0.70, 0.28)), ((0.84, 0.50), (0.84, 0.38))):
        _arrow(ax, a, b)
    for a, b in (((0.82, 0.80), (0.20, 0.385)), ((0.86, 0.80), (0.88, 0.385))):
        _arrow(ax, a, b, color="#c94a1c", ls="--")
    ax.text(0.5, 0.06, "Blue: physical prediction and concept learning.   Orange (dashed links): quality-assessment layer, evaluated against\n"
            "out-of-fold errors but excluded from the physical mean predictor.", ha="center", fontsize=7, color=INK2)
    save(fig, "fig01_research_framework")


# ----------------------------------------------------------------------------------------- Fig 2
def fig02():
    d = tab("table02_rf_cf_distribution")
    j = pd.read_csv(T / "table02b_rf_cf_joint_all.csv", index_col=0)
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, k, letter in zip(axs[:2], ("RF", "CF"), "ab"):
        s = d[(d.factor == k) & d.category.astype(str).isin(list("1234"))]
        ax.bar(s.category.astype(str), s.n_all, color=SEQ, width=0.7)
        for x, (n, p) in enumerate(zip(s.n_all, s.pct_all)):
            ax.text(x, n, f"{int(n)}\n{p:.1f} %", ha="center", va="bottom", fontsize=7)
        ax.set_ylim(0, s.n_all.max() * 1.25); ax.set_xlabel(f"{k} category"); ax.set_ylabel("number of tests")
        panel(ax, letter, "Reliability factor RF" if k == "RF" else "Complexity factor CF")
        ax.grid(axis="x", visible=False)
    ax = axs[2]
    M = j.iloc[:4, :4].to_numpy(float)
    ax.imshow(np.log10(M), cmap=SEQ_CMAP, aspect="auto")
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f"{int(M[a, b])}\n{100 * M[a, b] / M.sum():.1f} %", ha="center", va="center", fontsize=7,
                    color="white" if M[a, b] > 600 else INK)
    ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_yticks(range(4), [1, 2, 3, 4]); ax.set_xlabel("CF"); ax.set_ylabel("RF"); ax.grid(False)
    panel(ax, "c", "Joint RF x CF frequency (log colour scale)")
    fig.tight_layout()
    save(fig, "fig02_rf_cf_distribution")


# ----------------------------------------------------------------------------------------- Fig 3
def fig03(D):
    from .pipeline import load_pred
    df, y = D["df"], D["y"]
    r = y - load_pred("geometry__xgb")[0, :, 0]
    s = tab("table08a_residuals_by_rf_cf")
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for row, k in enumerate(("RF", "CF")):
        ax = axs[row, 0]
        parts = ax.violinplot([r[df[k] == v] for v in (1, 2, 3, 4)], positions=range(4), showextrema=False, widths=0.8)
        for b, c in zip(parts["bodies"], SEQ):
            b.set_facecolor(c); b.set_alpha(0.85); b.set_edgecolor("white")
        q = [np.percentile(r[df[k] == v], [25, 50, 75]) for v in (1, 2, 3, 4)]
        ax.vlines(range(4), [a[0] for a in q], [a[2] for a in q], color=INK, lw=1.2); ax.scatter(range(4), [a[1] for a in q], color="white", edgecolor=INK, s=14, zorder=3)
        ax.axhline(0, color=AXIS, lw=0.8); ax.set_ylim(-2.5, 2.5)
        ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_xlabel(k); ax.set_ylabel(r"residual $y-\hat{y}^{OOF}$  [log$_{10}$]")
        panel(ax, "ad"[row], f"Residual distribution by {k}")
        t = s[s.factor == k]
        for ax, col, lo, hi, lab, letter in ((axs[row, 1], "mae", "mae_lo", "mae_hi", "mean |residual|  [log$_{10}$]", "be"[row]),
                                             (axs[row, 2], "within_geometry_sd", "within_sd_lo", "within_sd_hi", "within-geometry residual SD", "cf"[row])):
            for i in range(4):
                err(ax, [i], [t[col].iloc[i]], [t[lo].iloc[i]], [t[hi].iloc[i]], SEQ[i])
                ax.text(i + 0.12, t[col].iloc[i], f"{t[col].iloc[i]:.2f}", va="center", fontsize=7, color=INK2)
            ax.set_xticks(range(4), [f"{v}\nn={n}" for v, n in zip((1, 2, 3, 4), t.n)]); ax.set_xlim(-0.5, 3.8)
            ax.set_xlabel(k); ax.set_ylabel(lab); ax.set_ylim(0, None)
            panel(ax, letter, "Absolute error" if col == "mae" else "Dispersion within geometry")
    fig.tight_layout()
    save(fig, "fig03_rf_cf_vs_prediction_error")


# ----------------------------------------------------------------------------------------- Fig 4 / 10
def fig04_10():
    t = tab("table09_hierarchical_error_decomposition")
    G = pd.read_csv(ROOT / "results/diagnostics/geometry_level_residuals.csv")
    comp_color = {"lab_unit": CAT[3], "lab": CAT[3], "series": CAT[2], "geometry": CAT[0], "geometry_coarse": CAT[0], "substructure": CAT[0],
                  "within-geometry residual": MUTED}
    nice = {"lab_unit": "source / laboratory unit", "lab": "source / laboratory unit", "series": "test series", "geometry": "geometry",
            "geometry_coarse": "geometry", "substructure": "geometry", "within-geometry residual": "within-geometry residual"}

    def stacked(ax, labels, analyses, horizontal=True):
        seen = set()
        for i, a in enumerate(analyses):
            s = t[t.analysis == a]
            left = 0
            for _, rw in s.iterrows():
                lab = nice[rw.component]
                ax.barh(i, rw.share, left=left, color=comp_color[rw.component], edgecolor="white", lw=1.2, height=0.65,
                        label=lab if lab not in seen else None)
                seen.add(lab)
                if rw.share > 0.07:
                    ax.text(left + rw.share / 2, i, f"{100 * rw.share:.0f}", ha="center", va="center", fontsize=7,
                            color=INK if rw.component in ("within-geometry residual", "lab_unit", "lab") else "white")
                left += rw.share
        ax.set_yticks(range(len(analyses)), labels); ax.set_xlim(0, 1); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlabel("share of out-of-fold residual variance")
    an = list(dict.fromkeys(t.analysis))
    prim = an[0]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    sel = [prim] + [a for a in an if a.startswith("grouping sensitivity")] + [a for a in an if a.startswith("random-CV")]
    stacked(axs[0], ["primary", "coarse geometry", "sub-structure", "series/geometry", "random CV"][:len(sel)], sel)
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2); panel(axs[0], "a", "Variance components")
    p = t[t.analysis == prim]
    ax = axs[1]
    for i, (_, rw) in enumerate(p.iterrows()):
        err(ax, [i], [rw.sd], [np.sqrt(rw.variance_lo)], [np.sqrt(rw.variance_hi)], comp_color[rw.component])
    ax.set_xticks(range(len(p)), ["source/\nlab", "series", "geometry", "within\ngeometry"][:len(p)]); ax.set_ylabel("component SD  [log$_{10}$]"); ax.set_ylim(0, None)
    panel(ax, "b", "Component SD, 95 % cluster-bootstrap CI")
    ax = axs[2]
    g = G[G.n >= 3]
    ax.hist(g.mean_resid, bins=np.linspace(-2, 2, 41), color=CAT[0], edgecolor="white", lw=0.4)
    ax.axvline(0, color=AXIS, lw=0.8); ax.set_xlabel("geometry mean residual  [log$_{10}$]"); ax.set_ylabel("number of geometries (n$\\geq$3 tests)")
    panel(ax, "c", "Geometry-level heterogeneity")
    fig.tight_layout()
    save(fig, "fig04_geometry_level_error_structure")

    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for ax, k, letter in zip(axs[:2], ("CF", "RF"), "ab"):
        for comp, col, off in (("geometry", CAT[0], -0.1), ("within-geometry residual", MUTED, 0.1)):
            s = pd.concat([t[(t.analysis == f"stratum {k}={v}") & (t.component == comp)] for v in (1, 2, 3, 4)])
            err(ax, np.arange(4) + off, s.sd, np.sqrt(s.variance_lo), np.sqrt(s.variance_hi), col, label=nice[comp])
        ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_xlabel(k); ax.set_ylabel("component SD  [log$_{10}$]"); ax.set_ylim(0, None)
        ax.legend(loc="upper left"); panel(ax, letter, f"Components within {k} strata")
    ax = axs[2]
    rp = tab("table08h_repeatability_near_replicates")
    s = rp[rp.subset.str.startswith("max RF") & rp.repeat_sd_log10.notna()] if "repeat_sd_log10" in rp else rp.iloc[:0]
    xs = [int(v[-1]) - 1 for v in s.subset]
    for x, (_, rw) in zip(xs, s.iterrows()):
        err(ax, [x], [rw.repeat_sd_log10], [rw.sd_lo], [rw.sd_hi], SEQ[x])
        ax.text(x, rw.sd_hi, f"n={int(rw.n_pairs)}", ha="center", va="bottom", fontsize=6.5, color=INK2)
    wr = t[(t.analysis == prim) & (t.component == "within-geometry residual")].sd.iloc[0]
    ax.axhline(wr, color=MUTED, lw=1, ls="--"); ax.text(3.45, wr, "within-geometry\nresidual SD", fontsize=6.5, color=INK2, va="bottom", ha="right")
    ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_xlim(-0.5, 3.5); ax.set_xlabel("max RF of the pair"); ax.set_ylabel("repeat SD of log$_{10}$ q*"); ax.set_ylim(0, None)
    panel(ax, "c", "Near-replicate measured tests")
    fig.tight_layout()
    save(fig, "fig10_error_decomposition_by_quality")


# ----------------------------------------------------------------------------------------- Fig 5
def fig05():
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.01, 0.50, 0.13, 0.22, "15 dimensionless\nphysical inputs X\n(no RF, CF)")
    _box(ax, 0.18, 0.50, 0.13, 0.22, "Tabular MLP\nbackbone f\n(pre-trained,\nfrozen)  Z")
    _box(ax, 0.35, 0.50, 0.13, 0.22, "Concept\nencoder g\n+ squash")
    for i, nm in enumerate(["freeboard", "steepness", "depth", "obliquity", "...", "foreshore"]):
        _box(ax, 0.52, 0.84 - i * 0.105, 0.13, 0.085, f"M: {nm}" if nm != "..." else "...", fc="#eaf7f1", ec="#1baf7a", fs=6.5)
    _box(ax, 0.70, 0.56, 0.13, 0.20, "Bias-free\nattention head\nA(M)")
    _box(ax, 0.70, 0.32, 0.13, 0.16, "Pairwise\ninteraction I(M)")
    _box(ax, 0.87, 0.47, 0.12, 0.16, "$\\hat{Y}$ = b + A + I\nlog$_{10}$ q*", bold=True)
    _box(ax, 0.35, 0.12, 0.30, 0.14, "Mediator: conditional RealNVP per concept\n$\\tilde{M}_j \\sim p(M_j \\mid c_j', X)$  (counterfactual state)", fc="#f1effb", ec="#6250d6", fs=6.8)
    _box(ax, 0.35, 0.84, 0.13, 0.11, "state read-out\n(prototypes) L$_{concept}$", fc="#eaf7f1", ec="#1baf7a", fs=6.5)
    _box(ax, 0.70, 0.84, 0.29, 0.12, "L$_{nec}$: null-state slot swap changes nothing\nL$_{suf}$: active slots reproduce Y\nL$_{ind}$: A additive; non-additivity only in I (L$_{int}$)", fc="#fff7e0", ec="#c98500", fs=6.3)
    _box(ax, 0.72, 0.08, 0.27, 0.14, "Quality branch (separate): RF, CF\n-> variance / interval width only;\nmean prediction detached", fc="#fdf1ec", ec="#c94a1c", fs=6.6)
    for a, b in (((0.14, 0.61), (0.18, 0.61)), ((0.31, 0.61), (0.35, 0.61)), ((0.48, 0.61), (0.52, 0.61)), ((0.65, 0.66), (0.70, 0.66)),
                 ((0.65, 0.50), (0.70, 0.42)), ((0.83, 0.64), (0.87, 0.58)), ((0.83, 0.40), (0.87, 0.50)), ((0.585, 0.84 + 0.085), (0.48, 0.90))):
        _arrow(ax, a, b)
    _arrow(ax, (0.50, 0.26), (0.56, 0.315), color="#6250d6", ls="--")
    _arrow(ax, (0.08, 0.50), (0.40, 0.26), color="#6250d6", ls="--")
    _arrow(ax, (0.93, 0.47), (0.88, 0.22), color="#c94a1c", ls="--")
    _arrow(ax, (0.84, 0.84), (0.80, 0.76), color="#c98500", ls="--")
    save(fig, "fig05_ccw_net_architecture")


# ----------------------------------------------------------------------------------------- Fig 6
def fig06():
    full, un = tab("table06_concept_attribution_ccw_full_geometry"), tab("table06_concept_attribution_ccw_unaligned_geometry")
    pr = tab("table06b_interaction_pairs_ccw_full_geometry")
    a5 = tab("table05_ccw_alignment_ablation_geometry").set_index("variant")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1.3, 1.0, 1.1]})
    ax = axs[0]
    yy = np.arange(len(full))
    ax.barh(yy - 0.2, full.effect_when_active, 0.36, color=CAT[1], label="CCW-Net, concept active")
    ax.barh(yy + 0.2, un.effect_when_active, 0.36, color=CAT[6], label="unaligned, concept active")
    nc = full.null_capable.to_numpy()
    ax.scatter(full.effect_when_null[nc], yy[nc] - 0.2, marker="D", s=16, color=INK, zorder=3, label="concept in null state")
    ax.scatter(un.effect_when_null[nc], yy[nc] + 0.2, marker="D", s=16, color=INK, zorder=3)
    ax.set_yticks(yy, [c.replace("_", " ") for c in full.concept]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |$\\Delta\\hat{Y}$| when slot is zeroed  [log$_{10}$]"); ax.legend(loc="lower right", fontsize=6.3)
    panel(ax, "a", "Concept-level effect (model-internal)")
    ax = axs[1]
    top = pr.head(8)[::-1]
    ax.barh([f"{a.replace('_', ' ')[:14]} x\n{b.replace('_', ' ')[:14]}" for a, b in zip(top.concept_a, top.concept_b)], top.mean_abs_pair_term_log10, color=CAT[2])
    ax.tick_params(axis="y", labelsize=6); ax.grid(axis="y", visible=False); ax.set_xlabel("mean |pair term|  [log$_{10}$]")
    panel(ax, "b", "Largest explicit interactions")
    ax = axs[2]
    order = [v for v in ["ccw_full", "ccw_unaligned", "ccw_no_nec", "ccw_no_suf", "ccw_no_ind", "ccw_no_interaction", "ccw_scalar", "ccw_no_cond_cf", "ccw_gaussian_mediator", "cbm"] if v in a5.index]
    val = 100 * a5.loc[order, "rho_inactive_per_null_slot"]
    sd = 100 * a5.loc[order, "rho_inactive_per_null_slot_sd"].fillna(0)
    ax.barh(range(len(order)), val, xerr=sd, color=[CAT[1] if v == "ccw_full" else MUTED for v in order], error_kw={"lw": 0.8, "ecolor": INK2})
    ax.set_yticks(range(len(order)), [v.replace("ccw_", "").replace("_", " ") for v in order]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("effect share per null-state slot  [%]\n(lower = more faithful; bars $\\pm$ SD over runs)")
    panel(ax, "c", "Effect routed through absent concepts")
    fig.tight_layout()
    save(fig, "fig06_causal_concept_attribution")


# ----------------------------------------------------------------------------------------- Fig 7
def fig07(D):
    from .pipeline import load_pred
    df, vt = D["df"], D["views"]
    sup = np.load(ROOT / "results/diagnostics/support__geometry.npz")["support"]
    cols = np.flatnonzero(vt.intervention.to_numpy() == "crest_up")
    x = np.concatenate([[0], vt.delta.to_numpy()[cols]])
    models = [("xgb", "XGBoost"), ("xgb_mono", "Monotonic XGBoost"), ("mlp", "MLP"), ("ccw_full", "CCW-Net (full)")]
    P = {m: load_pred(f"geometry__{m}")[0] for m, _ in models if (ROOT / f"results/predictions/geometry__{m}.npz").exists()}
    ok = sup[:, cols].all(1) & sup[:, 0]
    Yx = np.column_stack([P["xgb"][:, 0], P["xgb"][:, cols]])
    rise = np.diff(Yx, axis=1).max(1)
    rng = np.random.RandomState(3)
    fams = ["smooth_slope", "rough_slope", "vertical_wall"]
    picks = []
    for f in fams:  # one clearly violating and one consistent record per family (supported curves only)
        c = np.flatnonzero(ok & (df.family == f).to_numpy())
        bad = c[rise[c] > 0.05]
        good = c[rise[c] <= 0]
        picks += [(rng.choice(bad) if len(bad) else c[np.argmax(rise[c])], "violation (XGBoost)"), (rng.choice(good), "consistent")]
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for ax, (i, kind), letter in zip(axs.T.ravel(), picks, "abcdef"):
        for m, lab in models:
            if m in P:
                yv = np.concatenate([[P[m][i, 0]], P[m][i, cols]])
                ax.plot(x, yv, "-o", ms=3, lw=1.6, color=MODEL_COLOR[lab], label=lab)
        ax.axhline(df.y.iloc[i], color=INK, lw=0.8, ls=":")
        ax.text(1.0, df.y.iloc[i], "measured", fontsize=6.5, ha="right", va="bottom", color=INK2)
        panel(ax, letter, f"{df.family.iloc[i].replace('_', ' ')}: {kind}")
        ax.text(0.03, 0.04, f"test {df.name.iloc[i]}, RF={int(df.RF.iloc[i])}, CF={int(df.CF.iloc[i])}, $R_c/H_{{m0}}$={df.rc_h.iloc[i]:.2f}", transform=ax.transAxes, fontsize=6.5, color=INK2)
    for ax in axs[1]:
        ax.set_xlabel("crest raised by $\\delta$  [$H_{m0}$]")
    for ax in axs[:, 0]:
        ax.set_ylabel("predicted log$_{10}$ q*")
    axs[0, 0].legend(loc="upper right", fontsize=6.3)
    fig.tight_layout()
    save(fig, "fig07_counterfactual_response_curves")


# ----------------------------------------------------------------------------------------- Fig 8
def fig08():
    t7, s = tab("table07_counterfactual_inconsistency"), tab("table07b_reversals_by_rf_cf_family")
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.5, 1, 1, 1.2]})
    ax = axs[0]
    c = t7[t7.intervention == "crest_up"].reset_index(drop=True)
    yy = np.arange(len(c))
    ax.barh(yy - 0.2, c.local_reversal_pct, 0.36, color=CAT[0], label="local ($\\delta$=0.05 $H_{m0}$)")
    ax.barh(yy + 0.2, c.global_reversal_pct, 0.36, color=CAT[1], label="global ($\\delta$=1 $H_{m0}$)")
    ax.errorbar(c.local_reversal_pct, yy - 0.2, xerr=[c.local_reversal_pct - c.local_lo, c.local_hi - c.local_reversal_pct], fmt="none", ecolor=INK2, lw=0.8)
    ax.errorbar(c.global_reversal_pct, yy + 0.2, xerr=[c.global_reversal_pct - c.global_lo, c.global_hi - c.global_reversal_pct], fmt="none", ecolor=INK2, lw=0.8)
    ax.set_yticks(yy, [m.replace(" (", "\n(") for m in c.model], fontsize=6.5); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("reversal rate  [% of valid curves]"); ax.legend(loc="lower right", fontsize=6.3); panel(ax, "a", "Crest raised: reversals")
    show = [m for m in ("XGBoost", "MLP", "CCW-Net (full)") if m in set(s.model)]
    for ax, k, letter in zip(axs[1:3], ("RF", "CF"), "bc"):
        for i, m in enumerate(show):
            q = s[(s.model == m) & (s.stratum == k)]
            xs = q["value"].astype(float).astype(int).to_numpy() + (i - 1) * 0.15
            err(ax, xs, q.pct_curves_affected, q.aff_lo, q.aff_hi, MODEL_COLOR[m], label=m)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel(k); ax.set_ylim(0, None); ax.set_ylabel("curves with $\\geq$1 upward step  [%]")
        panel(ax, letter, f"by {k}")
    axs[1].legend(loc="upper left", fontsize=6)
    ax = axs[3]
    q = s[(s.model == "XGBoost") & (s.stratum == "RFxCF")]
    M, N = np.full((4, 4), np.nan), np.zeros((4, 4))
    for _, rw in q.iterrows():
        a, b = int(rw["value"][2]) - 1, int(rw["value"][-1]) - 1
        M[a, b], N[a, b] = rw.pct_curves_affected, rw.n_curves_valid
    im = ax.imshow(M, cmap=SEQ_CMAP, aspect="auto", vmin=0)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, "n<30" if np.isnan(M[a, b]) else f"{M[a, b]:.0f} %\nn={int(N[a, b])}", ha="center", va="center", fontsize=6.3,
                    color=MUTED if np.isnan(M[a, b]) else ("white" if M[a, b] > np.nanmax(M) * 0.6 else INK))
    ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_yticks(range(4), [1, 2, 3, 4]); ax.set_xlabel("CF"); ax.set_ylabel("RF"); ax.grid(False)
    panel(ax, "d", "XGBoost, affected curves")
    fig.tight_layout()
    save(fig, "fig08_reversals_by_rf_cf")


# ----------------------------------------------------------------------------------------- Fig 9
def fig09():
    vm, tr, cfg_, j = tab("table08c2_variance_model_main_effects"), tab("table08b_ordinal_trend_tests"), tab("table08d_cf_geometry_level"), tab("table10_rf_cf_joint")
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.1, 1.3, 1, 1.2]})
    ax = axs[0]
    v = vm[vm.term != "const"].reset_index(drop=True)
    cols = [SEQ[int(t[-1]) - 1] for t in v.term]
    for i, rw in v.iterrows():
        err(ax, [i], [rw.variance_ratio], [rw.ratio_lo], [rw.ratio_hi], cols[i])
    ax.axhline(1, color=AXIS, lw=0.8); ax.set_yscale("log"); ax.set_xticks(range(len(v)), v.term, rotation=60, ha="right")
    ax.set_ylabel("residual-variance ratio vs category 1"); panel(ax, "a", "Variance model")
    ax = axs[1]
    tr = tr.reset_index(drop=True)
    labs = sorted(set(tr.residuals), key=list(tr.residuals).index)
    for k, col, off in (("RF", CAT[0], -0.15), ("CF", CAT[1], 0.15)):
        q = tr[tr.factor == k].set_index("residuals").loc[labs]
        ax.errorbar(q.spearman_absres, np.arange(len(labs)) + off, xerr=[q.spearman_absres - q.spearman_lo, q.spearman_hi - q.spearman_absres],
                    fmt="o", color=col, ms=4, lw=1.1, capsize=2, label=k)
    ax.axvline(0, color=AXIS, lw=0.8); ax.set_yticks(range(len(labs)), [l.replace(", geometry-grouped", "").replace("_", " ") for l in labs], fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlabel("Spearman $\\rho$(|residual|, category)"); ax.legend(loc="lower right"); panel(ax, "b", "Ordinal trend, by model")
    ax = axs[2]
    for i, rw in cfg_.iterrows():
        err(ax, [i], [rw.abs_mean_resid], [rw.abs_mean_resid_lo], [rw.abs_mean_resid_hi], SEQ[i])
    ax.set_xticks(range(4), [f"{int(c)}\n{int(n)}" for c, n in zip(cfg_.CF, cfg_.n_geometries)]); ax.set_xlabel("CF (max in geometry)\nnumber of geometries")
    ax.set_ylabel("|geometry mean residual|  [log$_{10}$]"); ax.set_ylim(0, None); panel(ax, "c", "Geometry-level error")
    ax = axs[3]
    M = j.pivot(index="RF", columns="CF", values="mae").to_numpy(float)
    N = j.pivot(index="RF", columns="CF", values="n").to_numpy()
    okc = j.pivot(index="RF", columns="CF", values="support").to_numpy() == "adequate"
    Mm = np.where(okc, M, np.nan)
    ax.imshow(Mm, cmap=SEQ_CMAP, aspect="auto", vmin=0.2)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f"{M[a, b]:.2f}\nn={N[a, b]}" if okc[a, b] else f"n={N[a, b]}\n(sparse)", ha="center", va="center", fontsize=6.3,
                    color=MUTED if not okc[a, b] else ("white" if M[a, b] > 0.55 else INK))
    ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_yticks(range(4), [1, 2, 3, 4]); ax.set_xlabel("CF"); ax.set_ylabel("RF"); ax.grid(False)
    panel(ax, "d", "MAE by RF x CF  [log$_{10}$]")
    fig.tight_layout()
    save(fig, "fig09_rf_cf_validity")


# ----------------------------------------------------------------------------------------- Fig 11
def fig11():
    t, w = tab("table11_quality_aware_uncertainty"), tab("table11c_interval_width_by_rf_cf")
    show = ["split_abs", "norm_physical", "norm_quality_only", "norm_physical+quality", "mondrian_RF", "cqr_physical", "cqr_physical+quality"]
    col = dict(zip(show, [MUTED, CAT[0], CAT[3], CAT[1], CAT[2], CAT[6], CAT[4]]))
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})
    ax = axs[0]
    q = t.set_index("method").loc[show]
    for i, (m, rw) in enumerate(q.iterrows()):
        err(ax, [rw.mean_width], [100 * rw.coverage], [100 * rw.cov_lo], [100 * rw.cov_hi], col[m])
        ax.annotate(m.replace("_", " "), (rw.mean_width, 100 * rw.coverage), xytext=(4, 4 if i % 2 else -9), textcoords="offset points", fontsize=6, color=INK2)
    ax.axhline(90, color=INK, lw=0.8, ls="--"); ax.set_xlabel("mean interval width  [log$_{10}$]"); ax.set_ylabel("coverage on unseen geometries  [%]")
    panel(ax, "a", "Coverage vs width (nominal 90 %)")
    for ax, k, letter in zip(axs[1:3], ("RF", "CF"), "bc"):
        for m in ("split_abs", "norm_physical", "norm_physical+quality", "mondrian_RF" if k == "RF" else "mondrian_CF"):
            ax.plot([1, 2, 3, 4], [100 * t.set_index("method").loc[m, f"cov_{k}{v}"] for v in (1, 2, 3, 4)], "-o", ms=4, lw=1.6,
                    color=col.get(m, CAT[2]), label=m.replace("_", " "))
        ax.axhline(90, color=INK, lw=0.8, ls="--"); ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel(k); ax.set_ylabel("coverage  [%]")
        ax.legend(loc="lower left", fontsize=6); panel(ax, letter, f"Coverage by {k}")
    ax = axs[3]
    for m in ("split_abs", "norm_physical", "norm_physical+quality"):
        q = w[(w.method == m) & (w.factor == "RF")]
        ax.plot(q.category, q.mean_width, "-o", ms=4, lw=1.6, color=col[m], label=m.replace("_", " "))
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("RF"); ax.set_ylabel("mean interval width  [log$_{10}$]"); ax.legend(loc="upper left", fontsize=6)
    panel(ax, "d", "Width by RF")
    fig.tight_layout()
    save(fig, "fig11_quality_aware_uncertainty")


# ----------------------------------------------------------------------------------------- Fig 12
def fig12():
    t = tab("table12_category4_filtering_sensitivity")
    t = t[t.learner == "xgb"]
    order = ["full", "excl_rf4", "excl_cf4", "excl_rf4_or_cf4", "clash_weight_(4-RF)(4-CF)", "soft_weight_(5-RF)(5-CF)"]
    labs = ["full (primary)", "excl. RF=4", "excl. CF=4", "excl. RF=4 or CF=4", "CLASH weights\n(4-RF)(4-CF)", "soft weights\n(5-RF)(5-CF)"]
    subs = ["RF<4 and CF<4", "RF=4", "CF=4"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), sharey=True)
    for ax, sname, letter in zip(axs, subs, "abc"):
        q = t[t.test_subset == sname].set_index("training_set").loc[order]
        for i, (_, rw) in enumerate(q.iterrows()):
            ax.errorbar([rw.rmse_diff_vs_full], [i], xerr=[[rw.rmse_diff_vs_full - rw.diff_lo], [rw.diff_hi - rw.rmse_diff_vs_full]], fmt="o",
                        color=CAT[1] if i == 0 else CAT[0], ms=4.5, lw=1.2, capsize=2.5)
        ax.axvline(0, color=INK, lw=0.8, ls="--"); ax.set_yticks(range(len(order)), labs); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlabel("$\\Delta$RMSE vs full training set  [log$_{10}$]")
        panel(ax, letter, f"tested on {sname}  (n={int(q.n_test.iloc[0])})")
    fig.tight_layout()
    save(fig, "fig12_filtering_sensitivity")


def make_all():
    from .pipeline import load_model_data
    D = load_model_data()
    done, failed = [], []
    for name, fn in (("fig01", fig01), ("fig02", fig02), ("fig03", lambda: fig03(D)), ("fig04_10", fig04_10), ("fig05", fig05), ("fig06", fig06),
                     ("fig07", lambda: fig07(D)), ("fig08", fig08), ("fig09", fig09), ("fig11", fig11), ("fig12", fig12)):
        try:
            fn()
            done.append(name)
        except Exception as e:  # noqa: BLE001  (a figure whose inputs are missing is reported, not hidden)
            failed.append((name, repr(e)))
            plt.close("all")
    with open(out("diagnostics", "figure_status.json"), "w") as fh:
        json.dump({"done": done, "failed": failed}, fh, indent=1)
    return done, failed


if __name__ == "__main__":
    print(make_all())
