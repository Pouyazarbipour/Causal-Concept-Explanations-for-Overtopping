"""Publication figures (vector PDF + 600 dpi PNG).  All figures read saved tables/predictions only.

Typeface: Times New Roman for every text element.  If Times New Roman is not installed, matplotlib falls
back to Liberation Serif, which is metrically identical (same widths and line breaks); maths uses STIX.
Colour roles: ordinal RF/CF categories -> one-hue sequential ramp (light = 1, dark = 4);
models/policies -> fixed categorical order.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .pipeline import have, load_pred
from .utils import ROOT, out

SEQ = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]            # categories 1..4
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq", ["#f3f7fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DENS = LinearSegmentedColormap.from_list("dens", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
MODELS = [("rf", "Random Forest"), ("xgb", "XGBoost"), ("catboost", "CatBoost"), ("mlp", "MLP"), ("ccw_full", "CCW-Net")]
MODEL_COLOR = dict(zip([m for _, m in MODELS], [CAT[5], CAT[0], CAT[2], CAT[3], CAT[1]]))
POL = [("P1_retain_all", "P1 retain all"), ("P2_excl_RF4", "P2 excl. RF=4"), ("P3_excl_CF4", "P3 excl. CF=4"),
       ("P4_excl_RF4_or_CF4", "P4 excl. RF=4 or CF=4"), ("P5_CLASH_weights", "P5 weights (4-RF)(4-CF)"),
       ("P5b_CLASH_weights_and_remarks", "P5b P5 + remark removal"), ("P6a_inverse_variance_weights", "P6a empirical weights"),
       ("P6b_quality_aware_intervals", "P6b RF/CF-aware intervals"), ("P6c_weights_and_intervals", "P6c P6a + P6b")]
T = ROOT / "results" / "tables"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9.5, "axes.edgecolor": AXIS,
                     "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2,
                     "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.labelcolor": INK, "text.color": INK, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
                     "figure.facecolor": "white", "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42})
YL = r"$\log_{10} q^*$"


def tab(name):
    return pd.read_csv(T / f"{name}.csv")


def save(fig, name):
    fig.savefig(out("figures", name + ".pdf"))
    fig.savefig(out("figures", name + ".png"), dpi=600)
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", fontweight="bold")


def err(ax, x, y, lo, hi, color, **kw):
    ax.errorbar(x, y, yerr=[np.asarray(y) - np.asarray(lo), np.asarray(hi) - np.asarray(y)], fmt="o", color=color, ms=4.5, lw=1.2, capsize=2.5, **kw)


def _box(ax, x, y, w, h, text, fc="#f3f7fd", ec="#1c5cab", fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.02", fc=fc, ec=ec, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", color=INK)


def _arrow(ax, a, b, color=INK2, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=color, linestyle=ls, shrinkA=1, shrinkB=1))


def stats_txt(y, p):
    e = p - y
    return (f"RMSE = {np.sqrt(np.mean(e ** 2)):.3f}\nMAE = {np.mean(np.abs(e)):.3f}\n"
            f"$R^2$ = {1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2):.3f}\n$n$ = {len(y)}")


def scatter(ax, y, p, lim=(-7.6, -0.6), small=False):
    """Measured (x) versus out-of-fold predicted (y) with a 1:1 line; density for large samples."""
    if len(y) > 1500:
        hb = ax.hexbin(y, p, gridsize=55, extent=(*lim, *lim), cmap=DENS, norm=LogNorm(vmin=1, vmax=120), mincnt=1, linewidths=0)
    else:
        hb = None
        ax.scatter(y, p, s=7, color=CAT[0], alpha=0.35, linewidths=0)
    ax.plot(lim, lim, color=INK, lw=0.9)
    for k in (-1, 1):
        ax.plot(lim, [lim[0] + k, lim[1] + k], color=MUTED, lw=0.7, ls="--")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.text(0.04, 0.97, stats_txt(y, p), transform=ax.transAxes, va="top", fontsize=7.5 if small else 8)
    return hb


# ----------------------------------------------------------------------------------------- Fig 1
def fig01():
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    O = dict(fc="#fdf1ec", ec="#c94a1c")
    _box(ax, 0.02, 0.80, 0.27, 0.16, "Original CLASH database\n10 532 tests, RF and CF 1–4\n(no record removed)", bold=True)
    _box(ax, 0.365, 0.80, 0.27, 0.16, "Audit: response classes, flags,\ngeometry reconstruction,\nRF × CF consistency")
    _box(ax, 0.71, 0.80, 0.27, 0.16, "RF (test level) and CF (structure level)\nkept as quality metadata,\nnever inputs of the mean model", **O)
    _box(ax, 0.02, 0.50, 0.27, 0.18, "Diagnostic models (Optuna, nested,\ngeometry-grouped): RF, XGBoost,\nCatBoost, MLP, CCW-Net")
    _box(ax, 0.365, 0.50, 0.27, 0.18, "Leakage-free out-of-fold\npredictions and residuals\nfor every record")
    _box(ax, 0.71, 0.50, 0.27, 0.18, "Regression CCW-Net: concept\nnecessity, sufficiency, independence;\nexplanation quality vs RF, CF")
    _box(ax, 0.02, 0.17, 0.27, 0.20, "RF and CF validity\nordinal trends, variance model,\nrepeatability, geometry-level error", **O)
    _box(ax, 0.365, 0.17, 0.27, 0.20, "Error decomposition\nseries, geometry,\nwithin-geometry residual", **O)
    _box(ax, 0.71, 0.17, 0.27, 0.20, "Counterfactual data-selection policies\nretain, exclude, re-weight;\naccuracy, calibration, stability", **O)
    for a, b in (((0.29, 0.88), (0.365, 0.88)), ((0.155, 0.80), (0.155, 0.68)), ((0.29, 0.59), (0.365, 0.59)), ((0.64, 0.59), (0.71, 0.59)),
                 ((0.50, 0.50), (0.155, 0.37)), ((0.50, 0.50), (0.50, 0.37)), ((0.50, 0.50), (0.845, 0.37))):
        _arrow(ax, a, b)
    _arrow(ax, (0.965, 0.80), (0.965, 0.375), color="#c94a1c", ls="--")
    ax.text(0.5, 0.06, "Blue: prediction and concept learning used as diagnostic instruments.   Orange: the data-quality investigation "
            "(RF/CF metadata evaluated against out-of-fold errors).", ha="center", fontsize=8.5, color=INK2)
    save(fig, "fig01_research_framework")


# ----------------------------------------------------------------------------------------- Fig 2
def fig02():
    d = tab("table02_rf_cf_distribution")
    j = pd.read_csv(T / "table02b_rf_cf_joint_all.csv", index_col=0)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, k, letter in zip(axs[:2], ("RF", "CF"), "ab"):
        s = d[(d.factor == k) & d.category.astype(str).isin(list("1234"))]
        ax.bar(s.category.astype(str), s.n_all, color=SEQ, width=0.7)
        for x, (n, p) in enumerate(zip(s.n_all, s.pct_all)):
            ax.text(x, n, f"{int(n)}\n{p:.1f} %", ha="center", va="bottom", fontsize=8.5)
        ax.set_ylim(0, s.n_all.max() * 1.25); ax.set_xlabel(f"{k} category"); ax.set_ylabel("Number of tests")
        panel(ax, letter, "Reliability factor RF" if k == "RF" else "Complexity factor CF")
        ax.grid(axis="x", visible=False)
    ax = axs[2]
    M = j.iloc[:4, :4].to_numpy(float)
    ax.imshow(np.log10(M), cmap=SEQ_CMAP, aspect="auto")
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f"{int(M[a, b])}\n{100 * M[a, b] / M.sum():.1f} %", ha="center", va="center", fontsize=8.5, color="white" if M[a, b] > 600 else INK)
    ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_yticks(range(4), [1, 2, 3, 4]); ax.set_xlabel("CF"); ax.set_ylabel("RF"); ax.grid(False)
    panel(ax, "c", "Joint RF × CF frequency (all 10 532 tests)")
    fig.tight_layout()
    save(fig, "fig02_rf_cf_distribution")


# ----------------------------------------------------------------------------------------- Figs 3-7: scatter plots
def _preds(D):
    return {lab: load_pred(f"geometry__{m}").mean(0) for m, lab in MODELS if have(f"geometry__{m}")}


def fig03(D):
    y, P = D["y"], _preds(D)
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 7.2))
    hb = None
    for ax, (lab, p), letter in zip(axs.ravel(), P.items(), "abcde"):
        hb = scatter(ax, y, p)
        panel(ax, letter, lab)
        ax.set_xlabel(f"Measured {YL}"); ax.set_ylabel(f"Predicted {YL} (out-of-fold)")
    ax = axs.ravel()[-1]
    ax.axis("off")
    cb = fig.colorbar(hb, ax=ax, fraction=0.5, orientation="horizontal", pad=0.0)
    cb.set_label("Number of tests per hexagon")
    ax.text(0.5, 0.75, "Held-out geometries only (5-fold geometry-grouped\nnested cross-validation; hyper-parameters by Optuna\n"
            "on training folds). Solid line: 1:1. Dashed lines:\nfactor 10. All 9 333 positive-discharge tests,\nRF and CF 1–4. MLP and CCW-Net: mean of\nthree seeds (seed ensemble); Table 3 reports\nthe mean of per-seed metrics.",
            ha="center", va="center", transform=ax.transAxes, fontsize=9)
    fig.tight_layout()
    save(fig, "fig03_measured_vs_predicted_all_models")


def _strat_scatter(D, key, name):
    y, P, df = D["y"], _preds(D), D["df"]
    fig, axs = plt.subplots(len(P), 4, figsize=(10.5, 2.6 * len(P)), sharex=True, sharey=True)
    for r, (lab, p) in enumerate(P.items()):
        for c, v in enumerate((1, 2, 3, 4)):
            mk = (df[key] == v).to_numpy()
            scatter(axs[r, c], y[mk], p[mk], small=True)
            axs[r, c].set_title(f"{lab}, {key} = {v}", loc="left", fontweight="bold", fontsize=9)
            if c == 0:
                axs[r, c].set_ylabel(f"Predicted {YL}")
            if r == len(P) - 1:
                axs[r, c].set_xlabel(f"Measured {YL}")
    fig.tight_layout()
    save(fig, name)


def fig04(D):
    _strat_scatter(D, "RF", "fig04_measured_vs_predicted_by_RF")


def fig05(D):
    _strat_scatter(D, "CF", "fig05_measured_vs_predicted_by_CF")


def fig06(D):
    y, P, df = D["y"], _preds(D), D["df"]
    fig, axs = plt.subplots(2, len(P), figsize=(10.5, 4.6), sharex=True, sharey=True)
    for r, key in enumerate(("RF", "CF")):
        mk = (df[key] == 4).to_numpy()
        for c, (lab, p) in enumerate(P.items()):
            scatter(axs[r, c], y[mk], p[mk], small=True)
            axs[r, c].set_title(f"{lab}, {key} = 4", loc="left", fontweight="bold", fontsize=9)
            if c == 0:
                axs[r, c].set_ylabel(f"Predicted {YL}")
            if r == 1:
                axs[r, c].set_xlabel(f"Measured {YL}")
    fig.tight_layout()
    save(fig, "fig06_category4_measured_vs_predicted")


def fig07(D):
    y, P = D["y"], _preds(D)
    fig, axs = plt.subplots(1, len(P), figsize=(10.5, 2.9), sharey=True)
    for ax, (lab, p), letter in zip(axs, P.items(), "abcde"):
        ax.hexbin(p, y - p, gridsize=45, extent=(-7.2, -1, -3, 3), cmap=DENS, norm=LogNorm(vmin=1, vmax=120), mincnt=1, linewidths=0)
        ax.axhline(0, color=INK, lw=0.9)
        b = np.linspace(-6.5, -1.5, 11)
        mid, med = 0.5 * (b[1:] + b[:-1]), [np.median((y - p)[(p >= lo) & (p < hi)]) if ((p >= lo) & (p < hi)).sum() > 20 else np.nan for lo, hi in zip(b[:-1], b[1:])]
        ax.plot(mid, med, "-o", color=CAT[1], ms=3, lw=1.3, label="binned median")
        ax.set_xlim(-7.2, -1); ax.set_ylim(-3, 3); ax.set_xlabel(f"Predicted {YL}")
        panel(ax, letter, lab)
    axs[0].set_ylabel("Residual (measured − predicted)")
    axs[0].legend(loc="lower left")
    fig.tight_layout()
    save(fig, "fig07_residual_vs_predicted")


# ----------------------------------------------------------------------------------------- Fig 8
def fig08(D):
    df, y = D["df"], D["y"]
    r = y - load_pred("geometry__xgb")[0]
    s = tab("table05a_residuals_by_rf_cf")
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.6))
    for row, k in enumerate(("RF", "CF")):
        ax = axs[row, 0]
        parts = ax.violinplot([r[df[k] == v] for v in (1, 2, 3, 4)], positions=range(4), showextrema=False, widths=0.8)
        for b, c in zip(parts["bodies"], SEQ):
            b.set_facecolor(c); b.set_alpha(0.85); b.set_edgecolor("white")
        q = [np.percentile(r[df[k] == v], [25, 50, 75]) for v in (1, 2, 3, 4)]
        ax.vlines(range(4), [a[0] for a in q], [a[2] for a in q], color=INK, lw=1.2); ax.scatter(range(4), [a[1] for a in q], color="white", edgecolor=INK, s=14, zorder=3)
        ax.axhline(0, color=AXIS, lw=0.8); ax.set_ylim(-2.5, 2.5)
        ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_xlabel(k); ax.set_ylabel("Out-of-fold residual")
        panel(ax, "ad"[row], f"Residual distribution by {k}")
        t = s[s.factor == k]
        for ax, col, lo, hi, lab, letter in ((axs[row, 1], "mae", "mae_lo", "mae_hi", "Mean absolute residual", "be"[row]),
                                             (axs[row, 2], "within_geometry_sd", "within_sd_lo", "within_sd_hi", "Within-geometry residual SD", "cf"[row])):
            for i in range(4):
                err(ax, [i], [t[col].iloc[i]], [t[lo].iloc[i]], [t[hi].iloc[i]], SEQ[i])
                ax.text(i + 0.12, t[col].iloc[i], f"{t[col].iloc[i]:.2f}", va="center", fontsize=8.5, color=INK2)
            ax.set_xticks(range(4), [f"{v}\nn = {n}" for v, n in zip((1, 2, 3, 4), t.n)]); ax.set_xlim(-0.5, 3.8)
            ax.set_xlabel(k); ax.set_ylabel(lab); ax.set_ylim(0, None)
            panel(ax, letter, "Absolute error" if col == "mae" else "Dispersion within geometry")
    fig.tight_layout()
    save(fig, "fig08_residuals_by_rf_cf")


# ----------------------------------------------------------------------------------------- Fig 9
def fig09():
    vm, tr, cfg_, j = tab("table05d_variance_model_main_effects"), tab("table05b_ordinal_trend_tests"), tab("table06a_cf_geometry_level"), tab("table07_rf_cf_joint")
    rp = tab("table05h_repeatability_near_replicates")
    fig, axs = plt.subplots(1, 5, figsize=(13.5, 3.3), gridspec_kw={"width_ratios": [1.05, 1.25, 0.9, 0.9, 1.5]})
    ax = axs[0]
    v = vm[vm.term != "const"].reset_index(drop=True)
    for i, rw in v.iterrows():
        err(ax, [i], [rw.variance_ratio], [rw.ratio_lo], [rw.ratio_hi], SEQ[int(rw.term[-1]) - 1])
    ax.axhline(1, color=AXIS, lw=0.8); ax.set_yscale("log"); ax.set_xticks(range(len(v)), v.term, rotation=60, ha="right")
    ax.set_ylabel("Residual-variance ratio vs category 1"); panel(ax, "a", "Variance model")
    ax = axs[1]
    labs = list(dict.fromkeys(tr.residuals))
    for k, col, off in (("RF", CAT[0], -0.15), ("CF", CAT[1], 0.15)):
        q = tr[tr.factor == k].set_index("residuals").loc[labs]
        ax.errorbar(q.spearman_absres, np.arange(len(labs)) + off, xerr=[q.spearman_absres - q.spearman_lo, q.spearman_hi - q.spearman_absres],
                    fmt="o", color=col, ms=4, lw=1.1, capsize=2, label=k)
    ax.axvline(0, color=AXIS, lw=0.8); ax.set_yticks(range(len(labs)), [l.replace(", geometry-grouped", "").replace("_full", "").replace("ccw", "CCW-Net") for l in labs], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel(r"Spearman $\rho$(|residual|, category)"); ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); panel(ax, "b", "Ordinal trend")
    ax = axs[2]
    s = rp[rp.subset.str.startswith("max RF") & rp.repeat_sd_log10.notna()]
    for _, rw in s.iterrows():
        x = int(rw.subset[-1]) - 1
        err(ax, [x], [rw.repeat_sd_log10], [rw.sd_lo], [rw.sd_hi], SEQ[x])
        ax.text(x, rw.sd_hi, f"{int(rw.n_pairs)}", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_xlim(-0.5, 3.5); ax.set_xlabel("Max RF of the pair\n(labels: number of pairs)"); ax.set_ylabel(f"Repeat SD of {YL}"); ax.set_ylim(0, None)
    panel(ax, "c", "Near-replicates")
    ax = axs[3]
    for i, rw in cfg_.iterrows():
        err(ax, [i], [rw.abs_mean_resid], [rw.abs_mean_resid_lo], [rw.abs_mean_resid_hi], SEQ[i])
    ax.set_xticks(range(4), [f"{int(c)}\n{int(n)}" for c, n in zip(cfg_.CF, cfg_.n_geometries)]); ax.set_xlabel("CF (max in geometry)\nnumber of geometries")
    ax.set_ylabel("|Geometry mean residual|"); ax.set_ylim(0, None); panel(ax, "d", "Geometry level")
    ax = axs[4]
    M = j.pivot(index="RF", columns="CF", values="mae").to_numpy(float)
    N = j.pivot(index="RF", columns="CF", values="n").to_numpy()
    okc = j.pivot(index="RF", columns="CF", values="support").to_numpy() == "adequate"
    ax.imshow(np.where(okc, M, np.nan), cmap=SEQ_CMAP, aspect="auto", vmin=0.2)
    for a in range(4):
        for b in range(4):
            ax.text(b, a, f"{M[a, b]:.2f}\nn = {N[a, b]}" if okc[a, b] else f"n = {N[a, b]}\nsparse", ha="center", va="center", fontsize=8,
                    color=MUTED if not okc[a, b] else ("white" if M[a, b] > 0.5 else INK))
    ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_yticks(range(4), [1, 2, 3, 4]); ax.set_xlabel("CF"); ax.set_ylabel("RF"); ax.grid(False)
    panel(ax, "e", "Mean |residual| by RF × CF")
    fig.tight_layout()
    save(fig, "fig09_rf_cf_ordinal_validation")


# ----------------------------------------------------------------------------------------- Fig 10
def fig10():
    t = tab("table08_hierarchical_error_decomposition")
    G = pd.read_csv(ROOT / "results/diagnostics/geometry_level_residuals.csv")
    cc = {"lab": CAT[3], "source": CAT[3], "series": CAT[2], "geometry": CAT[0], "geometry_coarse": CAT[0], "substructure": CAT[0], "within-geometry residual": MUTED}
    nice = {"lab": "source", "source": "source (reference)", "series": "test series", "geometry": "geometry", "geometry_coarse": "geometry",
            "substructure": "geometry", "within-geometry residual": "within-geometry residual"}
    an = list(dict.fromkeys(t.analysis))
    prim = an[0]
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.2))
    ax = axs[0, 0]
    sel = [prim] + [a for a in an if a.startswith("grouping sensitivity")] + [a for a in an if a.startswith("random-CV") or a.startswith("series-held")]
    labs = ["primary", "coarse geometry", "sub-structure", "source/series/geometry", "random CV", "series held out"][:len(sel)]
    seen = set()
    for i, a in enumerate(sel):
        left = 0
        for _, rw in t[t.analysis == a].iterrows():
            lab = nice[rw.component]
            ax.barh(i, rw.share, left=left, color=cc[rw.component], edgecolor="white", lw=1.2, height=0.65, label=lab if lab not in seen else None)
            seen.add(lab)
            if rw.share > 0.07:
                ax.text(left + rw.share / 2, i, f"{100 * rw.share:.0f}", ha="center", va="center", fontsize=8.5, color=INK if cc[rw.component] in (MUTED, CAT[3]) else "white")
            left += rw.share
    ax.set_yticks(range(len(sel)), labs); ax.set_xlim(0, 1); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share of out-of-fold residual variance"); ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.2), ncol=2); panel(ax, "a", "Variance components")
    ax = axs[0, 1]
    p = t[t.analysis == prim]
    for i, (_, rw) in enumerate(p.iterrows()):
        err(ax, [i], [rw.sd], [np.sqrt(rw.variance_lo)], [np.sqrt(rw.variance_hi)], cc[rw.component])
    ax.set_xticks(range(len(p)), ["series", "geometry", "within\ngeometry"]); ax.set_ylabel("Component SD"); ax.set_ylim(0, None); ax.set_xlim(-0.5, 2.5)
    panel(ax, "b", "Component SD (95 % CI)")
    ax = axs[0, 2]
    g = G[G.n >= 3]
    ax.hist(g.mean_resid, bins=np.linspace(-2, 2, 41), color=CAT[0], edgecolor="white", lw=0.4)
    ax.axvline(0, color=AXIS, lw=0.8); ax.set_xlabel("Geometry mean residual"); ax.set_ylabel("Geometries with ≥ 3 tests")
    panel(ax, "c", "Geometry-level heterogeneity")
    for ax, k, letter in zip(axs[1, :2], ("CF", "RF"), "de"):
        for comp, col, off in (("geometry", CAT[0], -0.1), ("within-geometry residual", MUTED, 0.1)):
            s = pd.concat([t[(t.analysis == f"stratum {k}={v}") & (t.component == comp)] for v in (1, 2, 3, 4)])
            err(ax, np.arange(4) + off, s.sd, np.sqrt(s.variance_lo), np.sqrt(s.variance_hi), col, label=nice[comp])
        ax.set_xticks(range(4), [1, 2, 3, 4]); ax.set_xlabel(k); ax.set_ylabel("Component SD"); ax.set_ylim(0, None)
        ax.legend(loc="upper left"); panel(ax, letter, f"Components within {k} strata")
    ax = axs[1, 2]
    b = tab("table08b_repeatability_vs_residual").set_index("quantity")["value"]
    names = ["total, random CV", "total, geometry-grouped", "within geometry (REML)", "near-replicate\nrepeatability"]
    vals = [b.iloc[1], b.iloc[0], b.iloc[2], b.iloc[3]]
    ax.barh(names, np.sqrt(vals), color=[MUTED, MUTED, CAT[0], CAT[1]], height=0.6)
    for i, v in enumerate(vals):
        ax.text(np.sqrt(v) + 0.005, i, f"{np.sqrt(v):.2f}", va="center", fontsize=8.5)
    ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel(f"SD of {YL}"); panel(ax, "f", "Residual SD vs repeatability")
    fig.tight_layout()
    save(fig, "fig10_error_decomposition")


# ----------------------------------------------------------------------------------------- Fig 11
def fig11():
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    _box(ax, 0.01, 0.50, 0.13, 0.22, "15 dimensionless\nphysical inputs\n$X$ (no RF, CF)")
    _box(ax, 0.18, 0.50, 0.13, 0.22, "Tabular MLP\nbackbone $f$\n(pre-trained,\nfrozen)  $Z$")
    _box(ax, 0.35, 0.50, 0.13, 0.22, "Concept\nencoder $g$\n+ squash")
    for i, nm in enumerate(["freeboard", "steepness", "depth", "obliquity", "...", "foreshore"]):
        _box(ax, 0.52, 0.84 - i * 0.105, 0.13, 0.085, f"$M$: {nm}" if nm != "..." else "...", fc="#eaf7f1", ec="#1baf7a", fs=8)
    _box(ax, 0.70, 0.56, 0.13, 0.20, "Bias-free\nattention head\n$A(M)$")
    _box(ax, 0.70, 0.32, 0.13, 0.16, "Pairwise\ninteraction $I(M)$")
    _box(ax, 0.87, 0.47, 0.12, 0.16, r"$\hat{Y} = b + A + I$" + "\n" + YL, bold=True)
    _box(ax, 0.35, 0.12, 0.30, 0.14, "Mediator: conditional RealNVP per concept\n" + r"$\tilde{M}_j \sim p(M_j \mid c_j^{cf}, X)$", fc="#f1effb", ec="#6250d6", fs=8.5)
    _box(ax, 0.35, 0.84, 0.13, 0.11, "state read-out\n(prototypes) $L_{concept}$", fc="#eaf7f1", ec="#1baf7a", fs=8)
    _box(ax, 0.70, 0.84, 0.29, 0.12, "$L_{nec}$: null-state slot swap changes nothing\n$L_{suf}$: active slots reproduce $Y$\n$L_{ind}$: $A$ additive; non-additivity only in $I$ ($L_{int}$)", fc="#fff7e0", ec="#c98500", fs=7.8)
    _box(ax, 0.72, 0.08, 0.27, 0.14, "Quality metadata RF, CF (separate):\nexplanation quality stratified by RF, CF;\nvariance branch with detached mean", fc="#fdf1ec", ec="#c94a1c", fs=8)
    for a, b in (((0.14, 0.61), (0.18, 0.61)), ((0.31, 0.61), (0.35, 0.61)), ((0.48, 0.61), (0.52, 0.61)), ((0.65, 0.66), (0.70, 0.66)),
                 ((0.65, 0.50), (0.70, 0.42)), ((0.83, 0.64), (0.87, 0.58)), ((0.83, 0.40), (0.87, 0.50)), ((0.585, 0.925), (0.48, 0.90))):
        _arrow(ax, a, b)
    _arrow(ax, (0.50, 0.26), (0.56, 0.315), color="#6250d6", ls="--")
    _arrow(ax, (0.08, 0.50), (0.40, 0.26), color="#6250d6", ls="--")
    _arrow(ax, (0.93, 0.47), (0.88, 0.22), color="#c94a1c", ls="--")
    _arrow(ax, (0.84, 0.84), (0.80, 0.76), color="#c98500", ls="--")
    save(fig, "fig11_ccw_net_architecture")


# ----------------------------------------------------------------------------------------- Fig 12
def fig12():
    full, un = tab("table10_concept_attribution_ccw_full"), tab("table10_concept_attribution_ccw_unaligned")
    a5 = tab("table09_ccw_alignment_and_ablation").set_index("variant")
    eq = tab("table12a_explanation_quality_by_rf_cf")
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 6.6))
    ax = axs[0, 0]
    yy = np.arange(len(full))
    ax.barh(yy - 0.2, full.ccw_effect_when_active, 0.36, color=CAT[1], label="CCW-Net, concept active")
    ax.barh(yy + 0.2, un.ccw_effect_when_active, 0.36, color=CAT[6], label="unaligned, concept active")
    nc = full.null_capable.to_numpy()
    ax.scatter(full.ccw_effect_when_null[nc], yy[nc] - 0.2, marker="D", s=14, color=INK, zorder=3, label="concept in null state")
    ax.scatter(un.ccw_effect_when_null[nc], yy[nc] + 0.2, marker="D", s=14, color=INK, zorder=3)
    ax.set_yticks(yy, [c.replace("_", " ") for c in full.concept]); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    ax.set_xlabel(r"Mean $|\Delta\hat{Y}|$ when slot is zeroed"); ax.legend(loc="upper center", bbox_to_anchor=(0.35, -0.16), ncol=1, fontsize=7.5)
    panel(ax, "a", "Slot-ablation effect")
    order = [v for v in ["ccw_full", "ccw_unaligned", "ccw_no_nec", "ccw_no_suf", "ccw_no_ind", "ccw_no_interaction", "ccw_no_cond_cf", "cbm"] if v in a5.index]
    nm = [v.replace("ccw_", "").replace("_", " ").replace("cbm", "concept bottleneck") for v in order]
    for ax, col, lab, letter, ttl in ((axs[0, 1], "rho_inactive_per_null_slot", "Effect share per absent concept  [%]", "b", "Necessity (lower is better)"),
                                      (axs[0, 2], "suf_rmse_vs_output", "Active-only reconstruction error", "c", "Sufficiency (lower is better)")):
        sel = [v for v in order if pd.notna(a5.loc[v, col])]
        k = 100 if col.startswith("rho") else 1
        ax.barh(range(len(sel)), k * a5.loc[sel, col], xerr=k * a5.loc[sel, col + "_sd"].fillna(0), color=[CAT[1] if v == "ccw_full" else MUTED for v in sel],
                error_kw={"lw": 0.8, "ecolor": INK2})
        ax.set_yticks(range(len(sel)), [nm[order.index(v)] for v in sel]); ax.invert_yaxis(); ax.grid(axis="y", visible=False); ax.set_xlabel(lab + "\n(bars ± SD over fold × seed runs)")
        panel(ax, letter, ttl)
    ax = axs[1, 0]
    ax.scatter(full.xgb_mean_abs_shap_log10, full.ccw_slot_ablation_effect_log10, s=28, color=CAT[0], zorder=3)
    off = {"relative_depth": (4, -10), "wave_steepness": (4, 4), "wave_obliquity": (4, -11), "berm": (6, 2), "toe_berm": (4, 4),
           "foreshore": (-4, 6), "crest_configuration": (4, 4), "roughness_permeability": (-30, 6), "profile_type": (4, 3), "relative_freeboard": (-40, 7)}
    for _, rw in full.iterrows():
        ax.annotate(rw.concept.replace("_", " ").replace("permeability", "").strip(), (rw.xgb_mean_abs_shap_log10, rw.ccw_slot_ablation_effect_log10),
                    xytext=off.get(rw.concept, (4, 3)), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xlabel("XGBoost grouped mean |SHAP|"); ax.set_ylabel("CCW-Net slot-ablation effect"); panel(ax, "d", "Concept vs feature attribution")
    for ax, qn, lab, letter in ((axs[1, 1], "necessity_effect", "Effect of absent-concept swap", "e"), (axs[1, 2], "sufficiency_gap", "Sufficiency gap", "f")):
        for fac, col, off in (("RF", CAT[0], -0.08), ("CF", CAT[1], 0.08)):
            q = eq[(eq.model == "CCW-Net") & (eq.quantity == qn) & (eq.factor == fac)]
            err(ax, q.category + off, q["mean"], q.lo, q.hi, col, label=fac)
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("Category"); ax.set_ylabel(lab); ax.set_ylim(0, None); ax.legend(loc="upper left")
        panel(ax, letter, "Explanation quality by RF, CF")
    fig.tight_layout()
    save(fig, "fig12_concept_attribution_and_alignment")


# ----------------------------------------------------------------------------------------- Fig 13
def fig13():
    t = tab("table13_data_selection_policies")
    t = t[t.learner == "xgb"].set_index("policy")
    sel = [p for p, _ in POL if p in t.index and p != "P6b_quality_aware_intervals"]
    labs = [dict(POL)[p] for p in sel]
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.6), sharey=True)
    n = {"common": "RF<4 and CF<4", "RF=4": "RF = 4", "CF=4": "CF = 4"}
    for ax, tag, letter in zip(axs[:3], ("common", "RF=4", "CF=4"), "abc"):
        for i, p in enumerate(sel):
            rw = t.loc[p]
            ax.errorbar([rw[f"d_rmse_{tag}"]], [i], xerr=[[rw[f"d_rmse_{tag}"] - rw[f"d_rmse_{tag}_lo"]], [rw[f"d_rmse_{tag}_hi"] - rw[f"d_rmse_{tag}"]]], fmt="o",
                        color=CAT[1] if i == 0 else CAT[0], ms=4.5, lw=1.2, capsize=2.5)
        ax.axvline(0, color=INK, lw=0.8, ls="--"); ax.set_yticks(range(len(sel)), labs); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
        ax.set_xlabel("ΔRMSE vs retain-all (P1)"); panel(ax, letter, f"Tested on {n[tag]}")
    ax = axs[3]
    ax.barh(np.arange(len(sel)) - 0.2, t.loc[sel, "pct_predictions_changed_by_factor2_common"], 0.38, color=CAT[0], label="RF<4 and CF<4 tests")
    ax.barh(np.arange(len(sel)) + 0.2, t.loc[sel, "pct_predictions_changed_by_factor2_category4"], 0.38, color=CAT[1], label="category-4 tests")
    ax.set_xlabel("Predictions changed by > factor 2  [%]"); ax.grid(axis="y", visible=False); ax.legend(loc="lower right", fontsize=7.5)
    panel(ax, "d", "Stability vs retain-all")
    fig.tight_layout()
    save(fig, "fig13_data_selection_policies")


# ----------------------------------------------------------------------------------------- Fig 14
def fig14():
    c = tab("table14_policy_calibration_by_quality")
    c = c[c.learner == "xgb"].drop_duplicates(["policy", "test_subset"])
    show = ["P1_retain_all", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P6b_quality_aware_intervals"]
    col = dict(zip(show, [MUTED, CAT[0], CAT[2], CAT[1]]))
    fig, axs = plt.subplots(1, 4, figsize=(11.5, 3.4))
    for ax, fac, what, letter in ((axs[0], "RF", "coverage", "a"), (axs[1], "CF", "coverage", "b"), (axs[2], "RF", "mean_width", "c"), (axs[3], "CF", "mean_width", "d")):
        for p in show:
            q = pd.concat([c[(c.policy == p) & (c.test_subset == f"{fac}={v}")] for v in (1, 2, 3, 4)])
            k = 100 if what == "coverage" else 1
            ax.plot([1, 2, 3, 4], k * q[what], "-o", ms=4, lw=1.6, color=col[p], label=dict(POL)[p])
        if what == "coverage":
            ax.axhline(90, color=INK, lw=0.8, ls="--")
        ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel(fac); ax.set_ylabel("Coverage of 90 % intervals  [%]" if what == "coverage" else "Mean interval width")
        panel(ax, letter, ("Coverage" if what == "coverage" else "Width") + f" by {fac}")
    h, l = axs[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(h, l, loc="lower center", ncol=4)
    save(fig, "fig14_calibration_by_quality_category")


# ----------------------------------------------------------------------------------------- Fig 15
def fig15(D):
    y, df = D["y"], D["df"]
    Z = np.load(ROOT / "results/predictions/policy__xgb.npz")
    fig, axs = plt.subplots(2, 3, figsize=(9.6, 6.6), sharex=True, sharey=True)
    cases = [("RF", "P1_retain_all", "retain all (P1)"), ("RF", "P2_excl_RF4", "RF = 4 excluded (P2)"), ("RF", "P5_CLASH_weights", "weights (4-RF)(4-CF) (P5)"),
             ("CF", "P1_retain_all", "retain all (P1)"), ("CF", "P3_excl_CF4", "CF = 4 excluded (P3)"), ("CF", "P5_CLASH_weights", "weights (4-RF)(4-CF) (P5)")]
    for ax, (fac, pol, lab), letter in zip(axs.ravel(), cases, "abcdef"):
        mk = (df[fac] == 4).to_numpy()
        scatter(ax, y[mk], Z[f"{pol}__pred"][mk], small=True)
        panel(ax, letter, f"{fac} = 4 tests: {lab}")
    for ax in axs[1]:
        ax.set_xlabel(f"Measured {YL}")
    for ax in axs[:, 0]:
        ax.set_ylabel(f"Predicted {YL} (out-of-fold)")
    fig.tight_layout()
    save(fig, "fig15_full_vs_filtered_category4")


# ----------------------------------------------------------------------------------------- Fig S1
def figS1():
    d = ROOT / "results" / "models" / "optuna"
    order = ["rf", "xgb", "catboost", "mlp", "ccw"]
    files = sorted(d.glob("geometry__*__trials.csv"), key=lambda f: order.index(f.stem.split("__")[1]))
    fig, axs = plt.subplots(1, len(files), figsize=(2.4 * len(files), 2.9))
    for ax, f, letter in zip(np.atleast_1d(axs), files, "abcdefg"):
        t = pd.read_csv(f)
        for k, g in t.groupby("fold"):
            v = g.sort_values("number")["value"].to_numpy()
            ax.plot(np.arange(1, len(v) + 1), v, "o", ms=2.5, color=MUTED, alpha=0.6)
            ax.plot(np.arange(1, len(v) + 1), np.minimum.accumulate(v), "-", lw=1.4, color=CAT[int(k) % 8])
        ax.set_xlabel("Optuna trial"); panel(ax, letter, {"rf": "Random Forest", "xgb": "XGBoost", "catboost": "CatBoost", "mlp": "MLP", "ccw": "CCW-Net"}[f.stem.split("__")[1]])
    np.atleast_1d(axs)[0].set_ylabel("Inner grouped-CV RMSE\n(line: best so far, per outer fold)")
    fig.tight_layout()
    save(fig, "figS1_optuna_history")


def make_all():
    from .pipeline import load_model_data
    D = load_model_data()
    done, failed = [], []
    for name, fn in (("fig01", fig01), ("fig02", fig02), ("fig03", lambda: fig03(D)), ("fig04", lambda: fig04(D)), ("fig05", lambda: fig05(D)),
                     ("fig06", lambda: fig06(D)), ("fig07", lambda: fig07(D)), ("fig08", lambda: fig08(D)), ("fig09", fig09), ("fig10", fig10),
                     ("fig11", fig11), ("fig12", fig12), ("fig13", fig13), ("fig14", fig14), ("fig15", lambda: fig15(D)), ("figS1", figS1)):
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
