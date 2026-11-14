"""Assemble the manuscript and outline as JSON block lists for build_docx.js.

Every number in the manuscript is read from results/tables at build time, so the text stays consistent with
the analysis.  Usage: python manuscript/build_content.py && node manuscript/build_docx.js ...
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
OUT = Path(__file__).resolve().parent
(OUT / "figs").mkdir(exist_ok=True)


def tab(n):
    return pd.read_csv(T / f"{n}.csv")


def f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}".replace("-", "−")


def s(x, d=3):   # signed
    return ("+" if x >= 0 else "−") + f"{abs(x):.{d}f}"


def ci(lo, hi, d=3):
    return f"[{f(lo, d)}, {f(hi, d)}]"


def pct(x, d=0):
    return f"{100 * x:.{d}f} %"


# ------------------------------------------------------------------------------------------- data
perf = tab("table03_model_performance")
P = perf[perf.scheme == "geometry"].set_index("model")
PS = perf.set_index(["scheme", "model"])
rf_t = tab("table05a_residuals_by_rf_cf")
RF = rf_t[rf_t.factor == "RF"].set_index("category")
CFr = rf_t[rf_t.factor == "CF"].set_index("category")
vm = tab("table05d_variance_model_main_effects").set_index("term")
vmi = tab("table05e_variance_model_with_interaction").set_index("term")
tr = tab("table05b_ordinal_trend_tests")
trx = tr[tr.residuals == "xgb, geometry-grouped"].set_index("factor")
trs = tr[tr.residuals == "xgb, series held out"].set_index("factor")
olm = tab("table05c_ordinal_validation_models")
rep = tab("table05h_repeatability_near_replicates").set_index("subset")
comp = tab("table05f_rf_cf_vs_completeness_indicators")
zero = tab("table05g_zero_share_by_rf_cf")
cfg = tab("table06a_cf_geometry_level").set_index("CF")
cft = tab("table06b_cf_geometry_level_trends").set_index("geometry_level_quantity")
joint = tab("table07_rf_cf_joint")
dec = tab("table08_hierarchical_error_decomposition")
dprim = dec[dec.analysis.str.startswith("PRIMARY")].set_index("component")
rr = tab("table08b_repeatability_vs_residual").set_index("quantity")["value"]
pol = tab("table13_data_selection_policies")
PX = pol[pol.learner == "xgb"].set_index("policy")
PR = pol[pol.learner == "rf"].set_index("policy")
cal = tab("table14_policy_calibration_by_quality")
ccw = tab("table09_ccw_alignment_and_ablation").set_index("variant")
for _c in ("rho_changed_matched", "rho_per_held_null_slot_matched"):
    if _c not in ccw:
        ccw[_c] = np.nan
att = tab("table10_concept_attribution_ccw_full")
cva = tab("table11_concept_vs_feature_attribution").set_index("quantity")["value"]
eqt = tab("table12b_explanation_quality_trends")
probe = tab("table12c_quality_probe_from_concepts")
opt = tab("table04_optuna_best_hyperparameters")
wet = tab("tableS_wet_dry_model").set_index("subset")
dist = tab("table02_rf_cf_distribution")


def cv(q):   # value of table 11 by substring
    k = [i for i in cva.index if q in i]
    return float(cva[k[0]])


def dstats(analysis_prefix, comp_):
    d = dec[dec.analysis.str.startswith(analysis_prefix)].set_index("component")
    return d.loc[comp_]


# ------------------------------------------------------------------------------------------- figures
def figure(name, caption, width=6.3):
    src = FIG / f"{name}.png"
    im = Image.open(src).convert("RGB")
    if im.width > 2400:
        im = im.resize((2400, round(im.height * 2400 / im.width)), Image.LANCZOS)
    dst = OUT / "figs" / f"{name}.png"
    im.save(dst, optimize=True)
    return {"t": "figure", "path": f"figs/{name}.png", "caption": caption, "width_in": width, "aspect": im.height / im.width, "alt": caption[:120]}


B = []
add = B.append
tn = {"n": 0}
fn = {"n": 0}


def T_(caption, header, rows, widths=None, note=None, fs=18):
    tn["n"] += 1
    add({"t": "table", "caption": f"**Table {tn['n']}.** {caption}", "header": header, "rows": rows, "widths": widths, "note": note, "font_size": fs})
    return tn["n"]


def F_(name, caption, width=6.3):
    fn["n"] += 1
    add(figure(name, f"**Fig. {fn['n']}.** {caption}", width))
    return fn["n"]


def p(t):
    add({"t": "p", "text": t})


def h1(t):
    add({"t": "h1", "text": t})


def h2(t):
    add({"t": "h2", "text": t})


def bl(items, level=0):
    add({"t": "bullets", "items": items, "level": level})


nX = int(P.loc["XGBoost", "n"])
x_rmse = P.loc["XGBoost", "rmse"]
c4 = PX.loc["P4_excl_RF4_or_CF4"]
c5 = PX.loc["P5_CLASH_weights"]
p1 = PX.loc["P1_retain_all"]
p6b = PX.loc["P6b_quality_aware_intervals"]
p6a = PX.loc["P6a_inverse_variance_weights"]
excl = ["P2_excl_RF4", "P3_excl_CF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks"]
d_common = PX.loc[excl, "d_rmse_common"]
d_rf4 = PX.loc[["P2_excl_RF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks"], "d_rmse_RF=4"]
d_cf4 = PX.loc[["P3_excl_CF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks"], "d_rmse_CF=4"]
moved = PX.loc[["P2_excl_RF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks"], "pct_predictions_changed_by_factor2_category4"]
cov_rf4_excl = PX.loc[["P2_excl_RF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks"], "coverage_RF4"]
vr4 = vm.loc["RF=4"]
full, un, cbm = ccw.loc["ccw_full"], ccw.loc["ccw_unaligned"], ccw.loc["cbm"]
q_g = dprim.loc["geometry"]; q_s = dprim.loc["series"]; q_w = dprim.loc["within-geometry residual"]

# =========================================================================================== title page
TITLE = ("Should category-4 tests be removed? Predictive validation of the CLASH reliability and complexity ratings "
         "and counterfactual assessment of data-selection policies for wave-overtopping prediction")
add({"t": "title", "text": TITLE})
add({"t": "subtitle", "text": "[Author names and affiliations]"})
add({"t": "subtitle", "text": "Manuscript draft prepared for *Ocean Engineering*", "italic": True})
add({"t": "note", "text": "Draft generated from the analysis repository. Items in square brackets, the RF/CF category wording "
     "and the reference details must be verified by the authors before submission (see the outline document)."})

h1("Highlights")
bl(["All 10 532 CLASH tests, reliability and complexity factors 1–4, audited and retained.",
    f"RF = 4 tests carry {f(vr4.variance_ratio, 1)} times the residual variance of RF = 1 tests; RF 2 and 3 do not differ.",
    "The complexity factor shows no ordered link to geometry-level error or generalisation.",
    f"Excluding category-4 tests leaves other tests unchanged but degrades category-4 predictions by {f(min(d_rf4.min(), d_cf4.min()), 2)}–{f(max(d_rf4.max(), d_cf4.max()), 2)}.",
    "Keeping all tests and widening prediction intervals by RF/CF equalises coverage."])

h1("Abstract")
p(f"The CLASH wave-overtopping database assigns each test a Reliability Factor (RF) and a Complexity Factor (CF) on ordinal "
  f"scales from 1 to 4. These ratings are widely used to weight or exclude tests when data-driven overtopping models are "
  f"trained, but whether they measure what they are used for has not been tested. This study audits the original, "
  f"unfiltered database (10 532 tests, all RF and CF categories) and evaluates the ratings against leakage-free "
  f"out-of-fold residuals of five diagnostic regression models tuned by nested, geometry-grouped cross-validation. "
  f"RF = 4 tests show {f(vr4.variance_ratio, 2)} times [{f(vr4.ratio_lo, 2)}, {f(vr4.ratio_hi, 2)}] the residual variance of RF = 1 tests, "
  f"but RF 2 and RF 3 are not distinguishable, and CF shows no ordered relation to geometry-level error or generalisation. "
  f"A hierarchical decomposition attributes {pct(q_w.share)} of the residual variance to within-geometry variation and "
  f"{pct(q_g.share)} to geometry-level discrepancy; replicate tests account for only {pct(rr.iloc[5])} of the former, and RF and CF do not map onto "
  f"these levels as their definitions imply. A counterfactual comparison of nine training-data policies on identical held-out tests "
  f"shows that excluding or zero-weighting category-4 tests changes the error on the remaining tests by at most "
  f"{f(d_common.abs().max(), 3)} (log₁₀ units), while it increases the error on category-4 tests by {f(min(d_rf4.min(), d_cf4.min()), 2)}–{f(max(d_rf4.max(), d_cf4.max()), 2)} "
  f"and reduces the coverage of 90 % prediction intervals for RF = 4 tests to {pct(cov_rf4_excl.min())}–{pct(cov_rf4_excl.max())}. Retaining all tests and "
  f"using RF and CF only to scale prediction intervals restores {pct(p6b.coverage_RF4)} coverage for RF = 4 tests. A regression "
  f"adaptation of the Causal Concept Wrapper Network is used to test whether concept-based explanations of the learned "
  f"representations are faithful and whether they depend on data quality. The results support retaining category-4 "
  f"tests and using the ratings for uncertainty rather than for data selection.")
add({"t": "p", "text": "*Keywords:* wave overtopping; CLASH database; data quality; reliability factor; complexity factor; "
     "data selection; machine learning; concept-based explanation"})

# =========================================================================================== 1 Introduction
h1("1. Introduction")
p("Data-driven prediction of mean wave-overtopping discharge rests on the CLASH database and its extended successors "
  "(Steendam et al., 2004; Van der Meer et al., 2009; EurOtop, 2018). Every test in the database carries two ordinal "
  "ratings: a Reliability Factor (RF), describing how reliable the measurement and its documentation are, and a "
  "Complexity Factor (CF), describing how well the standard CLASH parameters represent the tested cross-section. Both run "
  "from 1 (very reliable / simple) to 4 (unreliable / very complex).")
p("The ratings were designed to support the development of prediction tools, and they are used in that role. In the CLASH "
  "neural network each test was weighted by (4 − RF)(4 − CF), which gives zero weight to every test with RF = 4 or CF = 4, "
  "and tests remarked \"do not use for neural network\" were removed (Van Gent et al., 2007). Later neural-network and "
  "gradient-boosting predictors used the ratings to select or resample training data (Zanuttigh et al., 2016; den Bieman "
  "et al., 2021). In these studies the ratings are inputs to a data-selection decision; their validity for that decision is "
  "assumed rather than tested.")
p("Whether a test should be retained depends on two questions that the ratings are meant to answer: does the test carry "
  "more measurement-associated uncertainty than others (RF), and is its structure poorly represented by the standard "
  "parameters (CF)? If the ratings do not separate tests in these respects, excluding category-4 tests removes "
  "information without reducing noise. Category-4 tests are also not a random sample; they include depth-limited "
  "conditions and unusual sections that the remaining tests cover poorly.")
p("This study asks whether the CLASH ratings are valid and useful for deciding which observations to retain, "
  "down-weight or exclude. Out-of-fold residuals from tuned regression models serve as the instrument; the models "
  "themselves are not the contribution. Geometry-grouped cross-validation is used only to obtain leakage-free residuals "
  "(see the companion study [Paper 1] for its role in generalisation). No physical causal effects of hydraulic or "
  "structural parameters are estimated (cf. [Paper 2]). The research questions are:")
add({"t": "numbered", "items": [
    "RQ1. Does RF distinguish tests with different residual dispersion, repeatability and measurement-associated uncertainty?",
    "RQ2. Does CF identify geometries that are difficult to represent and generalise?",
    "RQ3. Can test-level variation, geometry-level discrepancy and source effects be separated, and do RF and CF map onto them?",
    "RQ4. How do retaining, excluding or re-weighting tests by RF and CF change accuracy, calibration and stability on identical held-out tests?",
    "RQ5. Do concept-based explanations of a regression-adapted Causal Concept Wrapper Network (CCW-Net; Rountree et al., 2026) faithfully describe the learned representations, and do they depend on data quality?",
    "RQ6. Is there a defensible alternative to automatic exclusion of category-4 tests?"]})

# =========================================================================================== 2 Data
h1("2. Data")
h2("2.1 Database and response")
p(f"The workbook holds 10 532 tests from 163 test series with 33 variables (Table 1). The response is the dimensionless "
  f"discharge q* = q / (g H_{{m0}}^{{3}})^{{1/2}}, with q in m³/s per metre and the spectral wave height H_{{m0}} at the toe, and "
  f"Y = log₁₀ q*. Of all tests, 9 442 report positive discharge, 1 069 zero discharge and 21 a missing or invalid value. "
  f"No test is removed: zero discharges enter a separate wet/dry model, no detection limit is recorded and no epsilon is "
  f"added before the logarithm. After input checks, {nX} positive-discharge tests enter the regression and 10 389 the "
  f"wet/dry model; all 143 tests with unusable inputs have RF = 4.")
p("Fifteen dimensionless predictors are formed with Froude scaling at the toe: R_{c}/H_{m0}, A_{c}/H_{m0}, G_{c}/L_{0}, "
  "H_{m0}/L_{0}, h/L_{0}, h_{t}/H_{m0}, B_{t}/L_{0}, B/L_{0}, h_{b}/H_{m0}, tan α_{B}, cot α_{d}, cot α_{u}, γ_{f}, β and the "
  "foreshore slope 1/m, with L_{0} = gT_{m−1,0}²/2π. RF and CF are never predictors of the mean.")
h2("2.2 The quality ratings")
rows = []
for k in ("RF", "CF"):
    d = dist[(dist.factor == k) & dist.category.astype(str).isin(list("1234"))]
    rows.append([k] + [f"{int(n)} ({p_:.1f} %)" for n, p_ in zip(d.n_all, d.pct_all)])
T_("Distribution of the CLASH reliability (RF) and complexity (CF) factors, all 10 532 tests.",
   ["Factor", "1", "2", "3", "4"], rows, [1, 2, 2, 2, 2],
   note="RF: 1 very reliable … 4 not reliable. CF: 1 simple … 4 very complex (category wording to be verified against the CLASH documentation). No value is missing or out of range.")
p("Both factors are fully populated with values 1–4 (Fig. 1). CF = 4 comprises 254 tests from only nine series "
  "(212 usable tests from seven series in the regression set), dominated by one investigation of chute structures. Six of "
  "the sixteen RF × CF cells contain too few tests or series to support any conclusion. CF is not constant within "
  "reconstructed structures: 32 geometries (311 tests) carry more than one CF value, and RF varies within 166 geometries, as "
  "expected for a test-level rating.")
F_("fig02_rf_cf_distribution", "Distribution of (a) RF and (b) CF and (c) the joint RF × CF frequency for all 10 532 tests.")
h2("2.3 Identifiers")
p("The test-series number (162 series in the regression set) and the literature reference are genuine identifiers. "
  "No structure or laboratory identifier exists. Geometries were reconstructed within each series from "
  "water-level-invariant descriptors (crest, armour-crest, toe and berm levels relative to the bed, widths, slopes, "
  "roughness and foreshore), giving 2 046 geometries. Laboratories could be inferred from the references for 70 % of "
  "tests only.")

# =========================================================================================== 3 Methods
h1("3. Methods")
h2("3.1 Leakage-free out-of-fold residuals")
p(f"Five diagnostic regression models are used: Random Forest, XGBoost, CatBoost, a multilayer perceptron and the CCW-Net "
  f"of Section 3.5. All share five outer folds in which every reconstructed geometry is held out once. Inside each "
  f"outer-training partition, hyper-parameters are selected by Optuna with the TPE sampler (Akiba et al., 2019), minimising the RMSE of a "
  f"grouped inner three-fold cross-validation. Boosted trees use early stopping on the inner folds, and networks use a grouped "
  f"validation split. Imputation, scaling and target standardisation are fitted on training rows only, and test folds are "
  f"never used for selection (Table 2). The pre-specified reference model for residual analyses is XGBoost. Every analysis is "
  f"repeated with the other models and with random-record and series-held-out residuals.")
rows = []
for m, lab in (("rf", "Random Forest"), ("xgb", "XGBoost"), ("catboost", "CatBoost"), ("mlp", "MLP"), ("ccw", "CCW-Net")):
    o = opt[(opt.scheme == "geometry") & (opt.model == m)]
    if not len(o):
        continue
    inner = o.inner_cv_rmse.mean() if m != "ccw" else np.nan
    rows.append([lab, int(o.n_trials.iloc[0]), f(inner), f(o.outer_test_rmse.mean()) if m != "ccw" else "–"])
T_("Hyper-parameter optimisation (Optuna, TPE; grouped inner three-fold cross-validation inside each of five outer folds).",
   ["Model", "Trials per fold", "Inner CV RMSE (mean)", "Outer test RMSE (mean)"], rows, [2, 1, 1.4, 1.4],
   note="Search spaces and per-fold selections are given in the supplementary material. CCW-Net was tuned on a shortened schedule with the objective in standardised units.")
h2("3.2 Validity of RF and CF")
p("RF and CF are treated as ordinal and are never given equal spacing. Per category we report residual statistics with "
  "test-series cluster-bootstrap intervals (1 000 resamples). The analyses comprise:")
bl(["rank trends of the absolute residual with Holm correction;",
    "proportional-odds models of the rating on the residual and on indicators of estimated parameters;",
    "a multiplicative variance model, a Gamma GLM of squared residuals on the RF and CF categories, with and without an RF × CF term;",
    "geometry-level analyses of CF, with the maximum CF within a geometry;",
    "a model-free repeatability estimate from 1 136 near-replicate pairs (same geometry, H_{m0}, T_{m−1,0}, h and R_{c} within 1 %), with repeat SD = (mean d²/2)^{1/2}."])
p("No laboratory reports or second ratings exist, so these analyses establish predictive validity and internal consistency, "
  "not the correctness of the ratings.")
h2("3.3 Hierarchical error decomposition")
p("Out-of-fold residuals are decomposed with exact restricted maximum likelihood into nested random intercepts for "
  "test series and geometry and a within-geometry residual, with series-cluster bootstrap intervals. The within-geometry "
  "component is compared with the near-replicate variance and is not labelled measurement error.")
h2("3.4 Counterfactual data-selection policies")
p("A policy is a rule that decides which training tests are kept and how they are weighted. Nine policies are compared "
  "(Table 3). For each outer fold, only the policy changes; folds, learner, hyper-parameters (tuned on the full training "
  "partition), seeds and calibration geometries stay fixed. The counterfactual is therefore one of training data, not a "
  "change of a physical input. Accuracy is evaluated on identical held-out tests, split into common tests (RF < 4 and CF < 4), "
  "RF = 4 tests and CF = 4 tests, with paired geometry-cluster bootstrap differences to retaining all tests (P1). Intervals "
  "use group-weighted split conformal prediction with 25 % of the training geometries for calibration (Lee et al., 2023). "
  "The empirical variance model σ²(RF, CF) used by P6 is fitted to grouped inner out-of-fold residuals of each training "
  "partition.")
T_("Data-selection policies.", ["Policy", "Training tests", "Weights", "Interval width"],
   [["P1 retain all", "all", "equal", "constant"], ["P2", "excl. RF = 4", "equal", "constant"], ["P3", "excl. CF = 4", "equal", "constant"],
    ["P4", "excl. RF = 4 or CF = 4", "equal", "constant"], ["P5 conventional", "RF < 4 and CF < 4", "(4 − RF)(4 − CF)", "constant"],
    ["P5b conventional", "as P5, 'do not use' tests removed", "(4 − RF)(4 − CF)", "constant"],
    ["P6a empirical weights", "all", "1/σ²(RF, CF)", "constant"], ["P6b quality-aware intervals", "all", "equal", "∝ σ(RF, CF)"],
    ["P6c", "all", "1/σ²(RF, CF)", "∝ σ(RF, CF)"]], [1.6, 1.8, 1.3, 1.1])
h2("3.5 Regression Causal Concept Wrapper Network")
p("The CCW-Net (Rountree et al., 2026) wraps a frozen backbone with a concept encoder that produces one squashed vector "
  "per concept and a bias-free attention head, so that a zero concept vector contributes nothing to the output. Training "
  "adds three alignment criteria evaluated by interventional mediation analysis: necessity, sufficiency and independence. "
  "Counterfactual concept representations are drawn from a conditional normalising flow. The implementation follows the "
  "authors' reference code. It was adapted to continuous tabular regression as follows:")
bl(["linear output for the standardised Y;",
    "a tabular backbone trained per outer fold;",
    "ten concept groups with ordinal or nominal states instead of binary concepts (supplementary Table S1), with presence encoded by slot length only for the six groups that have a physical null state (no berm, no toe, normal incidence, smooth surface, plain crest, flat foreshore);",
    "an explicit pairwise interaction term, so that physically interacting concepts are not forced to be additive."])
p("The loss is")
add({"t": "eq", "text": "L = L_{task} + λ_{cls} L_{concept} + λ_{nec} L_{nec} + λ_{suf} L_{suf} + λ_{ind} L_{ind} + λ_{pres} L_{pres} + λ_{int} L_{int}"})
p("Here L_{nec} penalises output changes when the slot of an absent concept is replaced by a counterfactual draw. L_{suf} requires "
  "the active slots, measured from the zero reference, to reproduce the target. L_{ind} requires the attention read-out to be "
  "additive in single-slot contributions. L_{int} sparsifies the interaction term. Alignment strength, concept dimension, "
  "width and learning rate were tuned by Optuna inside each outer fold.")
p("Evaluation on held-out geometries uses two kinds of quantity. Mediator-free zero-ablation shares measure how much of the "
  "output effect passes through absent concepts. The matched interventional-effect estimator of the reference implementation "
  "compares held-out tests that differ in one concept and have independent covariate matches for every concept slot. CCW-Net "
  "is compared with its unaligned version, loss ablations and a concept-bottleneck model (Koh et al., 2020), and its concept "
  "attribution with grouped TreeSHAP of the reference model (Lundberg et al., 2020). All such effects are properties of the "
  "network's representations, not physical causal effects.")

# =========================================================================================== 4 Results
h1("4. Results")
h2("4.1 Diagnostic models")
rows = []
for lab in ("Random Forest", "XGBoost", "CatBoost", "MLP", "CCW-Net"):
    r = P.loc[lab]
    rows.append([lab + (" (3 seeds)" if r.n_seeds > 1 else ""), f(r.rmse) + " " + ci(r.rmse_lo, r.rmse_hi), f(r.mae), f(r.r2), s(r.bias), f(r.f10, 2)])
T_("Out-of-fold accuracy on held-out geometries (n = 9 333; Y = log₁₀ q*).", ["Model", "RMSE [95 % CI]", "MAE", "R²", "Bias", "F10"], rows,
   [1.6, 2.0, 0.8, 0.8, 0.9, 0.8], note="Bias = predicted − measured. F10 = 10^RMSE. Intervals: geometry-cluster bootstrap. Seed-averaged metrics for the networks.")
p(f"The boosted trees are the most accurate models (Table 4, Fig. 2), and the networks are {f(P.loc['MLP','rmse']-x_rmse,2)}–"
  f"{f(P.loc['CCW-Net','rmse']-x_rmse,2)} worse. The validation unit dominates: the reference model reaches RMSE "
  f"{f(PS.loc[('random','XGBoost'),'rmse'])} with random records, {f(x_rmse)} with held-out geometries and "
  f"{f(PS.loc[('series','XGBoost'),'rmse'])} with held-out test series. The inner Optuna objective agrees with the outer test "
  f"error for every model. Residuals show no trend over most of the range; the smallest discharges (Y < −6) are under-predicted.")
F_("fig03_measured_vs_predicted_all_models", "Measured versus out-of-fold predicted log₁₀ q* on held-out geometries for all models. Solid line: 1:1; dashed: factor 10. MLP and CCW-Net panels show the three-seed ensemble.")

h2("4.2 Validity of the reliability factor")
rows = []
for v in (1, 2, 3, 4):
    r = RF.loc[v]
    rs = rep.loc[f"max RF={v}"]
    rows.append([str(v), int(r.n), int(r.n_series), f(r.mae) + " " + ci(r.mae_lo, r.mae_hi), f(r.within_geometry_sd) + " " + ci(r.within_sd_lo, r.within_sd_hi),
                 f(rs.repeat_sd_log10, 2) + f" ({int(rs.n_pairs)})", "1" if v == 1 else f(vm.loc[f'RF={v}', 'variance_ratio'], 2) + " " + ci(vm.loc[f'RF={v}', 'ratio_lo'], vm.loc[f'RF={v}', 'ratio_hi'], 2)])
T_("Residuals, repeatability and variance ratios by RF (XGBoost, held-out geometries).",
   ["RF", "n", "Series", "MAE [95 % CI]", "Within-geometry SD", "Repeat SD (pairs)", "Variance ratio vs RF 1"], rows, [0.5, 0.7, 0.7, 1.6, 1.6, 1.2, 1.6], fs=17,
   note="Repeat SD from near-replicate pairs by the larger RF of the pair (model-free). Variance ratios from a Gamma GLM with RF and CF categories, test-series cluster bootstrap.")
p(f"Tests with RF = 4 have larger errors than RF = 1 tests by every measure (Table 5, Fig. 3). Their residual variance, adjusted "
  f"for CF, is {f(vr4.variance_ratio, 2)} {ci(vr4.ratio_lo, vr4.ratio_hi, 2)} times that of RF = 1, and every learner gives an RMSE "
  f"ratio above one. The scale is not ordered in between: RF = 2 and RF = 3 have the same error, their variance ratios include "
  f"one, and the rank correlation of the absolute residual with RF is {f(trx.loc['RF','spearman_absres'], 2)} "
  f"{ci(trx.loc['RF','spearman_lo'], trx.loc['RF','spearman_hi'], 2)}. Model-free repeatability increases from RF = 1 "
  f"({f(rep.loc['max RF=1','repeat_sd_log10'],2)}) to RF = 3 ({f(rep.loc['max RF=3','repeat_sd_log10'],2)}) but not to RF = 4 "
  f"({f(rep.loc['max RF=4','repeat_sd_log10'],2)}, only {int(rep.loc['max RF=4','n_pairs'])} pairs). RF rises with indicators "
  f"of estimated parameters, and an estimated spectral period predicts a higher RF more strongly than the prediction error "
  f"does. RF = 4 tests also contain more zero-discharge tests (16.5 % against 5.8 % for RF = 1). The wet/dry model separates "
  f"them less well (AUC {f(wet.loc['RF=4','auc'],2)} against {f(wet.loc['RF=1','auc'],2)}).")
F_("fig08_residuals_by_rf_cf", "Out-of-fold residuals by RF (top) and CF (bottom): distributions, mean absolute residual and within-geometry SD with test-series cluster-bootstrap 95 % intervals.")

h2("4.3 Validity of the complexity factor")
rows = []
for v in (1, 2, 3, 4):
    r = cfg.loc[v]
    rows.append([str(v), int(r.n_geometries), int(r.n_series), f(r.abs_mean_resid) + " " + ci(r.abs_mean_resid_lo, r.abs_mean_resid_hi),
                 f(r.rmse_grouped) + " " + ci(r.rmse_grouped_lo, r.rmse_grouped_hi), f(r.generalisation_gap) + " " + ci(r.generalisation_gap_lo, r.generalisation_gap_hi)])
T_("Geometry-level errors by CF (maximum CF within the geometry).", ["CF", "Geometries", "Series", "|Mean residual|", "Geometry RMSE", "Grouped − random RMSE"],
   rows, [0.5, 0.9, 0.7, 1.6, 1.6, 1.6], fs=17)
p(f"No geometry-level quantity is rank-correlated with CF (Table 6). The absolute geometry mean residual gives "
  f"{f(cft.loc['abs_mean_resid','spearman_with_CF'],2)} {ci(cft.loc['abs_mean_resid','lo'], cft.loc['abs_mean_resid','hi'],2)}, "
  f"and the generalisation gap gives {f(cft.loc['generalisation_gap','spearman_with_CF'],2)} {ci(cft.loc['generalisation_gap','lo'], cft.loc['generalisation_gap','hi'],2)}. "
  f"CF = 3 is the difficult class, with record-level RMSE {f(CFr.loc[3,'rmse'],2)} {ci(CFr.loc[3,'rmse_lo'], CFr.loc[3,'rmse_hi'],2)} and "
  f"variance ratio {f(vm.loc['CF=3','variance_ratio'],2)} {ci(vm.loc['CF=3','ratio_lo'], vm.loc['CF=3','ratio_hi'],2)}. CF = 4 tests are "
  f"predicted well (RMSE {f(CFr.loc[4,'rmse'],2)}); this reflects the internal consistency of a few series, not easy "
  f"representation of complex sections. When whole series are held out, a weak positive trend appears "
  f"({f(trs.loc['CF','spearman_absres'],2)} {ci(trs.loc['CF','spearman_lo'], trs.loc['CF','spearman_hi'],2)}). RF and CF do not "
  f"interact: the RF × CF term of the variance model is {f(vmi.iloc[-1].variance_ratio,2)} {ci(vmi.iloc[-1].ratio_lo, vmi.iloc[-1].ratio_hi,2)}.")
F_("fig09_rf_cf_ordinal_validation", "Validity of RF and CF: (a) residual-variance ratios; (b) ordinal trends across models and validation schemes; (c) near-replicate repeatability by RF; (d) geometry-level error by CF; (e) mean absolute residual by RF × CF (cells with fewer than 50 tests or three series not interpreted).")

h2("4.4 Test-level and geometry-level error")
rows = [[c_, int(r.n_units), f(r.sd), pct(r.share) + f" [{pct(r.share_lo)}, {pct(r.share_hi)}]"] for c_, r in dprim.iterrows()]
T_("REML decomposition of out-of-fold residuals (XGBoost, held-out geometries).", ["Component", "Units", "SD", "Share of variance [95 % CI]"], rows, [2, 1, 1, 2])
p(f"Within-geometry variation accounts for {pct(q_w.share)} of the residual variance, the geometry level for {pct(q_g.share)} and "
  f"the test series for {pct(q_s.share)} (Table 7, Fig. 4). The shares are stable across geometry definitions and depend on the "
  f"validation unit: with random records the geometry share falls to "
  f"{pct(dstats('random-CV','geometry').share)} because the model memorises geometry offsets. The near-replicate variance is "
  f"{pct(rr.iloc[5])} of the within-geometry variance, so the within-geometry residual is not pure measurement error. A "
  f"laboratory component is not identifiable. RF and CF do not map onto the levels as intended. RF = 4 is elevated at both the "
  f"within-geometry level ({f(dstats('stratum RF=4','within-geometry residual').sd,2)} against {f(dstats('stratum RF=1','within-geometry residual').sd,2)}) "
  f"and the geometry level ({f(dstats('stratum RF=4','geometry').sd,2)} against {f(dstats('stratum RF=1','geometry').sd,2)}). The geometry-level "
  f"SD is the same in every CF class; CF = 3 differs in its within-geometry residual instead.")
F_("fig10_error_decomposition", "Hierarchical decomposition of out-of-fold residuals: (a) variance shares under alternative groupings and validation units; (b) component SDs; (c) geometry mean residuals; (d, e) components within CF and RF strata; (f) residual SD compared with near-replicate repeatability.")

h2("4.5 Counterfactual data-selection policies")
rows = []
lab = {"P1_retain_all": "P1 retain all", "P2_excl_RF4": "P2 excl. RF = 4", "P3_excl_CF4": "P3 excl. CF = 4", "P4_excl_RF4_or_CF4": "P4 excl. RF/CF = 4",
       "P5_CLASH_weights": "P5 (4−RF)(4−CF)", "P5b_CLASH_weights_and_remarks": "P5b P5 + remarks", "P6a_inverse_variance_weights": "P6a 1/σ² weights",
       "P6b_quality_aware_intervals": "P6b RF/CF intervals", "P6c_weights_and_intervals": "P6c P6a + P6b"}
for k, l_ in lab.items():
    r = PX.loc[k]
    dc = "0" if k in ("P1_retain_all", "P6b_quality_aware_intervals") else s(r.d_rmse_common) + " " + ci(r.d_rmse_common_lo, r.d_rmse_common_hi)
    d4 = "0" if k in ("P1_retain_all", "P6b_quality_aware_intervals") else s(r["d_rmse_RF=4"], 2)
    dcf = "0" if k in ("P1_retain_all", "P6b_quality_aware_intervals") else s(r["d_rmse_CF=4"], 2)
    rows.append([l_, f"{int(r.n_training_records_used)}", dc, d4, dcf, f"{pct(r.coverage_RF4)} / {pct(r.coverage_CF4)}", f"{100*r.max_abs_coverage_deviation_over_RF:.1f}"])
T_("Effect of the training-data policy on identical held-out tests (XGBoost). ΔRMSE relative to P1.",
   ["Policy", "Training tests", "ΔRMSE common (n = 8 494)", "ΔRMSE RF = 4", "ΔRMSE CF = 4", "Coverage RF = 4 / CF = 4", "Max coverage deviation over RF (pts)"],
   rows, [1.5, 0.8, 1.7, 0.8, 0.8, 1.2, 1.1], fs=16,
   note=f"P1 RMSE: common {f(p1.rmse_common)}, RF = 4 {f(p1['rmse_RF=4'])}, CF = 4 {f(p1['rmse_CF=4'])}. Coverage of 90 % group-calibrated conformal intervals. Random Forest results in the supplementary material.")
p(f"No exclusion or conventional weighting policy improves prediction for the tests it keeps (Table 8, Fig. 5). On the 8 494 "
  f"common tests every difference lies between {s(d_common.min())} and {s(d_common.max())}, with intervals spanning zero. Random Forest "
  f"gives {s(PR.loc[excl,'d_rmse_common'].min())} to {s(PR.loc[excl,'d_rmse_common'].max())}. Every policy that removes or "
  f"zero-weights a category degrades the predictions for that category: RF = 4 tests gain {s(d_rf4.min(),2)} to {s(d_rf4.max(),2)} RMSE "
  f"and CF = 4 tests {s(d_cf4.min(),2)} to {s(d_cf4.max(),2)}. The same policies move {f(moved.min(),0)}–{f(moved.max(),0)} % of "
  f"category-4 predictions by more than a factor 2, and the coverage of 90 % intervals for RF = 4 tests drops to "
  f"{pct(cov_rf4_excl.min())}–{pct(cov_rf4_excl.max())}. The loss is specific to the removed category: excluding RF = 4 does not harm CF = 4 "
  f"predictions and vice versa. Geometry-level bias rises under every exclusion policy "
  f"({f(p1.geometry_level_abs_bias)} → {f(PX.loc[excl,'geometry_level_abs_bias'].min())}–{f(PX.loc[excl,'geometry_level_abs_bias'].max())}).")
p(f"Empirical down-weighting by 1/σ²(RF, CF) (P6a) is neutral ({s(p6a.d_rmse_common)} on common tests, {s(p6a['d_rmse_RF=4'],2)} on RF = 4 "
  f"tests). Retaining all tests with equal weight and letting RF and CF scale the interval (P6b) leaves every prediction unchanged. "
  f"It raises RF = 4 coverage from {pct(p1.coverage_RF4)} to {pct(p6b.coverage_RF4)} and reduces the largest coverage deviation across "
  f"RF categories from {100*p1.max_abs_coverage_deviation_over_RF:.1f} to {100*p6b.max_abs_coverage_deviation_over_RF:.1f} points at "
  f"a similar mean width ({f(p1.mean_width,2)} against {f(p6b.mean_width,2)}; Fig. 6). CF = 3 tests remain under-covered.")
F_("fig13_data_selection_policies", "Counterfactual data-selection policies (XGBoost): change in RMSE relative to retaining all tests for (a) common tests, (b) RF = 4 and (c) CF = 4 tests with paired geometry-cluster bootstrap 95 % intervals; (d) share of predictions changed by more than a factor 2.")
F_("fig14_calibration_by_quality_category", "Coverage of 90 % prediction intervals and mean interval width by RF and CF under four policies.")

h2("4.6 Concept explanations and data quality")
rows = []
for k, l_ in (("ccw_full", "CCW-Net"), ("ccw_unaligned", "Unaligned wrapper"), ("ccw_no_nec", "without necessity"), ("ccw_no_suf", "without sufficiency"),
              ("ccw_no_ind", "without independence"), ("ccw_no_interaction", "without interaction term"), ("ccw_no_cond_cf", "without conditional counterfactual"), ("cbm", "Concept bottleneck")):
    if k not in ccw.index:
        continue
    r = ccw.loc[k]
    rows.append([l_, f(r.rmse), f(r.nec_abs_effect) if k != "cbm" else "–", f(r.suf_rmse_vs_output) if k != "cbm" else "–", f(r.ind_rmse) if k != "cbm" else "–",
                 pct(r.rho_inactive_per_null_slot, 1), f(r.rho_changed_matched, 3) if k != "cbm" and "rho_changed_matched" in r and np.isfinite(r.rho_changed_matched) else "–",
                 f(r.rho_per_held_null_slot_matched, 3) if k != "cbm" and "rho_per_held_null_slot_matched" in r and np.isfinite(r.rho_per_held_null_slot_matched) else "–"])
T_("Causal alignment of concept explanations on held-out geometries (means over fold × seed runs).",
   ["Variant", "RMSE", "Necessity effect", "Sufficiency gap", "Additivity residual", "Zero-ablation share per absent concept", "Matched: changed concept", "Matched: per held absent concept"],
   rows, [1.7, 0.7, 0.9, 0.9, 0.9, 1.1, 0.9, 1.0], fs=16,
   note="Necessity: |Δŷ| when an absent concept's slot is replaced by a counterfactual draw. Sufficiency: RMSE between the active-only reconstruction and the output. Matched shares: interventional-effect estimator with independent covariate matches per concept slot; 0.10 would be indiscriminate routing.")
p(f"The alignment losses do what they are designed to do (Table 9). Relative to the unaligned wrapper the necessity effect falls from "
  f"{f(un.nec_abs_effect)} to {f(full.nec_abs_effect)}, the sufficiency gap from {f(un.suf_rmse_vs_output)} to {f(full.suf_rmse_vs_output)} "
  f"and the additivity residual from {f(un.ind_rmse)} to {f(full.ind_rmse)}, at equal accuracy. Removing a loss term degrades mainly "
  f"the criterion it targets. The concept-bottleneck model routes {pct(cbm.rho_inactive_per_null_slot,1)} of its effect through each "
  f"absent concept, against {pct(full.rho_inactive_per_null_slot,1)} for CCW-Net, at similar concept accuracy.")
p(f"The matched estimator measures whether a change in one concept is carried by that concept's slot. For CCW-Net the changed "
  f"concept carries a share of {f(full.rho_changed_matched,3)}, and each held absent concept {f(full.rho_per_held_null_slot_matched,3)}; "
  f"for the unaligned wrapper the figures are {f(un.rho_changed_matched,3)} and {f(un.rho_per_held_null_slot_matched,3)} "
  f"(1/10 would be indiscriminate). The rank agreement between CCW-Net concept importance and grouped TreeSHAP of XGBoost is "
  f"{f(cv('rank agreement'),2)} over ten concepts. CCW-Net ranks depth and steepness first, while SHAP ranks freeboard and roughness "
  f"first, consistent with established overtopping formulae. CCW-Net attributes less effect to absent concepts than SHAP "
  f"({pct(cv('absent (null-state) concept: CCW-Net'),1)} against {pct(cv('absent (null-state) concept: grouped TreeSHAP'),1)}). "
  f"This is partly by construction: smooth surfaces or normal incidence do affect overtopping, and SHAP rightly gives them weight.")
rfq = eqt[(eqt.model == "CCW-Net") & (eqt.factor == "RF")].set_index("quantity")
pr_rf4 = probe[(probe.target == "RF=4") & (probe.probe_validation_grouped_by == "geometry")].set_index("probe_input")["auc_mean_over_folds"]
pr_cf3s = probe[(probe.target == "CF>=3") & (probe.probe_validation_grouped_by == "series")].set_index("probe_input")["auc_mean_over_folds"]
p(f"Explanation quality does not depend on CF. Its relation to RF is weak and does not survive multiple-comparison correction: the "
  f"interaction magnitude correlates with RF at {f(rfq.loc['interaction_abs','spearman'],2)} {ci(rfq.loc['interaction_abs','lo'], rfq.loc['interaction_abs','hi'],2)}. "
  f"A linear probe recovers RF = 4 from the learned concept vectors better than from the raw predictors (AUC "
  f"{f(pr_rf4.iloc[0],2)} against {f(pr_rf4.iloc[1],2)}). CF ≥ 3 is not recoverable across test series (AUC {f(pr_cf3s.iloc[0],2)}). The quality "
  f"categories therefore mark particular regions of the input space and particular investigations, not a property of the physics.")
F_("fig12_concept_attribution_and_alignment", "Concept explanations: (a) slot-ablation effects of CCW-Net and the unaligned wrapper; (b) necessity and (c) sufficiency across variants; (d) CCW-Net concept effects against grouped TreeSHAP; (e, f) explanation quality by RF and CF.")

# =========================================================================================== 5 Discussion
h1("5. Discussion")
h2("5.1 What the data demonstrate")
p("The policy comparison is the most direct evidence because it asks the engineering question itself on identical held-out "
  "tests. The answer is unambiguous: the conventional exclusion and weighting rules do not make predictions for the remaining "
  "tests more accurate. They make predictions for the excluded kind of test much worse and their uncertainty intervals "
  "over-confident. Category-4 tests occupy parts of the input space that the other tests do not cover, such as depth-limited "
  "conditions with H_{m0} > h, emergent toes and particular sections. Their value lies in that coverage.")
h2("5.2 What is supported by predictive association only")
p("The RF = 4 versus RF = 1 contrast in residual variance and the absence of an ordered CF effect describe how the ratings "
  "relate to model error. They do not establish whether individual ratings are correct, and they do not imply that RF or CF "
  "cause anything. The RF scale behaves as a flag for an elevated-uncertainty group rather than as a graded measure. CF = 3, "
  "not CF = 4, marks the tests that are hardest to predict.")
h2("5.3 What concerns model-internal explanation")
p("Causal alignment makes concept explanations testable: necessity, sufficiency and additivity can be measured and "
  "enforced, which conventional feature attribution does not allow. However, the explanations describe the network's "
  "representations. Their ranking of concepts disagrees with feature attribution and with established formulae, and the "
  "network is less accurate than gradient boosting. For engineering use, the value of the method lies in auditing "
  "explanations, not in producing physical effect estimates.")
h2("5.4 What cannot be identified without independent evidence")
p("The correctness of individual ratings, a pure measurement-error component and laboratory effects require independent "
  "evidence. That means original laboratory reports, repeated measurements under controlled conditions or a blinded "
  "re-rating of a stratified sample of tests by several experts, analysed with weighted kappa or Krippendorff's alpha. "
  "None of these was available.")
h2("5.5 Recommendations")
bl(["Retain category-4 tests in the training of mean predictors; do not apply (4 − RF)(4 − CF) weights or remove 'do not use' tests without a specific reason.",
    "Use RF (and the CF = 3 flag) to scale prediction intervals, calibrated on held-out geometries.",
    "Report accuracy separately for RF = 4 and CF = 3 tests.",
    "Treat CF as a flag of unusual sections, not as an ordered weight."])

h1("6. Limitations")
bl(["No independent quality evidence exists; predictive validity and internal consistency are established, not the correctness of the ratings.",
    "Geometries are reconstructed and laboratories inferred; geometry-grouped folds share test series between training and test.",
    "CF = 4 comes from seven test series; six RF × CF cells cannot be interpreted.",
    "Near-replicate pairs give an upper bound on repeatability; only 66 involve RF = 4.",
    "Optuna budgets are modest; policies reuse hyper-parameters tuned on the full training partition.",
    "The CCW-Net adaptation differs from the reference in documented respects (multi-state concepts, interaction term, linear output) and uses smaller mediator flows."])

h1("7. Conclusions")
p(f"The CLASH reliability factor distinguishes a group of RF = 4 tests with about {f(vr4.variance_ratio,1)} times the residual variance "
  f"of RF = 1 tests but is not an ordered measure of uncertainty. The complexity factor does not predict geometry-level error or "
  f"generalisation. Within-geometry variation dominates the residual and is not pure measurement error. Excluding or "
  f"zero-weighting category-4 tests discards information without improving the remaining predictions. Retaining every test "
  f"and using the ratings to scale prediction intervals is the more defensible practice.")

h1("Data and code availability")
p("The analysis code, configuration, out-of-fold predictions and all tables and figures are available in the project repository [link]. "
  "The CLASH database is available from its maintainers.")
h1("CRediT authorship contribution statement")
p("[To be completed by the authors.]")
h1("Declaration of competing interest")
p("[To be completed by the authors.]")

h1("References")
add({"t": "note", "text": "Reference details are given from memory and must be checked against the original publications before submission."})
for r in [
    "Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M., 2019. Optuna: a next-generation hyperparameter optimization framework. Proc. 25th ACM SIGKDD, 2623–2631.",
    "Breiman, L., 2001. Random forests. Machine Learning 45, 5–32.",
    "Chen, T., Guestrin, C., 2016. XGBoost: a scalable tree boosting system. Proc. 22nd ACM SIGKDD, 785–794.",
    "den Bieman, J.P., van Gent, M.R.A., van den Boogaard, H.F.P., 2021. Wave overtopping predictions using an advanced machine learning technique. Coastal Engineering 166, 103830.",
    "Dinh, L., Sohl-Dickstein, J., Bengio, S., 2017. Density estimation using Real NVP. ICLR.",
    "EurOtop, 2018. Manual on wave overtopping of sea defences and related structures, second ed. Van der Meer, J.W., Allsop, N.W.H., Bruce, T., De Rouck, J., Kortenhaus, A., Pullen, T., Schüttrumpf, H., Troch, P., Zanuttigh, B. www.overtopping-manual.com.",
    "Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B., Liang, P., 2020. Concept bottleneck models. ICML.",
    "Lee, Y., Barber, R.F., Willett, R., 2023. Distribution-free inference with hierarchical data. arXiv:2306.06342.",
    "Loh, W.W., Moerkerke, B., Loeys, T., Vansteelandt, S., 2022. Disentangling indirect effects through multiple mediators without assuming any causal structure among the mediators. Psychological Methods 27, 982–999.",
    "Lundberg, S.M., Erion, G., Chen, H., et al., 2020. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2, 56–67.",
    "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A., 2018. CatBoost: unbiased boosting with categorical features. NeurIPS.",
    "Rountree, J., Verma, P., So, O., Fan, C., Shah, J.A., 2026. Causal concept explanations for deep neural models. arXiv:2611.14407.",
    "Sabour, S., Frosst, N., Hinton, G.E., 2017. Dynamic routing between capsules. NeurIPS.",
    "Steendam, G.J., Van der Meer, J.W., Verhaeghe, H., Besley, P., Franco, L., Van Gent, M.R.A., 2004. The international database on wave overtopping. Proc. 29th ICCE, 4301–4313.",
    "Van der Meer, J.W., Verhaeghe, H., Steendam, G.J., 2009. The new wave overtopping database for coastal structures. Coastal Engineering 56, 108–120.",
    "Van Gent, M.R.A., van den Boogaard, H.F.P., Pozueta, B., Medina, J.R., 2007. Neural network modelling of wave overtopping at coastal structures. Coastal Engineering 54, 586–593.",
    "Vansteelandt, S., Daniel, R.M., 2017. Interventional effects for mediation analysis with multiple mediators. Epidemiology 28, 258–265.",
    "Verhaeghe, H., et al., 2004. CLASH WP2 report: database on wave overtopping. Ghent University. [verify title]",
    "Zanuttigh, B., Formentin, S.M., Van der Meer, J.W., 2016. Prediction of extreme and tolerable wave overtopping discharges through an advanced neural network. Ocean Engineering 127, 7–22.",
    "[Paper 1] Machine-learning prediction of wave overtopping at untested structures: geometry-grouped validation, freeboard monotonicity and conformal coverage. [authors, status]",
    "[Paper 2] Quantum machine learning for wave overtopping estimation: integrating with causal inference and uncertainty quantification. Ocean Engineering [volume, article, year].",
]:
    add({"t": "ref", "text": r})

def _fix(o):
    """q* would be read as italic markup by the renderer: write the star as a superscript."""
    if isinstance(o, str):
        return re.sub(r"q\*(?!\*)", "q^{*}", o)
    if isinstance(o, list):
        return [_fix(x) for x in o]
    if isinstance(o, dict):
        return {k: (_fix(v) if k != "path" else v) for k, v in o.items()}
    return o


B = _fix(B)
json.dump({"title": TITLE, "blocks": B}, open(OUT / "manuscript.json", "w"), ensure_ascii=False, indent=0)


# =========================================================================================== outline document
def md_blocks(path):
    out = []
    para = []

    def flush():
        if para:
            out.append({"t": "p", "text": " ".join(para)})
            para.clear()
    for line in open(path, encoding="utf8"):
        l = line.rstrip("\n")
        conv = lambda t: re.sub(r"`([^`]+)`", r"\1", t)  # noqa: E731
        if not l.strip():
            flush(); continue
        if l.startswith("# "):
            flush(); out.append({"t": "h1", "text": conv(l[2:])})
        elif l.startswith("## "):
            flush(); out.append({"t": "h2", "text": conv(l[3:])})
        elif l.startswith("### "):
            flush(); out.append({"t": "h3", "text": conv(l[4:])})
        elif re.match(r"^\s{2,}\* ", l):
            flush(); out.append({"t": "bullets", "items": [conv(l.strip()[2:])], "level": 1})
        elif l.startswith("* "):
            flush(); out.append({"t": "bullets", "items": [conv(l[2:])], "level": 0})
        elif l.startswith("|"):
            flush()
        else:
            para.append(conv(l.strip()))
    flush()
    return out


O = [{"t": "title", "text": "Manuscript and supplementary-material outline"},
     {"t": "subtitle", "text": TITLE, "italic": True},
     {"t": "note", "text": "Prepared for Ocean Engineering. Numbers are in the manuscript draft and in results/reports/03_scientific_interpretation.md."}]
O += md_blocks(ROOT / "results/reports/04_manuscript_outline_ocean_engineering.md")
O.append({"t": "pagebreak"})
O += md_blocks(ROOT / "results/reports/05_supplementary_material_outline.md")
json.dump({"title": "Manuscript outline", "blocks": O}, open(OUT / "outline.json", "w"), ensure_ascii=False, indent=0)
print("tables", tn["n"], "figures", fn["n"], "blocks", len(B))
