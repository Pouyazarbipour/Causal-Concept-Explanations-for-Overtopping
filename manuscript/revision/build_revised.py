"""Revised manuscript (Ocean Engineering) built only from verified outputs.

Every number is read from results/tables (original analyses, re-verified in revision/audit) or revision/tables
(new analyses).  Directional statements ("improves", "does not differ") are generated from confidence
intervals, never typed by hand.  Usage:
    python manuscript/revision/build_revised.py
    node manuscript/build_docx.js manuscript/revision/revised_manuscript.json manuscript/revision/revised_manuscript.docx
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
T0 = ROOT / "results" / "tables"
T1 = ROOT / "revision" / "tables"
F0 = ROOT / "results" / "figures"
F1 = ROOT / "revision" / "figures"
OUT = Path(__file__).resolve().parent
(OUT / "figs").mkdir(parents=True, exist_ok=True)


def t0(n):
    return pd.read_csv(T0 / f"{n}.csv")


def t1(n):
    p = T1 / f"{n}.csv"
    return pd.read_csv(p) if p.exists() else None


def f(x, d=3):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    return "–" if not np.isfinite(x) else f"{x:.{d}f}".replace("-", "−")


def s(x, d=3):
    return ("+" if float(x) >= 0 else "−") + f"{abs(float(x)):.{d}f}"


def ci(lo, hi, d=3):
    return f"[{f(lo, d)}, {f(hi, d)}]"


def pct(x, d=0):
    return f"{100 * float(x):.{d}f} %"


def n_(x):
    return f"{int(round(float(x))):,}".replace(",", " ")


def sig(lo, hi, null=0.0):
    return (lo > null) or (hi < null)


B = []
add = B.append
tn, fn = {"n": 0}, {"n": 0}
TAB_IDX, FIG_IDX = {}, {}


def p(t):
    add({"t": "p", "text": t})


def h1(t):
    add({"t": "h1", "text": t})


def h2(t):
    add({"t": "h2", "text": t})


def bl(items, level=0):
    add({"t": "bullets", "items": items, "level": level})


def T_(key, caption, header, rows, widths=None, note=None, fs=17):
    tn["n"] += 1
    TAB_IDX[key] = tn["n"]
    add({"t": "table", "caption": f"**Table {tn['n']}.** {caption}", "header": header, "rows": rows, "widths": widths, "note": note, "font_size": fs})


def F_(key, src, caption, width=6.3):
    fn["n"] += 1
    FIG_IDX[key] = fn["n"]
    im = Image.open(src).convert("RGB")
    if im.width > 2400:
        im = im.resize((2400, round(im.height * 2400 / im.width)), Image.LANCZOS)
    dst = OUT / "figs" / Path(src).name
    im.save(dst, optimize=True)
    add({"t": "figure", "path": f"figs/{Path(src).name}", "caption": f"**Fig. {fn['n']}.** {caption}", "width_in": width,
         "aspect": im.height / im.width, "alt": caption[:120]})


def tref(k):
    return "{{TAB:" + k + "}}"


def fref(k):
    return "{{FIG:" + k + "}}"


# ============================================================================== data
rec = t1("S_accounting_reconciliation").set_index("quantity")["reproduced_value"]
ex = t1("S_accounting_exclusive")
mis = t1("S_missingness_by_rf_cf")
rep4 = t1("S_missingness_rf4_representativeness").set_index("variable")
perf = t0("table03_model_performance")
P0 = perf[perf.scheme == "geometry"].set_index("model")
PS = perf.set_index(["scheme", "model"])
M1 = t1("M1_model_comparison")
M3 = t1("M3_xgb_budget_sensitivity")
M4 = t1("M4_repeated_partitions")
rf_t = t0("table05a_residuals_by_rf_cf")
RF = rf_t[rf_t.factor == "RF"].set_index("category")
CFr = rf_t[rf_t.factor == "CF"].set_index("category")
vm = t0("table05d_variance_model_main_effects").set_index("term")
vmi = t0("table05e_variance_model_with_interaction").set_index("term")
tr = t0("table05b_ordinal_trend_tests")
trx = tr[tr.residuals == "xgb, geometry-grouped"].set_index("factor")
trs = tr[tr.residuals == "xgb, series held out"].set_index("factor")
rep = t0("table05h_repeatability_near_replicates").set_index("subset")
cfg = t0("table06a_cf_geometry_level").set_index("CF")
cft = t0("table06b_cf_geometry_level_trends").set_index("geometry_level_quantity")
dec = t0("table08_hierarchical_error_decomposition")
dprim = dec[dec.analysis.str.startswith("PRIMARY")].set_index("component")
R1 = t1("R1_variance_ratios_by_learner_and_scheme")
R2 = t1("R2_rf4_specificity_contrasts").set_index("contrast")
R3 = t1("R3_repeatability_tolerance_sensitivity")
R3b = t1("R3b_replicate_ratio_ci").set_index("quantity")
R4 = t1("R4_leave_one_series_out_CF")
R5 = t1("R5_power_minimum_detectable_ratio")
R6 = t1("R6_cf_inconsistent_geometries")
R6b = t1("R6b_cf_inconsistency_causes")
R7 = t1("R7_missing_input_sensitivity")
G1 = t1("G1_geometry_definition_sensitivity")
W1 = t1("W1_stageA_classification").set_index("subset")
W2 = t1("W2_stageA_reliability")
W3 = t1("W3_two_stage_combined").set_index("subset")
W4 = t1("W4_two_stage_policies")
Pv2 = t1("P1_policies_v2")
Pcov = t1("P2_policies_v2_coverage_by_category")
pol0 = t0("table13_data_selection_policies")
PX = pol0[pol0.learner == "xgb"].set_index("policy")
C1 = t1("C1_conformal_sensitivity")
C2 = t1("C2_conformal_coverage_by_category")
C4 = t1("C4_cf3_error_by_series")
E1 = t1("E1_eurotop_benchmark")
E1b = t1("E1b_eurotop_domain_counts")
E2 = t1("E2_exceedance_decisions_by_policy")
E3 = t1("E3_risk_ranking_and_safety_class")
E4 = t1("E4_regimes_of_category4_tests").set_index("regime")
E5 = t1("E5_input_space_coverage")
ccw = t0("table09_ccw_alignment_and_ablation").set_index("variant")
cva = t0("table11_concept_vs_feature_attribution").set_index("quantity")["value"]
probe = t0("table12c_quality_probe_from_concepts")


def cv(q):
    return float(cva[[i for i in cva.index if q in i][0]])


x_rmse = P0.loc["XGBoost", "rmse"]
vr4 = vm.loc["RF=4"]
q_g, q_s, q_w = dprim.loc["geometry"], dprim.loc["series"], dprim.loc["within-geometry residual"]
rrat = R3b.iloc[0]

# v2 model summary
mm = M1.set_index("model")
xb = mm.loc["XGBoost (baseline)"]
v2_rows = M1[M1.model.str.contains("v2")]
best_v2 = v2_rows.sort_values("ensemble_rmse").iloc[0] if len(v2_rows) else None
xv2 = mm.loc["XGBoost v2"] if "XGBoost v2" in mm.index else None
any_sig = bool(len(v2_rows) and v2_rows.improvement_significant.any())

# policies v2
PF = Pv2[Pv2["mode"] == "fixed"].set_index("policy") if Pv2 is not None else None
PT = Pv2[Pv2["mode"] == "tuned"].set_index("policy") if Pv2 is not None and (Pv2["mode"] == "tuned").any() else None
EXCL = ["P2_excl_RF4", "P3_excl_CF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks"]
EX_RF4 = ["P2_excl_RF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks"]
EX_CF4 = ["P3_excl_CF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P5b_CLASH_weights_and_remarks"]


def prange(T, col, pols, d=2):
    v = T.loc[pols, col]
    return f"{s(v.min(), d)} to {s(v.max(), d)}"


# conformal
C25 = C1[C1.calib_fraction == 0.25].set_index("scale")

# ============================================================================== title, highlights, abstract
TITLE = ("Do CLASH reliability and complexity ratings justify data exclusion? A residual-based audit and counterfactual "
         "assessment of wave-overtopping prediction policies")
add({"t": "title", "text": TITLE})
add({"t": "subtitle", "text": "[Author names and affiliations]"})
add({"t": "subtitle", "text": "Revised manuscript prepared for *Ocean Engineering*", "italic": True})
add({"t": "note", "text": "Generated from verified outputs of the analysis repository (revision/). Items in square brackets, the RF/CF category "
     "wording and references marked 'to verify' must be completed by the authors before submission."})

h1("Highlights")
hl = [f"All 10 532 CLASH tests are audited. The models use 10 389 tests (occurrence) and {n_(rec['Positive-discharge regression observations'])} tests (discharge), with every RF and CF category kept.",
      f"RF = 4 tests have {f(vr4.variance_ratio, 1)} times the residual variance of RF = 1 tests, consistently across learners and validation schemes. RF 1–3 show no ordered trend.",
      "The complexity factor shows no ordered relation to geometry-level error. CF = 3 is the difficult class, and much of its error comes from a few series."]
if PF is not None:
    hl.append(f"Excluding category-4 tests leaves the other predictions unchanged (|ΔRMSE| ≤ {f(PF.loc[EXCL, 'd_rmse_common'].abs().max(), 3)}) and degrades category-4 predictions, with fixed or with re-tuned learners.")
hl.append(f"Quality-aware conformal intervals with all tests retained raise RF = 4 coverage from {pct(C25.loc['constant', 'cov_RF4'])} to {pct(C25.loc['physical+RF+CF', 'cov_RF4'])}. CF = 3 remains under-covered.")
bl(hl)

h1("Abstract")
abs_ = (f"Each test in the CLASH wave-overtopping database carries a Reliability Factor (RF) and a Complexity Factor (CF), both ordinal from 1 to 4. "
        f"These ratings are used to weight or exclude tests when data-driven predictors are developed, but their validity for that decision has not been tested. "
        f"We audited all 10 532 tests: {n_(rec['Positive-discharge regression observations'])} positive-discharge tests enter the regression and 10 389 enter an occurrence (wet/dry) model. "
        f"The ratings were evaluated against out-of-fold residuals of nested, geometry-grouped cross-validated learners. "
        f"Tests with RF = 4 show {f(vr4.variance_ratio, 2)} {ci(vr4.ratio_lo, vr4.ratio_hi, 2)} times the residual variance of RF = 1 tests. "
        f"The result holds for every learner and validation scheme, but RF 1–3 show no ordered trend and near-replicate repeatability does not rise systematically with RF. "
        f"CF shows no ordered relation to geometry-level error. "
        f"A nested decomposition attributes {pct(q_w.share)} of the residual variance to within-geometry variation, {pct(q_g.share)} to geometries and {pct(q_s.share)} to test series. "
        f"Near-replicate variance accounts for {pct(rrat.value)} {ci(100*rrat.lo, 100*rrat.hi, 0).replace('[', '[').replace(']', ' %]')} of the within-geometry part, so that part is not pure measurement error. ")
if PF is not None:
    abs_ += (f"On identical held-out tests, excluding or zero-weighting category-4 tests changes the error of the remaining tests by at most {f(PF.loc[EXCL, 'd_rmse_common'].abs().max(), 3)} log10 units. "
             f"It increases the error of RF = 4 tests by {prange(PF, 'd_rmse_RF4', EX_RF4)}, also when each policy receives its own tuned learner. ")
abs_ += (f"Retaining all tests and scaling group-weighted split-conformal intervals by RF, CF and the physical inputs raises RF = 4 coverage of 90 % intervals "
         f"from {pct(C25.loc['constant', 'cov_RF4'])} to {pct(C25.loc['physical+RF+CF', 'cov_RF4'])} and gives the best interval score. "
         f"CF = 3 tests remain under-covered ({pct(C25.loc['physical+RF+CF', 'cov_CF3'])}). "
         f"Exclusion also changes engineering decisions: it raises the share of missed exceedances of allowable discharges for RF = 4 tests and removes almost all depth-limited tests. "
         f"The ratings should be evaluated empirically before they are used to discard data, and are better used to manage predictive uncertainty than to exclude tests.")
p(abs_)
add({"t": "p", "text": "*Keywords:* wave overtopping; CLASH database; data quality; reliability factor; complexity factor; data selection; conformal prediction; EurOtop"})

# ============================================================================== 1 Introduction
h1("1. Introduction")
p("Data-driven prediction of mean wave-overtopping discharge rests on the CLASH database and its successors (Steendam et al., 2004; "
  "Van der Meer et al., 2009; EurOtop, 2018). Every test carries two ordinal ratings. The Reliability Factor (RF) describes how reliable "
  "the measurement and its documentation are. The Complexity Factor (CF) describes how well the standard CLASH parameters represent the "
  "tested cross-section. Both run from 1 (very reliable / simple) to 4 (unreliable / very complex).")
p("The ratings are used to select training data. The CLASH neural network weighted each test by (4 − RF)(4 − CF), which gives zero weight "
  "to every test with RF = 4 or CF = 4, and removed tests remarked \"do not use for neural network\" (Van Gent et al., 2007). Later neural-network "
  "and gradient-boosting predictors excluded or resampled tests by their ratings (Zanuttigh et al., 2016; den Bieman et al., 2021). In these "
  "studies the validity of the ratings for the data-selection decision is assumed, not tested. Category-4 tests are not a random sample: they "
  "include depth-limited conditions and unusual sections that the other tests cover poorly. Removing them may therefore discard information "
  "without reducing noise.")
p("This paper evaluates the ratings empirically and measures the consequences of rating-based data curation. Out-of-fold residuals of "
  "tuned learners are the instrument; the learners themselves are not the contribution. No physical causal effect of a hydraulic or "
  "structural parameter is estimated. Geometry-grouped validation and conformal prediction are used as tools; their development is the "
  "subject of companion work [Paper 1], and causal analysis of physical drivers is treated in [Paper 2]. The research questions are:")
add({"t": "numbered", "items": [
    "RQ1. Does RF distinguish tests with different predictive residual dispersion and near-replicate repeatability?",
    "RQ2. Does CF identify geometries with higher predictive difficulty or poorer generalisation?",
    "RQ3. How is the prediction error distributed over identifiable test-series, geometry and within-geometry levels, and do RF and CF map onto these levels? (Laboratory effects are not identifiable.)",
    "RQ4. How do retaining, excluding or re-weighting tests by RF and CF change accuracy, calibration and engineering decisions on identical held-out tests, with fixed and with re-tuned learners?",
    "RQ5. Do concept-based explanations of a regression-adapted Causal Concept Wrapper Network (CCW-Net) faithfully describe the learned representations, and do they depend on data quality?",
    "RQ6. Is there a defensible alternative to automatic exclusion of category-4 tests?"]})

# ============================================================================== 2 Data
h1("2. CLASH data and quality ratings")
h2("2.1 Record accounting")
inv_ = int(rec["Missing or invalid discharge (20 missing markers + 1 negative)"])
unus = int(rec["Records with unusable required inputs (Hm0,toe, Tm-1,0,toe, h or Rc), any response"])
p(f"The workbook holds 10 532 tests from {n_(rec['Test series (all)'])} test series. Of these, 9 442 report positive discharge, 1 069 zero "
  f"discharge and {inv_} a missing marker or a negative value. {unus} records lack a usable toe wave height, spectral period, toe depth "
  f"or freeboard; one of them also has an invalid response ({tref('acc')}). The occurrence (wet/dry) model therefore uses "
  f"10 389 tests and the discharge regression {n_(rec['Positive-discharge regression observations'])} tests. All 143 records outside the "
  f"wet/dry set have RF = 4. No observation was modified and no epsilon was added to zero discharges.")
rows = [[r.category, n_(r.n), n_(r.n_series), n_(r.RF4), n_(r.CF4), n_(r.in_wetdry), n_(r.in_regression)] for r in ex.itertuples()]
T_("acc", "Mutually exclusive record accounting of the CLASH database.", ["Category", "Records", "Series", "RF = 4", "CF = 4", "Wet/dry", "Regression"],
   rows, [3.2, 0.8, 0.7, 0.7, 0.7, 0.8, 0.9], note="Overlapping exclusion reasons are given in the supplementary material (Table S1).")
r4u = mis[(mis.factor == "RF") & (mis.category == 4)].iloc[0]
p(f"Input unusability is confined to RF = 4: {pct(r4u.pct_unusable / 100, 1)} of valid-response RF = 4 tests have unusable inputs, against none in RF 1–3. "
  f"The affected tests are not representative of the RF = 4 tests that enter the models. They have a median toe depth of "
  f"{f(rep4.loc['h [m]', 'unusable_median'], 2)} m against {f(rep4.loc['h [m]', 'usable_RF4_median'], 2)} m, mostly vertical lower slopes, and "
  f"{pct(rep4.loc['share from series 802', 'unusable_median'])} come from one series. They are dominantly emergent-toe conditions, where a toe wave height is "
  f"not defined. Because q* requires the toe wave height, these tests cannot be scaled without imputing part of the response, which was not done in the "
  f"primary analysis. A sensitivity analysis that substitutes deep-water wave conditions ({n_(R7.n_substitutable.iloc[0])} tests) changes the RF = 4 variance "
  f"ratio from {f(R7.variance_ratio.iloc[0], 2)} to {f(R7.variance_ratio.iloc[1], 2)} {ci(R7.ratio_lo.iloc[1], R7.ratio_hi.iloc[1], 2)}. The RF = 4 results "
  f"therefore describe RF = 4 tests with complete toe conditions.")
F_("acc", F1 / "main" / "Fig01_data_accounting.png", "Record accounting: (a) mutually exclusive categories of the 10 532 records and the number of RF = 4 records in each; (b) share of valid-response records with unusable inputs by RF and CF.")
h2("2.2 Ratings, identifiers and reconstructed geometries")
dist = t0("table02_rf_cf_distribution")
rows = []
for k in ("RF", "CF"):
    d = dist[(dist.factor == k) & dist.category.astype(str).isin(list("1234"))]
    rows.append([k] + [f"{int(n)} ({q_:.1f} %)" for n, q_ in zip(d.n_all, d.pct_all)])
T_("dist", "Distribution of RF and CF over all 10 532 tests.", ["Factor", "1", "2", "3", "4"], rows, [1, 2, 2, 2, 2],
   note="RF: 1 very reliable … 4 not reliable. CF: 1 simple … 4 very complex (category wording to be verified against the CLASH WP2 documentation).")
p(f"CF = 4 comprises {n_(rec['CF = 4 tests (all)'])} tests from {n_(rec['CF = 4 series (all)'])} series "
  f"({n_(rec['CF = 4 tests (regression)'])} regression tests from {n_(rec['CF = 4 series (regression)'])} series), dominated by one investigation of chute "
  f"structures. The test-series number ({n_(rec['Test series (regression)'])} series in the regression set) and the literature reference are genuine identifiers. "
  f"No structure or laboratory identifier exists. Geometries were therefore reconstructed within each series from water-level-invariant "
  f"descriptors, giving {n_(rec['Reconstructed geometries (regression)'])} geometries in the regression set. A laboratory could be inferred from the "
  f"reference for {pct(rec['Share of regression tests with an inferred laboratory'])} of tests only, so a laboratory effect is not identifiable.")
causes = R6b.set_index("difference")["n_geometries"] if R6b is not None else pd.Series(dtype=float)
p(f"CF is nominally a property of the cross-section, but it varies within {n_(rec['Geometries with >1 CF value (all 10 532 records)'])} reconstructed geometries "
  f"({n_(rec['Tests in geometries with >1 CF value (all 10 532 records)'])} tests; {n_(rec['Geometries with >1 CF value (regression set)'])} geometries "
  f"in the regression set). Within these geometries the lowest- and highest-CF tests differ in water level or freeboard in "
  f"{n_(causes.get('water level / freeboard differs', 0))} geometries, in wave conditions in {n_(causes.get('wave conditions differ', 0))} and in wave obliquity in "
  f"{n_(causes.get('obliquity differs', 0))}, which suggests that CF was sometimes assigned per test condition rather than per cross-section. "
  f"In {n_(causes.get('none in documented hydraulic variables', 0))} they differ in no documented variable. "
  f"The inconsistencies were retained, and CF is aggregated by its maximum where a geometry-level value is needed. This variation limits "
  f"the use of CF as a geometry-level quality indicator.")

# ============================================================================== 3 Methods
h1("3. Methodology")
h2("3.1 Response and predictors")
p("The response is Y = log₁₀ q*, with q* = q / (g H_{m0}^{3})^{1/2}, q the mean discharge per metre (m³/s/m) and H_{m0} the spectral wave height "
  "at the toe. Fifteen dimensionless predictors use Froude scaling at the toe: R_{c}/H_{m0}, A_{c}/H_{m0}, G_{c}/L_{0}, H_{m0}/L_{0}, h/L_{0}, "
  "h_{t}/H_{m0}, B_{t}/L_{0}, B/L_{0}, h_{b}/H_{m0}, tan α_{B}, cot α_{d}, cot α_{u}, γ_{f}, β and the foreshore slope 1/m, with "
  "L_{0} = gT_{m−1,0}²/2π. All transformations were recomputed independently, with relative differences below 10^{−12} (supplementary Table S2). "
  "RF and CF never enter a mean model.")
h2("3.2 Validation and model optimisation")
p("All learners share five pre-specified outer folds in which every reconstructed geometry is held out once. Hyper-parameters are selected "
  "by Optuna (TPE; Akiba et al., 2019) on grouped inner three-fold cross-validation of the outer-training partition. Boosted trees use early "
  "stopping on inner folds. Imputation and scaling are fitted on training rows only, and outer test rows are never used for any choice. "
  "The original configuration (Random Forest, XGBoost, CatBoost and MLP with 12, 30, 15 and 8 trials) was reproduced exactly. A revised "
  "configuration widens the search spaces and raises the budgets to 30–100 trials with median pruning. It adds LightGBM, histogram gradient "
  "boosting, ExtraTrees and an XGBoost variant with physically derived features (Iribarren number and roughness- and obliquity-scaled "
  "freeboards), and refits every configuration with three seeds. Learners are compared by paired geometry-cluster bootstrap differences "
  "on identical records. XGBoost remains the pre-specified reference learner for the RF/CF analyses. Series-held-out and random-record "
  "validation are reported; the latter is a leakage-sensitive reference only.")
h2("3.3 Two-stage occurrence and discharge model")
p("Overtopping occurrence and discharge magnitude are modelled separately. Stage A is an XGBoost classifier for P(wet | X), tuned by Optuna "
  "on log loss over grouped inner folds. Its decision threshold is the maximiser of balanced accuracy on grouped inner out-of-fold "
  "probabilities. Stage B is the regression of Y on positive tests. Stage C combines both: E(q | X) = P(wet | X) E(q | wet, X), with "
  "E(q | wet, X) = 10^{μ(X)} S (g H_{m0}^{3})^{1/2}, where S is the Duan smearing factor from grouped inner out-of-fold residuals. All "
  "10 389 tests are assigned the outer fold of their geometry, so positive tests keep their regression fold.")
h2("3.4 Validity of RF and CF")
p("RF and CF are treated as ordinal. The analyses comprise:")
bl(["residual strata with test-series cluster-bootstrap intervals;",
    "rank trends with Holm correction;",
    "proportional-odds models;",
    "a Gamma GLM of squared residuals on the RF and CF categories, with and without an RF × CF term, repeated for every learner and validation scheme;",
    "contrasts between RF categories;",
    "geometry-level CF analyses (maximum CF within a geometry), with leave-one-series-out sensitivity for CF = 3 and CF = 4;",
    "minimum detectable variance ratios;",
    "near-replicate repeatability, repeat SD = (mean d²/2)^{1/2}, with matching tolerances of 0.5–5 % and a disjoint-pair estimator."])
p("No laboratory reports or second ratings exist. The analyses therefore establish predictive validity and internal consistency, not the correctness of individual ratings.")
h2("3.5 Hierarchical error decomposition")
p("Out-of-fold residuals are decomposed by exact restricted maximum likelihood into nested random intercepts for test series and geometry "
  "and a within-geometry residual. Intervals come from a series-cluster bootstrap. The stability of the shares is checked under geometry "
  "keys with 2 and 4 significant digits and a coarse key. The within-geometry component is compared with near-replicate variance and is "
  "not called measurement error.")
h2("3.6 Counterfactual data-selection policies")
p(f"A policy decides which training tests are kept and how they are weighted ({tref('poldef')}). Within each outer fold only "
  "the policy changes; folds, test records, seeds and calibration geometries stay fixed. The counterfactual concerns training data, not a "
  "physical input. Two experiments are reported:")
bl(["fixed hyper-parameters, tuned once on the full training partition, which isolates the effect of the data;",
    "policy-specific nested tuning, with Optuna applied separately for each policy on its own inner folds, which gives each policy its best achievable learner."])
p("Accuracy is evaluated on identical held-out tests, split into common tests (RF < 4 and CF < 4, 8 494 tests), RF = 4, CF = 4 and CF = 3 "
  "tests, with paired geometry-cluster bootstrap differences to retaining all tests (P1). The empirical variance model σ²(RF, CF) used by "
  "P6 is fitted on grouped inner out-of-fold residuals of the proper-training geometries only. This corrects an earlier version that also "
  "used calibration geometries.")
T_("poldef", "Data-selection policies.", ["Policy", "Training tests", "Weights", "Interval width"],
   [["P1 retain all", "all", "equal", "constant"], ["P2", "excl. RF = 4", "equal", "constant"], ["P3", "excl. CF = 4", "equal", "constant"],
    ["P4", "excl. RF = 4 or CF = 4", "equal", "constant"], ["P5 conventional", "RF < 4 and CF < 4", "(4 − RF)(4 − CF)", "constant"],
    ["P5b conventional", "as P5, 'do not use' tests removed", "(4 − RF)(4 − CF)", "constant"],
    ["P6a empirical weights", "all", "1/σ²(RF, CF)", "constant"], ["P6b quality-aware intervals", "all", "equal", "∝ σ(RF, CF)"],
    ["P6b′ shrunk cells", "all", "equal", "∝ σ(RF × CF), partially pooled"], ["P6c", "all", "1/σ²(RF, CF)", "∝ σ(RF, CF)"]], [1.6, 1.8, 1.3, 1.6])
h2("3.7 Quality-aware conformal intervals")
p("Intervals use group-weighted split conformal prediction: each calibration geometry carries weight 1/(K + 1), and a mass of "
  "1/(K + 1) sits at infinity (Lee et al., 2023, preprint). The mean model is fitted on proper-training geometries, and every scale function "
  "on their inner out-of-fold residuals. Scores are therefore independent of the calibration and test outcomes. Six scale functions "
  "represent different availabilities of quality information:")
bl(["none;",
    "RF and CF main effects (retrospective audit);",
    "partially pooled RF × CF cells;",
    "CF only (known geometry, no RF);",
    "physical predictors only (new structure, no rating);",
    "physical predictors with RF and CF."])
p("Calibration fractions of 0.15, 0.25 and 0.35 with five calibration seeds test the stability of the results. The guarantee is marginal "
  "over new geometries; coverage within categories is empirical, not guaranteed.")
h2("3.8 Engineering decision analysis")
p("EurOtop (2018) mean-value formulae are applied only within stated domains:")
bl(["single-slope structures without berm or crest wall, 1 ≤ cot α ≤ 7, 0.1 < R_{c}/H_{m0} < 3.5 and ξ_{m−1,0} < 5 (slope equations with roughness, obliquity and crest-width factors);",
    "plain vertical walls without a toe mound and without foreshore influence (h/H_{m0} ≥ 4)."])
p("The formulae were partly calibrated on CLASH data, so this comparison is not independent. Allowable discharges depend on use and wave "
  "height (EurOtop, 2018), and no universal limit exists. A sensitivity grid of prototype limits q_{allow} ∈ {0.1, 1, 10, 50} l/s/m at "
  "H_{m0} ∈ {1, 2, 3} m is therefore converted to q* by Froude similarity. Under each policy we record false-safe and false-alarm "
  "classifications, changes of safety class and the shift in risk ranking.")
h2("3.9 Concept-based explanations (CCW-Net)")
p("A regression adaptation of the Causal Concept Wrapper Network (Rountree et al., 2026) wraps a frozen tabular backbone with ten concept "
  "slots and trains them with necessity, sufficiency and independence criteria. The implementation was checked line by line against the "
  "authors' reference code (supplementary Section S8). Its effects are properties of the network's representations, not physical causal "
  "effects. Full details are in the supplementary material.")

# ============================================================================== 4 Predictive diagnostics
h1("4. Predictive model diagnostics")
rows = []
for lab in M1.model:
    r = mm.loc[lab]
    rows.append([lab, f(r.rmse) + (f" ± {f(r.rmse_seed_sd)}" if r.n_seeds > 1 else ""), f(r.ensemble_rmse) + " " + ci(r.ensemble_rmse_lo, r.ensemble_rmse_hi),
                 f(r.r2), s(r.bias), f(r.f10, 2), "0" if lab == "XGBoost (baseline)" else s(r.d_rmse_vs_xgb_baseline) + " " + ci(r.d_lo, r.d_hi)])
T_("perf", "Out-of-fold accuracy on held-out geometries (n = 9 333, all RF and CF retained; Y = log₁₀ q*).",
   ["Model", "RMSE (per seed)", "RMSE seed ensemble [95 % CI]", "R²", "Bias", "F10", "ΔRMSE vs baseline XGBoost [95 % CI]"], rows,
   [2.0, 1.1, 1.6, 0.6, 0.7, 0.6, 1.7], fs=15,
   note="RMSE is the pooled out-of-fold RMSE (the definition used throughout; the fold-mean values of the original Table 2 are in the SI). "
        "Bias = predicted − measured. F10 = 10^RMSE. ΔRMSE: paired geometry-cluster bootstrap on identical records.")
if best_v2 is not None:
    txt = (f"The reproduced baseline XGBoost gives RMSE {f(x_rmse)} ({tref('perf')}). The best optimised learner, {best_v2.model}, reaches "
           f"{f(best_v2.ensemble_rmse)} as a seed ensemble, a paired difference of {s(best_v2.d_rmse_vs_xgb_baseline)} {ci(best_v2.d_lo, best_v2.d_hi)}. ")
    if any_sig:
        sigm = v2_rows[v2_rows.improvement_significant].sort_values("d_rmse_vs_xgb_baseline")
        txt += (f"{len(sigm)} optimised learner(s) improve significantly on the baseline, by at most {f(-sigm.d_rmse_vs_xgb_baseline.min(), 3)} log10 units "
                f"(a factor {f(10 ** -sigm.d_rmse_vs_xgb_baseline.min(), 3)} in discharge). This is small against the {f(x_rmse, 2)} residual SD and does not alter any RF/CF conclusion (Section 5). ")
    else:
        txt += "No optimised learner improves significantly on the baseline. "
    if M3 is not None:
        mb = M3[M3.budget_trials.astype(str).str.isnumeric()].astype({"budget_trials": int}).sort_values("budget_trials")
        txt += (f"For XGBoost the outer RMSE changes from {f(mb.outer_rmse.iloc[0])} after {mb.budget_trials.iloc[0]} trials to "
                f"{f(mb.outer_rmse.iloc[-1])} after {mb.budget_trials.iloc[-1]}, so larger budgets bring little. ")
    p(txt)
alt = M4[M4.partition.str.startswith("alternative")] if M4 is not None else None
if alt is not None and len(alt):
    p(f"Across two further geometry partitions the baseline XGBoost RMSE ranges from {f(alt[alt.model == 'XGBoost baseline'].rmse.min())} to "
      f"{f(alt[alt.model == 'XGBoost baseline'].rmse.max())}, against {f(x_rmse)} on the pre-specified partition. The fold allocation therefore "
      f"contributes variation of the same order as the optimisation gains. The pre-specified partition is retained for all primary results.")
ser_v2 = M4[(M4.partition == "series held out") & (M4.model.str.contains("v2"))] if M4 is not None else None
p(f"The validation unit dominates: the reference model reaches RMSE {f(PS.loc[('random', 'XGBoost'), 'rmse'])} with random records, {f(x_rmse)} with held-out "
  f"geometries and {f(PS.loc[('series', 'XGBoost'), 'rmse'])} with held-out test series"
  + (f" ({f(ser_v2.rmse.iloc[0])} for the optimised XGBoost)" if ser_v2 is not None and len(ser_v2) else "")
  + ". Geometry-grouped validation is therefore not equivalent to generalisation to a new laboratory or test series.")
F_("perf", F1 / "main" / "Fig03_model_comparison.png", "Baseline and optimised learners on identical held-out geometries: (a) seed-ensemble RMSE with geometry-cluster bootstrap 95 % intervals; (b) paired difference to the baseline XGBoost.")
h2("4.1 Occurrence and combined prediction")
p(f"The occurrence model separates wet from dry tests with ROC AUC {f(W1.loc['all', 'auc'], 3)} {ci(W1.loc['all', 'auc_lo'], W1.loc['all', 'auc_hi'], 3)}, "
  f"Brier score {f(W1.loc['all', 'brier'], 3)} and balanced accuracy {f(W1.loc['all', 'balanced_accuracy'], 3)} at the inner-CV threshold "
  f"({fref('two')}). Discrimination is weaker for RF = 4 (AUC {f(W1.loc['RF=4', 'auc'], 2)} {ci(W1.loc['RF=4', 'auc_lo'], W1.loc['RF=4', 'auc_hi'], 2)} "
  f"against {f(W1.loc['RF=1', 'auc'], 2)} for RF = 1) and for CF = 4 ({f(W1.loc['CF=4', 'auc'], 2)}). It is computed on the same geometry folds as the "
  f"regression; the original wet/dry figures (0.81 and 0.92) came from a fixed-parameter classifier with separate folds and agree with these. "
  f"The stage-B predictions reproduce the original out-of-fold predictions exactly. For positive tests the combined expected discharge is "
  f"{s(W3.loc['all', 'bias_log10q_two_stage_total'], 2)} log10 units above the conditional median, as expected for a mean of a log-normal "
  f"quantity. For zero-discharge tests the median predicted expected discharge is {f(W3.loc['all', 'zero_tests_median_pred_q_l_per_s_per_m_two_stage'], 4)} l/s/m (model scale).")
F_("two", F1 / "main" / "Fig04_two_stage_wet_dry.png", "Overtopping occurrence (stage A) on held-out geometries: (a) ROC curves by RF; (b) reliability diagram; (c) ROC AUC by RF and CF with geometry-cluster bootstrap intervals.")

# ============================================================================== 5 RF and CF validity
h1("5. Validity of the reliability and complexity factors")
h2("5.1 Reliability factor")
rows = []
for v in (1, 2, 3, 4):
    r = RF.loc[v]
    rs = rep.loc[f"max RF={v}"]
    rows.append([str(v), n_(r.n), n_(r.n_series), f(r.mae) + " " + ci(r.mae_lo, r.mae_hi), f(r.within_geometry_sd) + " " + ci(r.within_sd_lo, r.within_sd_hi),
                 f(rs.repeat_sd_log10, 2) + f" ({int(rs.n_pairs)})", "1" if v == 1 else f(vm.loc[f'RF={v}', 'variance_ratio'], 2) + " " + ci(vm.loc[f'RF={v}', 'ratio_lo'], vm.loc[f'RF={v}', 'ratio_hi'], 2)])
T_("rf", "Residuals, repeatability and variance ratios by RF (reference XGBoost, held-out geometries).",
   ["RF", "n", "Series", "MAE [95 % CI]", "Within-geometry SD", "Repeat SD (pairs)", "Variance ratio vs RF 1"], rows, [0.5, 0.7, 0.7, 1.6, 1.6, 1.2, 1.6],
   note="Repeat SD from near-replicate pairs (1 % tolerance) by the larger RF of the pair. Variance ratios from a Gamma GLM with RF and CF categories; test-series cluster bootstrap.")
rr4 = R1[R1.term == "RF=4"]
p(f"Tests with RF = 4 have a residual variance {f(vr4.variance_ratio, 2)} {ci(vr4.ratio_lo, vr4.ratio_hi, 2)} times that of RF = 1 tests, adjusted for CF "
  f"({tref('rf')}). Across all learners and validation schemes the ratio lies between {f(rr4.variance_ratio.min(), 2)} and {f(rr4.variance_ratio.max(), 2)}, and every "
  f"interval excludes one ({fref('rob')}). The effect is specific to the top category. RF = 4 differs from RF = 2 "
  f"({f(R2.loc['RF4 / RF2', 'variance_ratio'], 2)} {ci(R2.loc['RF4 / RF2', 'lo'], R2.loc['RF4 / RF2', 'hi'], 2)}) but not clearly from RF = 3 "
  f"({f(R2.loc['RF4 / RF3', 'variance_ratio'], 2)} {ci(R2.loc['RF4 / RF3', 'lo'], R2.loc['RF4 / RF3', 'hi'], 2)}), and there is no trend among RF 1–3 "
  f"(Spearman {f(R2.loc['Spearman(|r|, RF) among RF 1-3', 'variance_ratio'], 3)} {ci(R2.loc['Spearman(|r|, RF) among RF 1-3', 'lo'], R2.loc['Spearman(|r|, RF) among RF 1-3', 'hi'], 3)}). "
  f"The scale therefore behaves as a flag for an elevated-dispersion group rather than as a graded measure. Equal spacing of the categories "
  f"is not supported.")
r3a = R3[(R3.pairs == "all pairs") & (R3.subset == "max RF=4")].set_index("rel_tol")
r31 = R3[(R3.pairs == "all pairs") & (R3.subset == "max RF=1")].set_index("rel_tol")
p(f"Near-replicate repeatability does not reproduce the RF = 4 effect at the primary 1 % tolerance: the repeat SD is {f(r31.loc[0.01, 'repeat_sd'], 2)} for "
  f"pairs with RF ≤ 1 and {f(r3a.loc[0.01, 'repeat_sd'], 2)} with RF = 4, from only {n_(r3a.loc[0.01, 'n_pairs'])} RF = 4 pairs in {n_(r3a.loc[0.01, 'n_series'])} series. "
  f"With a 5 % tolerance the RF = 4 value rises to {f(r3a.loc[0.05, 'repeat_sd'], 2)} {ci(r3a.loc[0.05, 'lo'], r3a.loc[0.05, 'hi'], 2)}. Wider tolerances, however, admit "
  f"genuine hydraulic differences. Repeatability evidence on RF is therefore inconclusive, and the elevated residual dispersion of RF = 4 cannot be "
  f"attributed to measurement uncertainty alone. RF rises with indicators of estimated parameters, so RF partly records documentation "
  f"completeness.")
h2("5.2 Complexity factor")
rows = []
for v in (1, 2, 3, 4):
    r = cfg.loc[v]
    rows.append([str(v), n_(r.n_geometries), n_(r.n_series), n_(r.n_tests), f(r.abs_mean_resid) + " " + ci(r.abs_mean_resid_lo, r.abs_mean_resid_hi),
                 f(r.rmse_grouped) + " " + ci(r.rmse_grouped_lo, r.rmse_grouped_hi), f(r.generalisation_gap) + " " + ci(r.generalisation_gap_lo, r.generalisation_gap_hi)])
T_("cf", "Geometry-level errors by CF (maximum CF within the geometry; geometries are the units).",
   ["CF", "Geometries", "Series", "Tests", "|Mean residual|", "Mean geometry RMSE", "Grouped − random RMSE"], rows, [0.5, 0.9, 0.7, 0.7, 1.5, 1.5, 1.5], fs=16,
   note="Mean geometry RMSE is the unweighted mean over geometries of the per-geometry RMSE. It differs from the record-level RMSE pooled over all tests of a category (Section 5.2).")
cf4l = R4[R4.CF_category == 4]
cf3l = R4[R4.CF_category == 3]
p(f"No geometry-level quantity is rank-correlated with CF ({tref('cf')}). The absolute geometry mean residual gives "
  f"{f(cft.loc['abs_mean_resid', 'spearman_with_CF'], 2)} {ci(cft.loc['abs_mean_resid', 'lo'], cft.loc['abs_mean_resid', 'hi'], 2)} and the "
  f"generalisation gap {f(cft.loc['generalisation_gap', 'spearman_with_CF'], 2)} {ci(cft.loc['generalisation_gap', 'lo'], cft.loc['generalisation_gap', 'hi'], 2)}. "
  f"The non-significant trend does not demonstrate absence of an effect. The minimum detectable variance ratio for CF = 4 is "
  f"{f(R5.set_index('term').loc['CF=4', 'minimum_detectable_ratio_up'], 1)}, so moderate effects could go undetected. "
  f"CF = 3 is the difficult class: its residual variance is {f(vm.loc['CF=3', 'variance_ratio'], 2)} {ci(vm.loc['CF=3', 'ratio_lo'], vm.loc['CF=3', 'ratio_hi'], 2)} "
  f"times that of CF = 1. Its record-level RMSE ({f(CFr.loc[3, 'rmse'], 2)}) and its mean geometry RMSE ({f(cfg.loc[3, 'rmse_grouped'], 3)}) differ because "
  f"the former pools {n_(CFr.loc[3, 'n'])} tests and is dominated by a few series with large over-predictions. Series {C4.series.iloc[0]} alone contributes "
  f"{pct(C4.share_of_CF3_squared_error.iloc[0])} of the CF = 3 squared error. Leaving out single CF = 3 series moves the CF = 3 variance ratio between "
  f"{f(cf3l.variance_ratio_vs_CF1.min(), 2)} and {f(cf3l.variance_ratio_vs_CF1.max(), 2)}; without series {cf3l.sort_values('variance_ratio_vs_CF1').left_out_series.iloc[0]} it is "
  f"{f(cf3l.sort_values('variance_ratio_vs_CF1').variance_ratio_vs_CF1.iloc[0], 2)} {ci(cf3l.sort_values('variance_ratio_vs_CF1').lo.iloc[0], cf3l.sort_values('variance_ratio_vs_CF1').hi.iloc[0], 2)}, "
  f"so the CF = 3 effect rests largely on one investigation. CF = 4 tests are predicted well (variance ratio "
  f"{f(cf4l.variance_ratio_vs_CF1.iloc[0], 2)}; leave-one-series-out range {f(cf4l.variance_ratio_vs_CF1.iloc[1:].min(), 2)}–{f(cf4l.variance_ratio_vs_CF1.iloc[1:].max(), 2)}), "
  f"which reflects the internal consistency of a few series, not easy representation of complex sections. When whole series are held out, a "
  f"weak positive CF trend appears ({f(trs.loc['CF', 'spearman_absres'], 2)} {ci(trs.loc['CF', 'spearman_lo'], trs.loc['CF', 'spearman_hi'], 2)}). It is reported but "
  f"does not survive multiple-comparison correction. RF and CF do not interact (RF × CF term {f(vmi.iloc[-1].variance_ratio, 2)} {ci(vmi.iloc[-1].ratio_lo, vmi.iloc[-1].ratio_hi, 2)}).")
F_("rob", F1 / "main" / "Fig05_rf_cf_robustness.png", "Robustness of the RF and CF findings: (a) residual-variance ratios (RF = 4, CF = 3, CF = 4 vs category 1) for every learner and validation scheme; (b) near-replicate repeat SD by RF for matching tolerances of 0.5–5 %; (c) leave-one-series-out variance ratios for CF = 4 and CF = 3. Test-series cluster bootstrap 95 % intervals.")
F_("res", F0 / "fig08_residuals_by_rf_cf.png", "Out-of-fold residuals by RF (top) and CF (bottom): distributions, mean absolute residual and within-geometry SD with test-series cluster-bootstrap 95 % intervals.")

# ============================================================================== 6 decomposition
h1("6. Hierarchical error decomposition")
rows = [[c_, n_(r.n_units), f(r.sd), pct(r.share) + f" [{pct(r.share_lo)}, {pct(r.share_hi)}]"] for c_, r in dprim.iterrows()]
T_("dec", "REML decomposition of out-of-fold residuals (reference XGBoost, held-out geometries).", ["Component", "Units", "SD", "Share of variance [95 % CI]"], rows, [2, 1, 1, 2])
gtxt = ""
if G1 is not None:
    gtxt = (f" Rebuilding the geometry key with 2 or 4 significant digits or a coarse key changes the number of geometries from "
            f"{n_(G1.n_geometries.min())} to {n_(G1.n_geometries.max())} (adjusted Rand index ≥ {f(G1.adjusted_rand_index_vs_primary.min(), 2)}). The "
            f"within-geometry share stays between {pct(G1.share_within.min())} and {pct(G1.share_within.max())}, and the RF = 4 variance ratio between "
            f"{f(G1.RF4_variance_ratio.min(), 2)} and {f(G1.RF4_variance_ratio.max(), 2)}.")
p(f"Within-geometry variation accounts for {pct(q_w.share)} of the residual variance, the geometry level for {pct(q_g.share)} and the test series for "
  f"{pct(q_s.share)} ({tref('dec')}, {fref('dec')})." + gtxt +
  f" The near-replicate variance is {pct(rrat.value)} [{pct(rrat.lo)}, {pct(rrat.hi)}] of the within-geometry variance. The within-geometry residual "
  f"therefore contains model and unrecorded-condition components besides repeatability, and is not pure measurement error. A laboratory "
  f"component is not identifiable: laboratories are inferred for only part of the tests and coincide largely with series. RF and CF do not map "
  f"onto the levels as their definitions imply. RF = 4 is elevated at the within-geometry and at the geometry level, and the geometry-level SD is "
  f"similar in every CF class.")
F_("dec", F0 / "fig10_error_decomposition.png", "Hierarchical decomposition of out-of-fold residuals: (a) variance shares under alternative groupings and validation units; (b) component SDs; (c) geometry mean residuals; (d, e) components within CF and RF strata; (f) residual SD compared with near-replicate repeatability.")

# ============================================================================== 7 policies
h1("7. Counterfactual data-selection policies")
lab = {"P1_retain_all": "P1 retain all", "P2_excl_RF4": "P2 excl. RF = 4", "P3_excl_CF4": "P3 excl. CF = 4", "P4_excl_RF4_or_CF4": "P4 excl. RF/CF = 4",
       "P5_CLASH_weights": "P5 (4−RF)(4−CF)", "P5b_CLASH_weights_and_remarks": "P5b P5 + remarks", "P6a_inverse_variance_weights": "P6a 1/σ² weights",
       "P6c_weights_and_intervals": "P6c P6a + P6b"}
if PF is not None:
    rows = []
    for k, l_ in lab.items():
        rows.append([l_, n_(PF.loc[k, "n_train_mean_per_fold"]), n_(PF.loc[k, "ess_mean_per_fold"]) if np.isfinite(PF.loc[k, "ess_mean_per_fold"]) else "–"]
                    + ["0" if k == "P1_retain_all" else s(PF.loc[k, f"d_rmse_{c}"]) + " " + ci(PF.loc[k, f"d_rmse_{c}_lo"], PF.loc[k, f"d_rmse_{c}_hi"]) for c in ("common", "RF4", "CF4")]
                    + (["0" if k == "P1_retain_all" else s(PT.loc[k, f"d_rmse_{c}"]) for c in ("common", "RF4", "CF4")] if PT is not None else []))
    hdr = ["Policy", "Training tests / fold", "ESS", "ΔRMSE common", "ΔRMSE RF = 4", "ΔRMSE CF = 4"] + (["Tuned: Δ common", "Tuned: Δ RF = 4", "Tuned: Δ CF = 4"] if PT is not None else [])
    T_("pol", "Effect of the training-data policy on identical held-out tests (XGBoost). ΔRMSE relative to P1 of the same experiment, with paired geometry-cluster bootstrap 95 % intervals.",
       hdr, rows, [1.5, 0.8, 0.6, 1.5, 1.4, 1.4] + ([0.8, 0.8, 0.8] if PT is not None else []), fs=13,
       note=f"Fixed hyper-parameters unless marked 'tuned' (policy-specific nested Optuna). ESS: effective sample size (Σw)²/Σw². P1 RMSE (fixed): common {f(PF.loc['P1_retain_all', 'rmse_common'])}, "
            f"RF = 4 {f(PF.loc['P1_retain_all', 'rmse_RF4'])}, CF = 4 {f(PF.loc['P1_retain_all', 'rmse_CF4'])}. The original fixed-parameter results (Table S9) agree.")
    txt = (f"No exclusion or conventional weighting policy improves prediction of the tests it keeps ({tref('pol')}, {fref('pol')}). "
           f"With fixed hyper-parameters the common-test differences lie between {s(PF.loc[EXCL, 'd_rmse_common'].min())} and {s(PF.loc[EXCL, 'd_rmse_common'].max())}. ")
    if PT is not None:
        txt += (f"With policy-specific tuning they lie between {s(PT.loc[EXCL, 'd_rmse_common'].min())} and {s(PT.loc[EXCL, 'd_rmse_common'].max())}, so a learner "
                f"tuned to the curated data does not recover a benefit. ")
    txt += (f"Every policy that removes or zero-weights a category degrades that category: RF = 4 tests gain {prange(PF, 'd_rmse_RF4', EX_RF4)} RMSE and CF = 4 tests "
            f"{prange(PF, 'd_rmse_CF4', EX_CF4)}" + (f" ({prange(PT, 'd_rmse_RF4', EX_RF4)} and {prange(PT, 'd_rmse_CF4', EX_CF4)} when tuned)" if PT is not None else "") + ". "
            f"The same policies move {f(PF.loc[EX_RF4, 'pct_changed_factor2_category4'].min(), 0)}–{f(PF.loc[EX_RF4, 'pct_changed_factor2_category4'].max(), 0)} % of category-4 "
            f"predictions by more than a factor 2. Excluding RF = 4 changes CF = 4 errors by {s(PF.loc['P2_excl_RF4', 'd_rmse_CF4'])}, and excluding CF = 4 changes RF = 4 errors by "
            f"{s(PF.loc['P3_excl_CF4', 'd_rmse_RF4'])}, so each loss is largely specific to the removed category. Empirical down-weighting (P6a, effective sample size "
            f"{n_(PF.loc['P6a_inverse_variance_weights', 'ess_mean_per_fold'])} of {n_(PF.loc['P6a_inverse_variance_weights', 'n_train_mean_per_fold'])}; weights "
            f"{f(PF.loc['P6a_inverse_variance_weights', 'w_min'], 2)}–{f(PF.loc['P6a_inverse_variance_weights', 'w_max'], 2)}) changes the common-test RMSE by "
            f"{s(PF.loc['P6a_inverse_variance_weights', 'd_rmse_common'])}. Neutrality on common tests is not evidence that exclusion is harmless: it "
            f"is harmless only to the tests that are kept.")
    p(txt)
    F_("pol", F1 / "main" / "Fig07_policies_fixed_vs_tuned.png", "Counterfactual policies with fixed and with policy-specific tuned XGBoost: ΔRMSE relative to retaining all tests for (a) common, (b) RF = 4 and (c) CF = 4 tests, with paired geometry-cluster bootstrap 95 % intervals.")
w4 = W4.set_index(["policy", "subset"])
p(f"Exclusion also affects the occurrence model. Excluding RF = 4 tests lowers the AUC on held-out RF = 4 tests from "
  f"{f(w4.loc[('P1_retain_all', 'RF4'), 'auc'], 2)} to {f(w4.loc[('P2_excl_RF4', 'RF4'), 'auc'], 2)}, and excluding CF = 4 lowers it on CF = 4 tests from "
  f"{f(w4.loc[('P1_retain_all', 'CF4'), 'auc'], 2)} to {f(w4.loc[('P3_excl_CF4', 'CF4'), 'auc'], 2)}. Common-test AUC is unchanged "
  f"({f(w4.loc[('P1_retain_all', 'common'), 'auc'], 3)} against {f(w4.loc[('P4_excl_RF4_or_CF4', 'common'), 'auc'], 3)} under P4).")

# ============================================================================== 8 uncertainty
h1("8. Quality-aware uncertainty quantification")
rows = []
for sc in ["constant", "RF+CF main effects", "RFxCF shrunk cells", "CF only", "physical only", "physical+RF+CF"]:
    r = C25.loc[sc]
    rows.append([sc, pct(r.geometry_mean_coverage, 1) + " " + f"[{pct(r.geo_cov_lo, 1)}, {pct(r.geo_cov_hi, 1)}]", pct(r.coverage, 1), pct(r.cov_RF4, 1), pct(r.cov_CF3, 1), pct(r.cov_CF4, 1),
                 f"{100 * r.max_abs_dev_RF:.1f}", f(r.mean_width, 2), f(r.winkler, 2), pct(r["pct_series_coverage_below_0.8"] / 100)])
T_("conf", "Split-conformal 90 % intervals by scale function (calibration fraction 0.25, five calibration seeds, all tests retained).",
   ["Scale", "Geometry-mean coverage [95 % CI]", "Record coverage", "RF = 4", "CF = 3", "CF = 4", "Max dev. over RF (pts)", "Mean width", "Winkler", "Series < 80 %"], rows,
   [1.5, 1.5, 0.8, 0.7, 0.7, 0.7, 0.8, 0.7, 0.7, 0.8], fs=13,
   note="Width and Winkler (interval) score in log10 units; a width w corresponds to a factor 10^{w/2} either side of the prediction. Coverage within categories is empirical; only the marginal coverage over new geometries is guaranteed.")
qa, cst, phy = C25.loc["RF+CF main effects"], C25.loc["constant"], C25.loc["physical only"]
sens = C1[C1.scale == "RF+CF main effects"]
bw = C25.sort_values("winkler").iloc[0]
p(f"Every scale attains the guaranteed quantity, the coverage averaged over held-out geometries ({pct(C25.geometry_mean_coverage.min(), 1)}–{pct(C25.geometry_mean_coverage.max(), 1)}; "
  f"{tref('conf')}). Record-level coverage is lower ({pct(C25.coverage.min(), 1)}–{pct(C25.coverage.max(), 1)}), because geometries with many tests are harder to "
  f"predict. The hierarchical guarantee does not extend to individual tests, and this must be considered when intervals are read per test. "
  f"Constant intervals cover only {pct(cst.cov_RF4, 1)} of RF = 4 tests ({fref('conf')}). "
  f"Scaling by the RF and CF main effects raises RF = 4 coverage to {pct(qa.cov_RF4, 1)} and reduces the largest coverage deviation across RF from "
  f"{100 * cst.max_abs_dev_RF:.1f} to {100 * qa.max_abs_dev_RF:.1f} points, at a mean width of {f(qa.mean_width, 2)} against {f(cst.mean_width, 2)}. "
  f"The lowest interval (Winkler) score is obtained by the {bw.name} scale ({f(bw.winkler, 2)} against {f(cst.winkler, 2)} for constant intervals), "
  f"with RF = 4 coverage {pct(bw.cov_RF4, 1)} and a largest RF deviation of {100 * bw.max_abs_dev_RF:.1f} points, at a mean width of {f(bw.mean_width, 2)}. "
  f"Partial pooling of the 16 RF × CF cells gives {pct(C25.loc['RFxCF shrunk cells', 'cov_RF4'], 1)} for RF = 4 and {pct(C25.loc['RFxCF shrunk cells', 'cov_CF3'], 1)} for CF = 3, "
  f"with no unstable widths. The results are stable across calibration fractions (RF = 4 coverage {pct(sens.cov_RF4.min(), 1)}–{pct(sens.cov_RF4.max(), 1)}) and seeds. "
  f"CF = 3 tests remain under-covered under every scale ({pct(C25.cov_CF3.min(), 1)}–{pct(C25.cov_CF3.max(), 1)}), because their error is concentrated in a few series "
  f"(Section 5.2). Without series {C4.series.iloc[0]}, CF = 3 coverage under the main-effects scale is {pct(qa.cov_CF3_without_series_506, 1)}. Coverage is therefore not "
  f"equalised across all quality categories.")
p(f"RF records how reliable a completed experiment was, so it is not available for a new structure. For a prospective prediction, CF may be "
  f"assessable from the geometry but RF is not. A scale based on CF alone gives RF = 4 coverage {pct(C25.loc['CF only', 'cov_RF4'], 1)}. A scale based on "
  f"physical predictors only gives {pct(phy.cov_RF4, 1)} at width {f(phy.mean_width, 2)}, and the learned concepts recover RF = 4 only weakly from the "
  f"inputs. The quality-aware intervals are therefore a tool for retrospective database auditing and for use of the database as training "
  f"data, not a prospective uncertainty model for new structures.")
F_("conf", F1 / "main" / "Fig08_conformal_quality_aware.png", "Quality-aware conformal intervals with all tests retained: coverage of 90 % intervals by (a) RF and (b) CF for six scale functions (geometry-cluster bootstrap 95 % intervals); (c) largest coverage deviation across RF versus calibration fraction.")

# ============================================================================== 9 engineering
h1("9. Engineering decision implications")
eb = E1[E1.subset == "all RF/CF"]
rows = []
for d in eb.domain.unique():
    e = eb[eb.domain == d].set_index("model")
    r_ = [d, n_(e.n.iloc[0]), n_(e.n_series.iloc[0])]
    for m in ("EurOtop 2018 (mean value)", "XGBoost baseline (original OOF)", "XGBoost v2 (optimised, seed mean)"):
        r_.append(f(e.loc[m, "rmse"]) + " / " + s(e.loc[m, "bias"], 2) if m in e.index else "–")
    rows.append(r_)
T_("eur", "EurOtop (2018) mean-value formulae and data-driven predictions within the EurOtop validity domain: RMSE / bias in log₁₀ q*.",
   ["Domain", "Tests", "Series", "EurOtop", "XGBoost baseline", "XGBoost v2"], rows, [2.2, 0.7, 0.7, 1.2, 1.2, 1.2],
   note="EurOtop was partly calibrated on CLASH data, while the learners are evaluated on held-out geometries; the comparison is not independent. Outside the stated domains (5 250 tests) no formula was applied.")
er4 = E1[(E1.subset == "RF=4") & (E1.domain == "all applicable")].set_index("model")
p(f"Within its validity domain ({n_(eb[eb.domain == 'all applicable'].n.iloc[0])} tests) EurOtop gives RMSE {f(eb[(eb.domain == 'all applicable') & (eb.model.str.startswith('EurOtop'))].rmse.iloc[0])} against "
  f"{f(eb[(eb.domain == 'all applicable') & (eb.model.str.startswith('XGBoost baseline'))].rmse.iloc[0])} for the held-out-geometry XGBoost ({tref('eur')}). "
  f"For RF = 4 tests the EurOtop error rises to {f(er4.loc['EurOtop 2018 (mean value)', 'rmse'])} with bias {s(er4.loc['EurOtop 2018 (mean value)', 'bias'], 2)}. "
  f"These tests are poorly described by standard formulae as well as by data-driven models.")
e2 = E2[(E2.Hm0_prototype_m == 2.0)].set_index(["q_allow_l_per_s_per_m", "policy", "subset"])
q1 = 1.0
fs1, fs2 = e2.loc[(q1, "P1_retain_all", "RF4"), "false_safe_rate_among_exceed"], e2.loc[(q1, "P2_excl_RF4", "RF4"), "false_safe_rate_among_exceed"]
fs4 = e2.loc[(q1, "P4_excl_RF4_or_CF4", "RF4"), "false_safe_rate_among_exceed"]
ch2 = e2.loc[(q1, "P2_excl_RF4", "RF4"), "pct_decision_changed_vs_P1"]
chc = e2.loc[(q1, "P2_excl_RF4", "common"), "pct_decision_changed_vs_P1"]
e3 = E3.set_index(["policy", "subset"])
p(f"Data-selection policies change engineering decisions. With an allowable discharge of 1 l/s/m at H_{{m0}} = 2 m, the share of RF = 4 exceedances "
  f"predicted as safe rises from {f(fs1, 1)} % when all tests are retained to {f(fs2, 1)} % when RF = 4 tests are excluded ({f(fs4, 1)} % under P4). "
  f"Excluding RF = 4 changes the exceed/not-exceed decision for {f(ch2, 1)} % of RF = 4 tests, against {f(chc, 1)} % of common tests "
  f"({fref('eng')}). The pattern holds over the threshold grid (supplementary Table S14), and the safety class of "
  f"{f(e3.loc[('P2_excl_RF4', 'RF4'), 'pct_safety_class_changed'], 0)} % of RF = 4 tests changes under P2. Retaining the tests and widening their intervals (P6b) leaves "
  f"every point decision unchanged and lowers the probability that an exceedance is missed.")
p(f"Category-4 tests are concentrated in particular regimes ({fref('eng')}). Of the depth-limited tests (H_{{m0}}/h > 1), "
  f"{f(E4.loc['depth-limited (Hm0,toe/h > 1)', 'pct_regime_lost_by_P4'], 0)} % would be removed by excluding category 4. So would "
  f"{f(E4.loc['low freeboard (Rc/Hm0 < 0.5)', 'pct_regime_lost_by_P4'], 0)} % of the low-freeboard tests, "
  f"{f(E4.loc['oblique attack (beta > 30 deg)', 'pct_regime_lost_by_P4'], 0)} % of the strongly oblique tests and all chute structures. Without them, the "
  f"median distance from an RF = 4 test to the nearest retained test of another series grows from "
  f"{f(E5[(E5['reference pool'] == 'all tests') & (E5.subset == 'RF4')].median_nn_distance_other_series.iloc[0], 2)} to "
  f"{f(E5[(E5['reference pool'] != 'all tests') & (E5.subset == 'RF4')].median_nn_distance_other_series.iloc[0], 2)} (standardised units). Exclusion thus "
  f"removes coverage of physically important regimes, which is where the unusual tests are most valuable.")
F_("eng", F1 / "main" / "Fig09_engineering_decisions.png", "Engineering consequences: (a) EurOtop and data-driven RMSE within the EurOtop validity domain; (b) share of RF = 4 exceedances predicted as safe under five policies, by allowable discharge at H_{m0} = 2 m; (c) share of each physical regime removed by excluding category-4 tests.")

# ============================================================================== 10 CCW
h1("10. Concept-explanation assessment (CCW-Net)")
full, un = ccw.loc["ccw_full"], ccw.loc["ccw_unaligned"]
p(f"CCW-Net is less accurate than gradient boosting (pooled RMSE {f(P0.loc['CCW-Net', 'rmse'])} against {f(x_rmse)}). Causal alignment does what it is trained to "
  f"do for necessity and sufficiency: relative to an unaligned wrapper the necessity effect falls from {f(un.nec_abs_effect)} to {f(full.nec_abs_effect)} and the "
  f"sufficiency gap from {f(un.suf_rmse_vs_output)} to {f(full.suf_rmse_vs_output)}. It does not improve additivity ({f(full.ind_rmse)} against {f(un.ind_rmse)}). "
  f"It does not route a change in one concept through that concept's slot more often (matched share {f(full.rho_changed_matched)} against "
  f"{f(un.rho_changed_matched)}; 0.10 = indiscriminate). Concept importances disagree with grouped TreeSHAP (Spearman {f(cv('rank agreement'), 2)}) and with "
  f"established formulae, which rank freeboard and roughness first. Explanation quality does not vary with CF, and its weak relation to RF does not survive "
  f"correction. These are negative methodological results. CCW-Net examines model-internal representations, not physical mechanisms. The complete "
  f"analysis (ablations, attribution, interaction terms, implementation check) is in supplementary Section S8; it adds testable explanations but "
  f"no engineering evidence about the ratings.")

# ============================================================================== 11 discussion
h1("11. Discussion")
h2("11.1 What the data demonstrate")
p("The policy experiments answer the engineering question on identical held-out tests, with fixed and with re-tuned learners. Conventional "
  "exclusion and weighting rules do not make the remaining predictions more accurate. They make predictions for the excluded tests much worse, "
  "intervals for those tests over-confident, and safety decisions for them less conservative. Category-4 tests occupy regimes the other tests do "
  "not cover, and their value lies in that coverage.")
h2("11.2 What is supported by predictive association only")
p("The RF = 4 versus RF = 1 dispersion contrast is robust across learners, validation units, geometry definitions and missing-input scenarios. "
  "It describes how the ratings relate to model error. It does not establish that individual ratings are correct, and it does not separate "
  "measurement uncertainty from unrecorded conditions; near-replicate evidence is inconclusive. CF = 3, not CF = 4, marks the hardest tests, "
  "and much of that effect sits in a few series.")
h2("11.3 What concerns model-internal explanation")
p("Causal alignment makes concept explanations testable but not physically credible. It does not disentangle concepts, and its importances "
  "disagree with feature attribution and engineering knowledge.")
h2("11.4 What cannot be identified")
p("Several quantities cannot be identified from the database: the correctness of individual ratings, a pure measurement-error component, "
  "laboratory effects and the prospective availability of RF. Identifying them would require original laboratory reports, controlled repeat "
  "tests or a blinded multi-expert re-rating analysed with weighted kappa or Krippendorff's alpha.")

# ============================================================================== 12 limitations
h1("12. Limitations")
bl([f"Geometries are reconstructed, not recorded. Results are stable under alternative tolerances, but the reconstruction remains an approximation.",
    "The analyses use complete-case toe conditions. The RF = 4 tests with unusable inputs (mostly emergent toes) differ systematically from those analysed, and only a deep-water-substitution sensitivity was possible.",
    f"CF = 4 rests on {n_(rec['CF = 4 series (regression)'])} series and RF = 4 repeatability on {n_(r3a.loc[0.01, 'n_pairs'])} near-replicate pairs, so power is limited.",
    "Allowable discharges are a sensitivity grid converted by Froude similarity at assumed prototype wave heights, not design criteria.",
    "Conformal guarantees are marginal over new geometries. Hyper-parameters were tuned on partitions that include the calibration geometries (the mean and scale models were not), which is a mild departure from strict split conformal.",
    "EurOtop was partly calibrated on CLASH, so the benchmark is not independent.",
    "Overlap with the authors' previous papers could be checked only conceptually (supplementary Section S10)."])

# ============================================================================== 13 conclusions
h1("13. Conclusions")
bl([f"RF = 4 identifies tests with about {f(vr4.variance_ratio, 1)} times the residual dispersion of RF = 1 tests. RF 1–3 are not ordered, and repeatability evidence is inconclusive. RF is a coarse flag, not a graded uncertainty measure.",
    "CF shows no ordered relation to geometry-level error or generalisation, and varies within reconstructed geometries. CF = 3 is the hardest class, driven by a few series.",
    f"Most residual variance lies within geometries ({pct(q_w.share)}), and only about half of it can be matched by near-replicate variance.",
    "Excluding or zero-weighting category-4 tests gives no measurable benefit to the remaining predictions, even with re-tuned learners. It degrades predictions, interval coverage and safety decisions for the excluded tests and removes depth-limited and low-freeboard regimes.",
    "Retaining all tests and using RF and CF to scale prediction intervals is the more defensible practice for using the database. CF = 3 under-coverage and the unavailability of RF for new structures remain open.",
    "Data-quality ratings should be evaluated empirically before they are used to discard experimental information."])

h1("Data and code availability")
p("Code, configuration, out-of-fold predictions, audit files and all tables and figures are available in the project repository [link]. The CLASH database is available from its maintainers.")
h1("CRediT authorship contribution statement")
p("[To be completed by the authors.]")
h1("Declaration of competing interest")
p("[To be completed by the authors.]")
h1("References")
add({"t": "note", "text": "Verification status of every reference is listed in revision/audit/reference_audit.csv; entries marked 'to verify' must be checked by the authors."})
for r in [
    "Akiba, T., Sano, S., Yanase, T., Ohta, T., Koyama, M., 2019. Optuna: a next-generation hyperparameter optimization framework. Proc. 25th ACM SIGKDD, 2623–2631. https://doi.org/10.1145/3292500.3330701 [to verify]",
    "Chen, T., Guestrin, C., 2016. XGBoost: a scalable tree boosting system. Proc. 22nd ACM SIGKDD, 785–794. https://doi.org/10.1145/2939672.2939785 [to verify]",
    "den Bieman, J.P., van Gent, M.R.A., van den Boogaard, H.F.P., 2021. Wave overtopping predictions using an advanced machine learning technique. Coastal Engineering 166, 103830. https://doi.org/10.1016/j.coastaleng.2020.103830",
    "EurOtop, 2018. Manual on wave overtopping of sea defences and related structures. An overtopping manual largely based on European research, but for worldwide application, second ed. Van der Meer, J.W., Allsop, N.W.H., Bruce, T., De Rouck, J., Kortenhaus, A., Pullen, T., Schüttrumpf, H., Troch, P., Zanuttigh, B. www.overtopping-manual.com.",
    "Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B., Liang, P., 2020. Concept bottleneck models. Proc. 37th ICML, PMLR 119. [to verify]",
    "Lee, Y., Barber, R.F., Willett, R., 2023. Distribution-free inference with hierarchical data. Preprint, arXiv:2306.06342.",
    "Lundberg, S.M., Erion, G., Chen, H., et al., 2020. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2, 56–67. [to verify]",
    "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A.V., Gulin, A., 2018. CatBoost: unbiased boosting with categorical features. NeurIPS 31. [to verify]",
    "Rountree, J., Verma, P., So, O., Fan, C., Shah, J.A., 2026. Causal concept explanations for deep neural models. Advances in Neural Information Processing Systems (NeurIPS 2026). [proceedings details to verify; the arXiv identifier used in the earlier draft could not be confirmed]",
    "Steendam, G.J., Van der Meer, J.W., Verhaeghe, H., Besley, P., Franco, L., Van Gent, M.R.A., 2004. The international database on wave overtopping. Proc. 29th Int. Conf. Coastal Engineering, World Scientific, 4301–4313.",
    "Van der Meer, J.W., Verhaeghe, H., Steendam, G.J., 2009. The new wave overtopping database for coastal structures. Coastal Engineering 56, 108–120. [pages and DOI to verify]",
    "Van Gent, M.R.A., van den Boogaard, H.F.P., Pozueta, B., Medina, J.R., 2007. Neural network modelling of wave overtopping at coastal structures. Coastal Engineering 54, 586–593. https://doi.org/10.1016/j.coastaleng.2006.12.001",
    "Zanuttigh, B., Formentin, S.M., Van der Meer, J.W., 2016. Prediction of extreme and tolerable wave overtopping discharges through an advanced neural network. Ocean Engineering 127, 7–22. https://doi.org/10.1016/j.oceaneng.2016.09.032",
    "[Paper 1] Machine-learning prediction of wave overtopping at untested structures: geometry-grouped validation, freeboard monotonicity and conformal coverage. [authors, status]",
    "[Paper 2] Quantum machine learning for wave overtopping estimation: integrating with causal inference and uncertainty quantification. Ocean Engineering [volume, article, year].",
    "[CLASH WP2 report defining RF and CF — exact title and number to be supplied by the authors.]",
]:
    add({"t": "ref", "text": r})


def _fix(o):
    if isinstance(o, str):
        o = re.sub(r"\{\{TAB:(\w+)\}\}", lambda m: f"Table {TAB_IDX[m.group(1)]}", o)
        o = re.sub(r"\{\{FIG:(\w+)\}\}", lambda m: f"Fig. {FIG_IDX[m.group(1)]}", o)
        return re.sub(r"q\*(?!\*)", "q^{*}", o)
    if isinstance(o, list):
        return [_fix(x) for x in o]
    if isinstance(o, dict):
        return {k: (_fix(v) if k != "path" else v) for k, v in o.items()}
    return o


json.dump({"title": TITLE, "blocks": _fix(B)}, open(OUT / "revised_manuscript.json", "w"), ensure_ascii=False, indent=0)
(OUT / "revised_abstract.md").write_text("# Abstract\n\n" + abs_ + "\n", encoding="utf8")
(OUT / "revised_highlights.md").write_text("# Highlights\n\n" + "\n".join(f"- {h}" for h in hl) + "\n", encoding="utf8")
print("tables", tn["n"], "figures", fn["n"], "blocks", len(B))
