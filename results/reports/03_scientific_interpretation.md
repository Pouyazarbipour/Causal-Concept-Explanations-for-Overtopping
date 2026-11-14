# 3. Scientific interpretation

**Engineering question.** Are the CLASH data-quality ratings valid and useful for deciding which experimental
observations should be retained, down-weighted or excluded when developing prediction models?

Response: Y = log10 q*, q* = q / sqrt(g Hm0,toe^3). Errors are in log10 units (0.30 = factor 2). Unless stated
otherwise: 9 333 positive-discharge tests, 5-fold geometry-grouped nested cross-validation with Optuna tuning
inside the training folds, 95 % cluster-bootstrap intervals in brackets (test series for RF/CF statistics,
geometry for prediction, policy and coverage statistics). RF/CF residual analyses use the pre-specified
reference model (XGBoost) and are repeated with the other models.

Evidence classes: **[D]** directly demonstrated by the data; **[P]** predictive association only;
**[M]** property of a model's internal representation; **[N]** not identifiable with the available evidence.

---

## 3.1 Diagnostic models (Tables 3, 4; Figures 3–7, S1)

| Model | RMSE | MAE | R² | bias | F10 |
|---|---|---|---|---|---|
| Random Forest | 0.539 [0.489, 0.590] | 0.375 | 0.787 | +0.002 | 3.46 |
| XGBoost | **0.501** [0.458, 0.547] | 0.348 | 0.816 | −0.010 | 3.17 |
| CatBoost | 0.513 [0.462, 0.567] | 0.347 | 0.808 | −0.011 | 3.26 |
| MLP (3 seeds) | 0.576 [0.522, 0.636] | 0.403 | 0.757 | −0.003 | 3.77 |
| CCW-Net (3 seeds) | 0.601 [0.542, 0.665] | 0.415 | 0.736 | −0.020 | 3.99 |

F10 here is 10^RMSE, the geometric standard-deviation factor of predicted/measured discharge. Bias is
predicted minus measured. For the MLP and CCW-Net the table gives the mean of per-seed metrics; averaging the three
seeds' predictions (seed ensemble, Figure 3) gives RMSE 0.550 and 0.564.

* [D] The models serve as instruments for obtaining leakage-free residuals; the ranking is not a
  contribution. The boosted trees are the most accurate; the networks are 0.07–0.10 worse.
* [D] The Optuna objective (inner grouped CV) is close to the outer test error for every model (XGBoost
  0.544 inner vs 0.500 outer). The best-so-far curves flatten within the budget for the tree models
  (Figure S1); the MLP and CCW-Net budgets are small (L9).
* [D] Residual-versus-prediction plots show no systematic trend over the bulk of the range. The binned
  median rises to +0.3 to +0.5 below log10 q* ≈ −6 for XGBoost, CatBoost and the MLP, where the smallest
  measured discharges are under-predicted (Figure 7).
* [D] Sensitivity of the reference model to the validation unit: random records 0.354, geometry 0.501, test
  series held out 0.749.

## 3.2 RQ1 – RF validity: does RF separate tests by measurement-associated uncertainty?

**Partly. RF = 4 is distinguished from RF = 1; RF = 2 and RF = 3 are not distinguishable from each other.
The scale is not an ordered measure of uncertainty.**

| RF | n | series | MAE | RMSE | within-geometry SD | repeat SD (near-replicates) |
|---|---|---|---|---|---|---|
| 1 | 3 189 | 54 | 0.306 [0.262, 0.357] | 0.431 [0.375, 0.494] | 0.340 [0.298, 0.384] | 0.19 [0.14, 0.23] (130 pairs) |
| 2 | 3 174 | 71 | 0.363 [0.298, 0.455] | 0.527 [0.419, 0.661] | 0.395 [0.318, 0.492] | 0.23 [0.17, 0.32] (483) |
| 3 | 2 333 | 76 | 0.357 [0.294, 0.442] | 0.516 [0.416, 0.633] | 0.350 [0.309, 0.394] | 0.36 [0.21, 0.44] (457) |
| 4 | 637 | 41 | 0.451 [0.373, 0.545] | 0.630 [0.522, 0.751] | 0.470 [0.378, 0.570] | 0.22 [0.13, 0.31] (66) |

(Tables 5a, 5h; Figures 8, 9.)

* [P] Residual variance of RF = 4 relative to RF = 1, adjusted for CF: 2.34 [1.41, 3.81]. For RF = 2 and RF = 3
  the ratios are 1.38 [0.90, 2.15] and 1.50 [0.86, 2.34]. Every learner gives an RMSE ratio RF4/RF1 between
  1.39 and 1.58 with intervals above 1 (Table 5b).
* [P] As an ordered scale RF is weakly informative: Spearman ρ(|residual|, RF) = 0.07 [−0.01, 0.15]; odds ratio
  per SD of |residual| 1.17 [0.98, 1.35]; no trend test survives Holm correction (Tables 5b, 5c).
* [D] Model-free repeatability increases from RF 1 to RF 3 (0.19 → 0.36) but not to RF 4 (0.22, 66 pairs).
  This is the only evidence in the repository that does not depend on a model.
* [D] Construct-related evidence: RF rises with inferred indicators of estimated or missing parameters
  (Spearman 0.30 [0.11, 0.46]). A spectral period equal to exactly Tp/1.1 predicts a higher RF more strongly
  (OR 2.06 [1.14, 3.53]) than the prediction error does (Table 5c). All records with unusable inputs have
  RF = 4.
* [P] RF = 4 tests have more zero-overtopping records (16.5 % vs 5.8 % for RF = 1) and a weaker wet/dry
  discrimination (AUC 0.81 vs 0.92–0.96).
* [N] Whether individual RF assignments are correct, and how much of the RF-4 excess is measurement error
  rather than input uncertainty or unrepresented physics, cannot be determined.

## 3.3 RQ2 – CF validity: does CF identify geometries that are hard to represent?

**No ordered relation. CF = 3 is the difficult class; CF = 2 and CF = 4 are not harder than CF = 1.**

| CF (max in geometry) | geometries | series | |geometry mean residual| | geometry RMSE | grouped − random RMSE |
|---|---|---|---|---|---|
| 1 | 1 367 | 120 | 0.259 [0.232, 0.297] | 0.316 | 0.046 [0.033, 0.066] |
| 2 | 363 | 48 | 0.239 [0.201, 0.282] | 0.303 | 0.043 [0.023, 0.065] |
| 3 | 263 | 27 | 0.319 [0.261, 0.399] | 0.391 | 0.024 [0.003, 0.055] |
| 4 | 53 | 7 | 0.255 [0.147, 0.380] | 0.326 | 0.087 [0.041, 0.189] |

(Tables 6a, 6b; Figure 9d.)

* [P] No geometry-level quantity is rank-correlated with CF: |mean residual| 0.03 [−0.06, 0.10], geometry
  RMSE 0.03 [−0.10, 0.13], generalisation gap 0.00 [−0.07, 0.06]. Geometry-level ordinal model: OR 1.07
  [0.89, 1.24] per SD of |mean residual|.
* [P] CF = 3 has the largest record-level error (RMSE 0.71 [0.47, 0.98]) and variance ratio 2.18 [1.01, 3.87].
  When whole test series are held out, CF shows a weak positive trend (ρ = 0.11 [0.01, 0.22], Holm-adjusted
  p = 0.32), which suggests that CF relates to transfer between investigations rather than between
  geometries within one investigation.
* [D] CF = 4 consists of 212 usable tests from 7 series, dominated by one chute investigation, and is
  predicted well (RMSE 0.36). This says that those tests are internally consistent, not that complex
  sections are easy to represent.
* [D] CF is not constant within 32 reconstructed geometries, so it is not a pure structure-level label in
  this file.
* [N] Whether CF correctly rates geometric complexity needs an independent re-rating of cross-sections.

**RF × CF (Table 7).** No interaction: the linear-by-linear term in the variance model is 1.06 [0.83, 1.29].
Six of sixteen cells are too sparse to interpret (all CF = 4 cells and RF 4/CF 3).

## 3.4 RQ3 – Can test-level, geometry-level and source effects be separated? (Table 8; Figure 10)

**Test-level and geometry-level components are separated with useful precision. A series component is poorly
determined; a laboratory component is not identifiable.**

| Component (XGBoost, geometry-grouped residuals) | units | SD | share of residual variance |
|---|---|---|---|
| test series | 162 | 0.177 | 13 % [2, 29] |
| geometry within series | 2 046 | 0.262 | 29 % [20, 40] |
| within-geometry residual | 9 333 | 0.372 | 58 % [48, 64] |

* [D] The shares are stable under other geometry definitions (geometry 25–28 %, residual 56–60 %).
* [D] They depend on the validation unit and the model, which is the model-induced part. Under random
  splits the geometry share falls to 11 % because the model memorises geometry offsets. With whole series
  held out, the series share rises to 45 %. For the networks the series share is 35–39 % against 13–19 %
  for the tree models.
* [N] Source and laboratory: a reference-level component is 2 %. With 12 inferred laboratories
  (6 514 records) the laboratory variance is estimated at zero. Given confounding with structure type and
  30 % unassignable records, this is absence of evidence, not evidence of absence.
* [D] Model-free repeatability: the near-replicate variance (0.080; SD 0.28) is 58 % of the within-geometry
  residual variance (0.138). The within-geometry residual is therefore **not** pure measurement error; about
  40 % of it is unexplained test-to-test variation beyond what replicate pairs show. Replicate pairs are
  themselves an upper bound on repeatability.
* [P] **Does RF explain the test level and CF the geometry level, as their definitions imply? Not cleanly.**
  * Within-geometry SD by RF: 0.33, 0.40, 0.35, 0.42 (RF 1–4). This is the expected direction between RF 1
    and RF 4, with overlapping intervals in between.
  * The geometry-level SD is also larger for RF = 4 (0.36 [0.28, 0.52] vs 0.26), so RF = 4 marks harder
    geometries as well as noisier tests.
  * The geometry-level SD is the same in every CF class (0.26, 0.24, 0.26, 0.21). CF = 3 differs instead in
    its *within-geometry* residual (0.53 [0.33, 0.74] vs 0.35), the opposite of what a structure-level
    complexity rating should show.

## 3.5 RQ4/RQ6 – Counterfactual data-selection policies (Tables 13–15; Figures 13–15)

Only the training policy changes; folds, learner, hyper-parameters, seeds and calibration geometries are
held fixed. ΔRMSE is relative to retaining all records (P1), on the same held-out tests (XGBoost).

| Policy | records used | ΔRMSE, RF<4 & CF<4 (n = 8 494) | ΔRMSE, RF = 4 (n = 637) | ΔRMSE, CF = 4 (n = 212) | 90 % coverage RF = 4 / CF = 4 | max coverage deviation across RF | predictions changed > factor 2 (category-4 tests) |
|---|---|---|---|---|---|---|---|
| P1 retain all | 9 333 | 0 | 0 (RMSE 0.641) | 0 (0.360) | 81 % / 94 % | 8.7 pts | – |
| P2 exclude RF = 4 | 8 696 | +0.001 [−0.007, +0.009] | **+0.282** [+0.185, +0.395] | −0.016 | **64 %** / 93 % | 25.6 | 26 % |
| P3 exclude CF = 4 | 9 121 | 0.000 [−0.006, +0.006] | −0.011 | **+0.225** [+0.171, +0.295] | 81 % / **77 %** | 9.5 | 9 % |
| P4 exclude RF = 4 or CF = 4 | 8 494 | −0.005 [−0.012, +0.002] | +0.266 [+0.160, +0.388] | +0.222 [+0.163, +0.294] | 64 % / 74 % | 25.6 | 35 % |
| P5 weights (4−RF)(4−CF) | 8 494 | −0.003 [−0.013, +0.008] | +0.297 [+0.189, +0.416] | +0.266 [+0.205, +0.357] | 63 % / 75 % | 27.0 | 38 % |
| P5b P5 + "do not use" removed | 8 195 | −0.007 [−0.028, +0.011] | +0.310 | +0.262 | 64 % / 76 % | 26.3 | 41 % |
| P6a empirical weights 1/σ²(RF,CF) | 9 333 | −0.001 [−0.010, +0.011] | +0.013 [−0.014, +0.036] | +0.013 | 78 % / 94 % | 12.1 | 4 % |
| P6b retain all + RF/CF-aware intervals | 9 333 | 0 | 0 | 0 | **89 %** / 90 % | **4.3** | 0 % |
| P6c P6a + P6b | 9 333 | −0.001 | +0.013 | +0.013 | 87 % / 90 % | 3.6 | 4 % |

* [D] **No exclusion or conventional weighting policy improves prediction on the tests it keeps**: all
  differences on the common tests lie within ±0.01 with intervals spanning zero. Random Forest agrees
  (−0.002 to +0.009).
* [D] **Every policy that removes or zero-weights category 4 makes the predictions for that category much
  worse**: +0.22 to +0.31 RMSE, i.e. from a typical error factor of 4.4 to 8.7 for RF = 4 tests, and moves
  26–41 % of category-4 predictions by more than a factor 2 (up to 10 % by more than a factor 10). Coverage
  of 90 % intervals for RF = 4 tests drops to 63–64 %.
* [D] Excluding RF = 4 does not harm CF = 4 predictions and vice versa: the information loss is specific to
  the removed category.
* [D] Geometry-level bias increases under every exclusion policy (0.251 → 0.279–0.287); structure-family
  bias changes little (largest family bias 0.07–0.09; Table 15).
* [P] The empirically weighted policy (P6a) is neutral. Down-weighting noisy tests in proportion to their
  estimated residual variance neither helps the common tests nor harms category 4 materially.
* [P] **The defensible alternative is P6b**: keep every test with equal weight in the mean model and let
  RF/CF widen the uncertainty band. It equalises coverage across RF categories (largest deviation 4.3 points
  against 8.7 for P1 and 26–27 for conventional practice) at the same average width. Coverage for CF = 3
  improves from 79 % to 84 % but remains below nominal.
* The difference between these policies is a change in the training data, not a change in any physical
  input. It is an "intervention" on the data-selection decision.

## 3.6 RQ5 – Causal concept explanations (Tables 9–12; Figures 11, 12)

CCW-Net was tuned with Optuna in every outer fold (concept dimension 4–16, encoder width, learning rate,
interaction rank, alignment-weight multiplier 0.1–10). Ablations reuse the tuned configuration. The
losses and the matched interventional-effect estimator follow the reference implementation of Rountree
et al. (2026) after the corrections listed in report 08. **All CCW-Net numbers in earlier drafts,
including the "13.5 % routed through the changed concept" statement, were produced by a biased estimator
and are withdrawn.**

| Variant | RMSE | necessity: effect of swapping an absent concept | sufficiency: active-only vs output | additivity residual | zero-ablation share per absent concept | matched share via changed concept | matched share per held absent concept |
|---|---|---|---|---|---|---|---|
| CCW-Net | 0.597 ± 0.073 | **0.038** ± 0.012 | **0.108** ± 0.029 | 0.062 | **3.4 %** ± 0.5 | 0.171 ± 0.017 | **3.8 %** ± 0.5 |
| unaligned wrapper | 0.592 | 0.055 ± 0.016 | 0.273 ± 0.134 | 0.049 | 4.5 % | 0.169 | 5.2 % |
| without necessity | 0.593 | 0.045 | 0.117 | 0.053 | 3.6 % | 0.176 | 4.4 % |
| without sufficiency | 0.605 | 0.043 | 0.209 | 0.052 | 4.3 % | 0.168 | 5.0 % |
| without independence | 0.605 | 0.035 | 0.118 | 0.128 | 3.4 % | 0.166 | 3.8 % |
| without interaction term | 0.607 | 0.037 | 0.107 | 0.059 | 3.9 % | 0.152 | 4.2 % |
| without conditional counterfactual | 0.600 | 0.030 | 0.103 | 0.056 | 3.2 % | 0.162 | 3.2 % |
| concept bottleneck | 0.599 | – | – | – | 9.6 % | – | – |

(± = SD over fold × seed runs; 15 runs for CCW-Net, unaligned and bottleneck, 10 for ablations. RMSE here is
the mean over runs; Table 3 gives the mean of per-seed pooled metrics, 0.601. Reference value for an
indiscriminate split over ten concepts: 0.10.)

* [M] **Alignment does what it is trained to do, but only for its own criteria.** Relative to the unaligned
  wrapper the necessity effect falls by a third (0.038 vs 0.055, paired difference 0.017 ± 0.009, 95 %), the
  sufficiency gap from 0.27 to 0.11 (0.165 ± 0.066) and the share of the explicit interaction term from 0.55
  to 0.23. Accuracy is unchanged (paired ΔRMSE −0.005 ± 0.007, not significant). Each ablation degrades mainly
  the criterion its loss targets: without necessity +0.008 ± 0.006 necessity effect, without sufficiency
  +0.099 ± 0.084 sufficiency gap, without independence +0.069 ± 0.023 additivity residual.
* [M] **Negative results.**
  * (i) The additivity residual of the aligned model (0.062) is not lower than that of the unaligned wrapper
    (0.049; difference not significant). The independence loss only prevents a deterioration that alignment
    otherwise causes.
  * (ii) The conditional flow is unnecessary. Drawing counterfactuals from other rows of the batch gives the
    same or slightly better scores (necessity −0.007 ± 0.003).
  * (iii) Concept routing is better than indiscriminate but not improved by alignment. When a concept's label
    differs between matched tests, 17.1 % of the interventional effect passes through that concept's slot,
    against 16.9 % for the unaligned wrapper and 10 % for an indiscriminate split. The slots stay entangled
    because all are computed from one shared backbone latent.
* **Value beyond conventional feature attribution (Table 11).**
  * [M] Under matched interventions CCW-Net attributes 3.8 % of the effect to each held absent concept (no
    berm, no toe, normal incidence, smooth, plain crest, flat foreshore), against 5.2 % for the unaligned
    wrapper (difference 0.015 ± 0.013, marginal). Zero-ablation gives 3.4 % (CCW-Net), 4.5 % (unaligned), 9.6 %
    (bottleneck) and 6.4 % (grouped TreeSHAP of XGBoost).
  * This difference is partly by construction, not a demonstration of greater physical truth. CCW-Net is
    trained to give null states no effect, whereas "smooth slope" or "normal incidence" do physically affect
    overtopping, and SHAP correctly gives them weight (for example roughness: |SHAP| 0.53 when smooth).
  * The two methods disagree on which concepts matter (Spearman 0.07 over ten concepts). CCW-Net spreads
    importance almost evenly over profile type, relative depth, steepness and relative freeboard
    (0.22–0.26 log10 units each); SHAP ranks relative freeboard and roughness first (0.47 each), which agrees
    with established overtopping formulae.
  * The added value of CCW-Net is therefore a *testable* explanation: necessity, sufficiency and additivity
    can be measured and enforced. Its concept importances are not more physically credible than SHAP and
    should not be read as physical effects.
* **Explanation quality versus data quality (Tables 12a–c).**
  * [M] Explanation-quality measures do not differ by CF.
  * [M] With RF they move weakly. Absolute error rises with RF (ρ = 0.12 [0.02, 0.21]). The zero-ablation
    share of absent concepts falls (−0.15 [−0.31, −0.01]). Concept read-out accuracy is lower for RF = 4 (0.84
    vs 0.92 for RF = 1). Necessity, sufficiency and additivity do not change with RF or CF. None of the 28
    trend tests survives Holm correction.
  * [P] A linear probe recovers RF = 4 from the learned concept vectors better than from the raw predictors
    (AUC 0.73 vs 0.59, geometry-grouped). CF ≥ 3 is recoverable across geometries (0.77) but not across test
    series (0.48). The quality categories therefore mark particular regions of the physical input space and
    particular investigations; they do not transfer as a property of the physics.
* [M] The RF/CF variance branch on the frozen network has a better held-out likelihood than a constant
  variance (0.912 vs 0.933), whereas branches fed with concept vectors overfit (1.17–1.24).

## 3.7 Acceptance criteria (brief, Section 12)

| Criterion | Result |
|---|---|
| 1. Measurable predictive validity of RF and CF | RF: yes for the RF 4 vs 1 contrast (variance ratio 2.3), not as an ordered scale. CF: no ordered relation; CF = 3 differs. Established as predictive association only |
| 2. Test-level vs geometry-level separation | Yes statistically (58 % within geometry, 29 % between geometries). RF and CF do not map onto these levels as intended: RF = 4 is elevated at both levels; CF does not track the geometry level |
| 3. Is category-4 exclusion justified? | **No.** It does not improve the retained tests and costs 0.22–0.31 RMSE and 26 coverage points on category-4 tests |
| 4. Does quality-aware selection improve reliability? | Empirical down-weighting: no measurable effect. Quality-aware *intervals* with all data retained: yes, coverage equalised across RF |
| 5. Does causal concept learning add explanatory value over feature attribution? | Partly. It makes explanations testable and more faithful by its own necessity and sufficiency criteria. It does not route effects to the changed concept better than an unaligned wrapper (17 % vs 17 %, chance 10 %), its importances disagree with SHAP and with established formulae, and it is less accurate than the boosted trees (0.60 vs 0.50). Inconclusive as an engineering explanation tool |
| 6. Distinct from the two previous papers? | Yes for the core contributions (Section 6 report). Residual overlap: use of geometry-grouped validation and conformal intervals as instruments; to be checked against the full texts |
