# 3. Scientific interpretation

Response: Y = log10 q*, q* = q / sqrt(g Hm0,toe^3). All errors are in log10 units (0.30 = factor 2).
Unless stated otherwise, results are for the 9 333 positive-discharge records, 5-fold geometry-grouped
nested cross-validation, with 95 % cluster-bootstrap intervals in brackets (clusters: geometry for
prediction and coverage, test series for RF/CF statistics). Residual-based RF/CF analyses use the
pre-specified reference model (XGBoost on physical predictors).

Evidence classes used below:
**[D]** demonstrated directly by the data; **[P]** predictive association only; **[M]** property of a
model's internal representation; **[C]** physical input counterfactual of a fitted model;
**[N]** not identifiable with this repository.

---

## 3.1 Predictive performance (Experiment 2; Table 3, `table03*`)

| Validation scheme | RF | XGBoost | Monotone XGBoost | MLP | CCW-Net |
|---|---|---|---|---|---|
| Random records (optimistic) | 0.399 | 0.354 | 0.369 | 0.389 | – |
| **Geometry-grouped (primary)** | 0.526 | **0.493** [0.448, 0.541] | 0.521 | 0.590 | 0.594 [0.539, 0.653] |
| Coarse geometry | 0.611 | 0.605 | 0.630 | – | – |
| Test series held out | 0.769 | 0.755 [0.670, 0.836] | 0.781 | 0.836 | 0.844 |
| Source (reference) held out | 0.785 | 0.782 | 0.811 | – | – |
| Structural family held out | 0.905 | 0.828 | 0.858 | – | – |

(RMSE; MLP and CCW-Net are means of three seeds.)

* [D] The validation scheme changes the error more than the choice of learner. Random-record
  cross-validation understates the error on an unseen geometry by about 30 % and on an unseen test series
  by about 50 %. Reported CLASH accuracies from random splits should be read as interpolation within
  series.
* [D] XGBoost is the most accurate learner under every scheme. The neural models are about 0.10 worse
  under geometry grouping. On the primary scheme: backbone 0.594, concept bottleneck 0.608, unaligned
  wrapper 0.602, CCW-Net 0.594.
* [D] Monotonicity constraints cost 0.028 RMSE (0.493 -> 0.521) and remove every reversal by construction.
* [D] Holding out a whole structural family gives RMSE 0.48–0.94 and biases up to 0.5 (smooth slopes are
  under-predicted by 0.5 on average when absent from training). No claim of validity outside the represented families is made.
* Wet/dry model (10 389 records with q >= 0; Table 3d): AUC 0.927 overall; 0.92–0.96 for RF 1–3 and CF 1–3,
  0.81 for RF = 4 and 0.77 for CF = 4.
* Secondary ablation, RF/CF as inputs of the mean model: RMSE changes by -0.005 [-0.013, +0.005]
  (Table 3c). There is no predictive reason to put them into the mean model.

## 3.2 RQ1 – Does RF distinguish tests with different predictive and measurement-associated uncertainty?

**Partly. RF = 1 and RF = 4 are separated; RF = 2 and RF = 3 are not distinguishable from each other.**

| RF | n | series | MAE | RMSE | within-geometry residual SD |
|---|---|---|---|---|---|
| 1 | 3 189 | 54 | 0.294 [0.252, 0.343] | 0.418 [0.362, 0.479] | 0.329 [0.288, 0.367] |
| 2 | 3 174 | 71 | 0.355 [0.287, 0.447] | 0.520 [0.405, 0.657] | 0.385 [0.310, 0.481] |
| 3 | 2 333 | 76 | 0.349 [0.287, 0.437] | 0.512 [0.410, 0.626] | 0.349 [0.303, 0.398] |
| 4 | 637 | 41 | 0.430 [0.348, 0.527] | 0.615 [0.509, 0.733] | 0.470 [0.377, 0.571] |

(Table 8a, Figure 3.)

* [P] RMSE ratio RF4/RF1 = 1.47 [1.16, 1.83]; residual-variance ratio RF4 vs RF1, adjusted for CF,
  2.36 [1.38, 3.61] (Table 8c2, Figure 9a). The same contrast holds for every learner (ratios 1.39–1.80,
  all intervals above 1; Table 8b).
* [P] The ordinal trend as a whole is weak: Spearman rho(|residual|, RF) = 0.07 [-0.02, 0.16]; odds ratio
  of a higher RF per SD of |residual| 1.17 [0.99, 1.37] (Table 8). The intermediate steps are not ordered:
  RF 2 and RF 3 have the same error, and the variance ratios for RF 2 and RF 3 (1.41, 1.56) have intervals
  that include 1. The naive Jonckheere test is "highly significant" (z = 6.8) only because it treats 9 333
  clustered records as independent; it is reported for completeness and should not be quoted alone.
* [D] Model-free repeatability (1 136 near-replicate pairs: same geometry, Hm0, Tm-1,0, h, Rc within 1 %;
  Table 8h, Figure 10c): repeat SD of log10 q* is 0.19 [0.14, 0.23] when both tests have RF = 1, 0.23
  [0.17, 0.32] with RF <= 2, 0.36 [0.21, 0.44] with RF <= 3, and 0.22 [0.13, 0.31] for pairs involving an
  RF = 4 test (only 66 pairs). The increase from RF 1 to RF 3 is the only evidence in the repository that
  does not depend on any model; it is not monotone through RF = 4.
* [D] Construct-related evidence (Table 8f): RF rises with the inferred number of estimated or missing
  parameters (Spearman 0.30 [0.11, 0.46]); all records with unusable inputs and nearly all physically
  questionable records are RF = 4. In an ordinal model, the default-period indicator (Tm-1,0 = Tp/1.1)
  predicts RF more strongly (OR 2.06 [1.14, 3.53]) than the prediction error does.
* [N] Whether an individual RF assignment is correct, and how much of the RF-4 excess variance is
  measurement error rather than input uncertainty or unrepresented physics, cannot be determined.

## 3.3 RQ2 – Does CF identify structures that are hard to represent and generalise?

**Not as an ordered scale. CF = 3 stands out; CF = 2 and CF = 4 do not show larger errors than CF = 1.**

| CF (max in geometry) | geometries | series | |geometry mean residual| | generalisation gap (grouped - random RMSE) |
|---|---|---|---|---|
| 1 | 1 367 | 120 | 0.248 [0.220, 0.288] | 0.035 [0.024, 0.057] |
| 2 | 363 | 48 | 0.227 [0.182, 0.281] | 0.025 [0.011, 0.042] |
| 3 | 263 | 27 | 0.311 [0.252, 0.395] | 0.024 [0.000, 0.055] |
| 4 | 53 | 7 | 0.221 [0.129, 0.343] | 0.028 [0.006, 0.137] |

(Table 8d, Figure 9c.)

* [P] No geometry-level quantity is rank-correlated with CF: |mean residual| 0.01 [-0.08, 0.10], grouped
  RMSE 0.02 [-0.13, 0.13], generalisation gap -0.02 [-0.09, 0.03] (Table 8e). Geometry-level ordinal model:
  OR 1.07 [0.87, 1.26] per SD.
* [P] CF = 3 has the largest record-level error (RMSE 0.71 [0.46, 0.99]; variance ratio vs CF 1 of
  2.2 [0.98, 4.2]) and the largest fraction of XGBoost response curves with an upward step (41 %
  [34, 47] against 28 % for CF 1). The interval of the variance ratio includes 1; the result rests on 27
  series.
* [D] CF = 4 does **not** show larger errors (RMSE 0.33 [0.26, 0.60]). This is not evidence that very
  complex sections are easy: CF = 4 consists of 212 usable records from 7 series, 62 % of them chute tests
  from one investigation, tested over a narrow range of conditions. What the data show is that the CF = 4
  records in this database are internally consistent and predictable from each other, and that they are
  poorly predicted when removed from training (Section 3.8).
* [D] CF is not constant within a reconstructed structure (32 geometries, 311 records), so it is not a
  pure structure-level label in this file.
* [N] Whether CF is a correct rating of geometric complexity cannot be tested without independent
  re-rating of the cross-sections.

## 3.4 RQ – RF x CF interaction (Experiment 7; Table 10, Figure 9d)

* [P] No interaction is detectable: linear-by-linear RF x CF term in the variance model 1.06 per
  step x step [0.83, 1.30] (Table 8c). The main-effects description is adequate.
* [D] Six of the sixteen cells cannot support any statement (fewer than 50 records or fewer than three
  series): every CF = 4 cell and (RF 4, CF 3). They are reported with counts and marked "INSUFFICIENT".
* [P] Among adequately supported cells, MAE ranges from 0.26 (RF 1, CF 2) to 0.59 (RF 4, CF 2) and
  0.58 (RF 3, CF 3); intervals are wide outside CF = 1.

## 3.5 RQ3 – Does causal concept learning give more faithful, physically interpretable predictions?

**It makes the model's own explanation measurably more faithful by the alignment criteria, at no cost in
accuracy relative to its backbone. It does not make the prediction more accurate than gradient boosting,
and it does not disentangle the concepts.**

Model-internal alignment, geometry-grouped, mean over fold x seed runs (Table 5; SD over runs):

| Variant | RMSE | necessity: |dY| on null-slot swap | sufficiency: active-only vs output | effect share per null-state slot | routed through changed slot (matched) |
|---|---|---|---|---|---|
| CCW-Net (full) | 0.590 | 0.033 +- 0.008 | 0.098 +- 0.014 | 3.2 % +- 0.5 | 0.138 +- 0.012 |
| unaligned wrapper | 0.597 | 0.055 +- 0.017 | 0.295 +- 0.051 | 4.6 % +- 0.9 | 0.137 +- 0.015 |
| no necessity | 0.605 | 0.042 | 0.102 | 3.5 % | 0.140 |
| no sufficiency | 0.593 | 0.042 | 0.218 | 4.5 % | 0.137 |
| no independence | 0.594 | 0.033 | 0.102 | 3.3 % | 0.139 |
| no interaction term | 0.594 | 0.040 | 0.111 | 4.1 % | 0.129 |
| scalar concepts (D = 1) | 0.597 | 0.059 | 0.396 | 3.8 % | 0.157 |
| no conditional counterfactual | 0.597 | 0.028 | 0.087 | 2.9 % | 0.140 |
| Gaussian mediator | 0.595 | 0.034 | 0.101 | 3.3 % | 0.140 |
| concept bottleneck | 0.605 | – | – | 9.6 % +- 1.5 | – |

* [M] Alignment lowers the effect of physically absent concepts on the prediction (necessity 0.055 ->
  0.033; null-slot share 4.6 % -> 3.2 %) and makes the "active concepts only" reconstruction three times
  closer to the model output (0.295 -> 0.098). The same ordering holds on the series-held-out scheme
  (necessity 0.046 -> 0.029, sufficiency 0.367 -> 0.147). The concept-bottleneck model routes three times
  more effect through absent concepts than CCW-Net, at similar concept accuracy (85 % vs 90 %): concept
  accuracy does not imply faithful use, as in the original paper.
* [M] What each term contributes: sufficiency is responsible for most of the sufficiency gain and part of
  the null-slot gain; necessity for the necessity score; independence only for the additivity residual
  (0.050 vs 0.098 without it) with no effect on the other scores; the explicit interaction term improves
  necessity and null-slot share slightly. Vector concepts matter for concept read-out (accuracy 90 % vs
  76 % scalar) and sufficiency.
* [M] **Negative results.** (i) The conditional flow is not needed: drawing counterfactuals by permuting
  batch rows, or from a conditional Gaussian, gives the same or slightly better scores. (ii) The matched
  label-swap estimator shows that when a concept's label changes between comparable tests, only 14 % of
  the output change passes through that concept's slot (10 % would be indiscriminate) for every variant.
  The slots are entangled: each one carries information about the whole input, because all slots are
  computed from one shared backbone latent and nothing in the losses restricts slot j to concept j.
  (iii) Unaligned wrapper and CCW-Net use more interaction than intended (interaction share 0.28 for the
  full model). (iv) Moving the freeboard slot to the next-higher freeboard state raises the prediction in
  29 % of cases (25 % unaligned): the representation-level direction is not reliably physical.
  (v) Alignment training had not fully converged in the 60-epoch budget (validation objective still
  falling; best epoch on average 54 of 60).
* [M] Concept-level effects (Table 6, Figure 6): zeroing the steepness, depth, freeboard and profile slots
  changes the prediction by 0.30, 0.30, 0.24 and 0.20 on average; null-capable concepts by 0.05–0.15 when
  present and 0.03–0.05 when absent. The largest explicit pair terms are steepness x depth (0.054),
  freeboard x steepness (0.045) and freeboard x depth (0.034) - physically plausible pairs, but these are
  magnitudes of model terms, not physical interaction strengths.
* [D]+[C] Physical consistency is a property of smoothness, not of concept learning: the plain MLP has
  fewer crest-raising reversals than CCW-Net (Section 3.6).
* Loss-weight sensitivity (one seed): x10 weights improve necessity (0.019), sufficiency (0.067) and
  null-slot share (2.5 %) with RMSE 0.595; x0.1 moves all scores toward the unaligned model. The default
  is therefore not at an optimum, and stronger alignment was not penalised in accuracy within this range.

## 3.6 RQ4 – Are physically inconsistent counterfactual predictions concentrated in particular RF/CF categories?

**They are concentrated in particular model classes, not in particular RF categories; for CF there is a
CF = 3 excess for tree models only.**

Type A, crest raised by up to one Hm0 at fixed water level and waves (Rc and Ac together), supported
counterfactual points only (5.6 % of counterfactual points fall outside the training support and are
excluded); a reversal is an increase of more than 0.02 log10 units (Table 7, Figures 7–8):

| Model | curves with >= 1 upward step | upward steps / valid steps | local reversal (0.05 Hm0) | global reversal (1 Hm0) | median / max rise |
|---|---|---|---|---|---|
| Random Forest | 24.2 % [22.3, 26.6] | 6.9 % | 8.2 % | 3.2 % | 0.06 / 0.57 |
| XGBoost | 28.2 % [25.8, 30.9] | 7.4 % | 9.7 % | 1.6 % | 0.06 / 0.79 |
| Monotone XGBoost | 0 | 0 | 0 | 0 | – |
| MLP | 3.9 % [2.7, 5.6] | 1.3 % | 0.2 % | 1.9 % | 0.10 / 0.61 |
| Concept bottleneck | 8.3 % | 2.7 % | 1.2 % | 2.4 % | 0.09 / 1.61 |
| Unaligned wrapper | 5.8 % | 2.1 % | 0.6 % | 2.6 % | 0.10 / 1.84 |
| CCW-Net | 5.3 % [4.0, 7.2] | 1.9 % | 0.5 % | 2.5 % | 0.10 / 2.08 |

8 992 valid curves and about 43 600 valid steps per model; curves and steps are different units.

* [C] More than a quarter of unconstrained tree-model response curves contain at least one step in the
  wrong direction, mostly small local steps (step artefacts of piecewise-constant models); global reversals
  over one wave height are rare for all models (1.6–3.2 %). Smooth networks have few local reversals but as
  many global ones. The other interventions behave similarly (wall raised, crest widened, rougher armour;
  Table 7); "rougher" is the least consistent for XGBoost (34 % of curves, 25 % local).
* [C]/[P] By RF (XGBoost): 27, 28, 29, 33 % of curves for RF 1–4; difference RF4 - RF1 = +5.9 points
  [-4.1, +16.9]; no association for any model after Holm correction except a small one for the concept
  bottleneck (Table 7c). By CF (XGBoost): 28, 21, 41, 31 %; CF 3 is higher (41 % [34, 47]). For CCW-Net and
  the MLP the highest rate is in CF 4 (12.5 % [7.8, 21.4] and 8.4 %), based on 210 curves from 7 series.
* [D] Type C, measured pairs (same sub-structure, same h, Hm0, Tm-1,0, beta within tolerance, higher
  crest; 3 906 pairs in 208 sub-structures; Table 7d): the *measurements* show more overtopping at the
  higher crest in 4.7 % [3.5, 6.0] of pairs, with no trend in RF (6.8, 4.2, 4.8, 5.5 %). Out-of-fold
  predictions reverse in 2.5–4.9 % of the same pairs; for pairs involving RF = 4 tests XGBoost reverses in
  17 % against 5.5 % measured. A model reversal is not evidence that a measurement is wrong, and a measured
  reversal of this size is within the repeat scatter (repeat SD 0.2–0.35).
* [M] Type B (representation interventions) is reported in Section 3.5 and is not comparable with type A.

## 3.7 RQ5 – Can hierarchical analysis separate test-level, geometry-level and source effects?

**Geometry-level and within-geometry components are separated with useful precision; a series component
is poorly determined; a laboratory component cannot be estimated.**

REML decomposition of geometry-grouped XGBoost residuals (Table 9, Figures 4 and 10):

| Component | units | SD | share of variance |
|---|---|---|---|
| test series | 162 | 0.155 | 11 % [1, 27] |
| geometry within series | 2 046 | 0.260 | 30 % [20, 40] |
| within-geometry residual | 9 333 | 0.365 | 59 % [50, 65] |

* [D] Stable to the grouping definition (coarse geometry 26 % / sub-structure 28 % geometry share;
  residual 60–61 %).
* [D] Under random-record cross-validation the geometry share falls to 12 % and the series share to 1 %:
  the model memorises structure-specific offsets, which is exactly the leakage that grouped validation
  removes.
* [D] The decomposition depends on the model: for the MLP and CCW-Net the series component is 37–40 %
  (they generalise worse across series). The "model-induced" part is therefore not a fixed property of the
  data; the difference between models at a level is the model-induced error at that level.
* [N] Laboratory: no laboratory identifier exists. With laboratories inferred from reference keywords
  (12 laboratories, 6 514 records), the laboratory variance is estimated at zero and the series share at
  2 % [0, 33]. With 12 units, heavy confounding with structure type, and 30 % of records unassignable, this
  is absence of evidence, not evidence of no laboratory effect. A reference-level component is 1 %.
* [P] By CF: the geometry SD is the same in every CF class (0.25, 0.26, 0.26, 0.20); CF = 3 differs in its
  *within-geometry* residual (0.51 [0.30, 0.71] against 0.35) - the opposite of the expected pattern, in
  which CF would track the geometry-level component.
* [P] By RF: the within-geometry SD is 0.32, 0.39, 0.35, 0.42 for RF 1–4 (expected direction between RF 1
  and RF 4, intervals overlap in between), and the geometry SD is also larger for RF 4 (0.35 [0.27, 0.49]
  against 0.25). RF therefore relates to both levels, not only to the test level.
* [D] Model-free repeatability variance from near-replicates is 0.080 (SD 0.28 [0.21, 0.36]), about 60 % of
  the within-geometry residual variance of 0.133. The within-geometry residual is consequently **not**
  pure measurement error, and near-replicates are themselves an upper bound on repeatability because
  unrecorded differences (wind, spectral shape, measurement set-up) may exist between them.

## 3.8 RQ6 – Can RF/CF improve calibration without entering the mean predictor?

**Yes for RF, modestly: RF/CF-informed interval widths remove the under-coverage of RF = 4 tests. The gain
comes from interval width; the mean predictor is untouched by construction.**

90 % intervals, geometry-grouped split conformal with group-weighted calibration, 3 calibration splits
(Table 11, Figure 11):

| Method | coverage (records) | coverage (mean over geometries) | mean width | RF 1 / 2 / 3 / 4 | CF 1 / 2 / 3 / 4 |
|---|---|---|---|---|---|
| constant width | 85.4 % [82.7, 87.7] | 89.8 % | 1.43 | 89 / 85 / 83 / **78** | 86 / 89 / 78 / 93 |
| scale from physical inputs | 89.2 % [86.8, 91.1] | 90.1 % | 1.75 | 93 / 87 / 89 / 82 | 90 / 89 / 79 / 93 |
| scale from RF, CF only | 85.5 % | 90.0 % | 1.45 | 87 / 85 / 83 / 87 | 86 / 86 / 82 / 88 |
| scale from physical + RF, CF | 89.3 % [87.2, 91.1] | 90.0 % | 1.75 | 92 / 88 / 89 / **88** | 90 / 88 / 83 / 92 |
| Mondrian by RF | 86.2 % | 90.5 % | 1.53 | 89 / 83 / 84 / 94 | – |
| conformalised quantile regression | 89.7 % | 90.3 % | 2.65 | 88 / 91 / 93 / 80 | 90 / 90 / 88 / 89 |
| Gaussian, uncalibrated | 76.5 % | 76.9 % | 1.16 | 82 / 74 / 75 / 68 | – |

* [D] The guarantee of group-calibrated conformal prediction is for a test drawn from a new geometry with
  geometries weighted equally; that quantity is 90 % for every conformal method, as designed. Record-level
  coverage is lower for constant-width intervals (85 %) because a few large geometries are harder than
  average. No per-category guarantee is claimed, except within-category calibration for Mondrian variants.
* [P] Adding RF/CF to the scale model leaves overall coverage and width unchanged and reduces the largest
  deviation across RF categories from 7.7 to 2.4 points (RF 4: 82 % -> 88 %, with RF-4 intervals widened
  from 2.16 to 2.70). Across CF the largest deviation falls from 10.8 to 7.3 points; CF = 3 stays
  under-covered (83 %) with every method except quantile regression, which pays with 50 % wider intervals.
* [P] The worst adequately supported cell is (RF 3, CF 3) or (RF 2, CF 3) at 64–76 % for the
  absolute-residual methods; worst structural family 82–86 %.
* [D] Uncalibrated heteroscedastic intervals under-cover badly (77 %): out-of-geometry errors are larger
  than in-sample variance models suggest.
* [M] The CCW-Net quality branch (variance heads on the frozen network, Table 5b) agrees in direction: the
  eight-parameter RF/CF head has a better held-out likelihood than a constant variance (0.904 vs 0.916),
  whereas variance heads fed with concept vectors overfit (1.07) and under-cover (82 %).

## 3.9 RQ7 – Does keeping RF = 4 and CF = 4 records provide information that filtering discards?

**Yes. Removing them does not improve predictions for the remaining tests and makes predictions for the
removed kind of test much worse.**

Change in RMSE relative to training on all records (XGBoost; same folds, same test records; Table 12,
Figure 12):

| Training set | tested on RF<4 and CF<4 (n = 8 494) | tested on RF = 4 (n = 637) | tested on CF = 4 (n = 212) |
|---|---|---|---|
| excluding RF = 4 | +0.002 [-0.005, +0.011] | **+0.277** [+0.186, +0.378] | +0.012 |
| excluding CF = 4 | +0.012 [+0.004, +0.021] | -0.009 | **+0.236** [+0.169, +0.323] |
| excluding RF = 4 or CF = 4 | +0.001 [-0.006, +0.008] | +0.287 [+0.185, +0.394] | +0.248 [+0.178, +0.335] |
| CLASH weights (4-RF)(4-CF) | +0.006 [-0.004, +0.017] | +0.288 | +0.274 |
| soft weights (5-RF)(5-CF), nothing removed | +0.001 [-0.007, +0.008] | +0.015 [-0.015, +0.058] | +0.028 [+0.005, +0.051] |

* [D] Conventional filtering buys nothing measurable on the "clean" tests (differences of 0.001–0.012,
  intervals including or touching zero) and costs 0.24–0.29 on category-4 tests (RMSE 0.62 -> 0.90 for
  RF 4; 0.33 -> 0.58 for CF 4). Random Forest gives the same picture (+0.003 / +0.21 / +0.21).
* [D] The category-4 records occupy parts of the input space that the other records do not cover (depth-
  limited conditions with Hm0 > h, emergent toes, the chute configurations). Their value is coverage of
  those conditions, not extra precision elsewhere.
* [P] They are noisier (Section 3.2), which is handled by wider intervals (Section 3.8), not by deletion.
* Caveat: this holds for flexible learners that can localise noisy data. It does not show that category-4
  records are accurate, and "RF = 4 predicted from other RF = 4 tests" partly reflects shared series.

## 3.10 What the study does and does not establish

| Statement | Class |
|---|---|
| Random-split accuracy on CLASH overstates accuracy for new structures; grouping changes RMSE from 0.35 to 0.49–0.83 | D |
| RF = 4 tests have about 2.4 times the residual variance of RF = 1 tests; RF 2 and RF 3 are not separable by error | P |
| CF is not an ordered predictor of error or of generalisation difficulty; CF = 3 is the difficult class; CF = 4 rests on 7 series | P |
| Alignment losses make the concept wrapper's explanation more faithful by necessity and sufficiency scores | M |
| The concept slots are not disentangled; label changes are not routed through the matching slot | M |
| Unconstrained tree models reverse the freeboard response locally in about a quarter of curves; reversals are not tied to RF | C |
| 59 % of out-of-geometry residual variance is within geometry; near-replicate scatter explains about 60 % of that | D |
| RF/CF-aware interval widths equalise coverage across RF without touching the mean model | P |
| Filtering category 4 does not help clean tests and harms category-4 predictions | D |
| RF or CF are correct ratings; RF measures measurement error; any laboratory effect | N |

Notes on sign conventions: `bias` in Table 3 is prediction minus measurement; `bias` in Table 8a is
measurement minus prediction (the residual).
