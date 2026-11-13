# 4. Proposed manuscript outline (Ocean Engineering)

**Working title.** Do the CLASH reliability and complexity factors mean what we use them for?
A geometry-aware audit with concept-based and counterfactual diagnostics of wave-overtopping predictors

**Central message.** The usual practice of discarding RF = 4 / CF = 4 tests is not supported by predictive
evidence in the original database: it does not improve predictions for the retained tests and removes the
only information on conditions those tests cover. RF carries real but coarse information about residual
dispersion (useful for interval width, not for the mean); CF does not behave as an ordered difficulty
scale. Concept-based and counterfactual tools are used as auditing instruments, not as a new
best-performing predictor.

## Highlights (draft)
* All 10 532 CLASH tests retained; validation by reconstructed geometry, test series and structural family.
* Random-split RMSE (0.35 log10) understates out-of-geometry error (0.49) and out-of-series error (0.76).
* RF = 4 tests have about 2.4 times the residual variance of RF = 1 tests; intermediate RF steps and CF are not ordered by error.
* Filtering category 4 changes clean-test RMSE by 0.001 and raises category-4 RMSE by 0.25–0.29.
* Causal alignment makes a concept wrapper's explanations more faithful but does not disentangle concepts.

## 1 Introduction
Overtopping databases and data-driven predictors; the role of RF/CF as weights and filters in CLASH-type
neural networks; lack of any published check of what the ratings predict; record-level validation and
leakage between tests of the same structure; concept-based explanation and counterfactual auditing as
diagnostic tools. Research questions RQ1–RQ7. Contribution statement limited to: an auditing framework
and an evaluation of RF/CF. No claim of a first causal concept model in engineering is made (no dedicated
literature review supports it).

## 2 Data
2.1 The original database (Table 1, Figure 2). 2.2 Response classes and two-part treatment of zeros.
2.3 RF and CF: documented meaning, distribution, thin cells, within-structure inconsistency (Table 2).
2.4 Identifiers that exist (series, reference) and those reconstructed or inferred (geometry, laboratory).

## 3 Methods
3.1 Dimensionless predictors and target. 3.2 Geometry reconstruction and six validation schemes; nested
tuning; leakage checks. 3.3 Baseline learners. 3.4 Concept ontology (Table 4). 3.5 Regression CCW-Net:
architecture, mediator, loss terms, ablations (Figure 5). 3.6 Counterfactual auditing: input interventions
with support diagnostics, representation interventions, matched measured pairs; curve- and step-level
metrics. 3.7 RF/CF validity analyses: ordinal trends, variance model, repeatability from near-replicates,
geometry-level analysis. 3.8 Hierarchical error decomposition (REML). 3.9 Group-calibrated conformal
intervals with and without RF/CF. 3.10 Statistical treatment: cluster bootstrap, Holm correction.

## 4 Results
4.1 Predictive performance across validation schemes (Table 3).
4.2 RF: residuals, dispersion, repeatability, completeness indicators (Tables 8a–c, 8f, 8h; Figures 3, 9).
4.3 CF: geometry-level error and generalisation (Tables 8d–e).
4.4 RF x CF (Table 10).
4.5 Error decomposition (Table 9; Figures 4, 10).
4.6 Concept alignment and ablations (Tables 5, 6; Figure 6).
4.7 Counterfactual audit (Table 7; Figures 7, 8).
4.8 Quality-aware uncertainty (Table 11; Figure 11).
4.9 Category-4 filtering (Table 12; Figure 12).

## 5 Discussion – organised by evidential status
**5.1 Directly demonstrated by the data.** Dependence of error on validation unit; filtering result;
variance shares by level; near-replicate scatter by RF; CF inconsistency within structures; thin support
of CF = 4.
**5.2 Supported only by predictive association.** RF = 4 vs RF = 1 variance ratio; absence of an ordered
CF effect; CF = 3 excess; coverage equalisation with RF/CF-aware widths. These say how the ratings relate
to model error, not whether the ratings are right, and not that RF or CF cause anything.
**5.3 Model-internal causal explanations.** Necessity, sufficiency, independence and interaction scores;
ablation effects per concept; entanglement shown by the matched estimator. Statements are about a network's
representations only.
**5.4 Physically meaningful counterfactual interventions.** Crest raising, wall raising, crest widening,
roughness: reversal rates of fitted models inside the training support. These describe model behaviour
under a defined input change; they are not estimates of physical effects and not evidence against any
measurement.
**5.5 Not identifiable without independent experimental evidence.** Correctness of individual RF/CF
assignments; pure measurement error; laboratory effects; inter-rater reliability; behaviour of
structural families absent from CLASH.
5.6 Practical implications: validate by structure or series; keep category-4 tests and widen intervals;
use RF as a variance covariate; do not treat CF as an ordered weight; use monotone constraints where a
guaranteed direction is required (cost 0.03 RMSE).

## 6 Limitations (from `05_limitations.md`)
## 7 Conclusions – answers to RQ1–RQ7 with the numbers of Section 3 of the interpretation report.

## Figures and tables
Figures 1–12 (`results/figures/`, PDF and PNG). Tables 1–12 (`results/tables/`); suggested main-text
tables: 2, 3a, 5, 7, 8a, 9, 11, 12; the rest as supplementary material.

## Before submission
* Verify RF/CF category wording against the CLASH WP2 report.
* Add the literature review (CLASH/EurOtop ANN lineage, weighting by RF/CF, grouped validation in
  coastal ML, concept bottleneck and CCW-Net literature).
* Consider an independent blinded re-rating of a stratified sample of series (design in limitations, L3).
* Obtain the database owners' confirmation of the release and of the confidential-data status.
