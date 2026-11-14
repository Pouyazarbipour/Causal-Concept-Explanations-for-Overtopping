# 4. Manuscript outline (Ocean Engineering)

*Typeface for the manuscript, tables and figures: Times New Roman.*

**Working title.** Should category-4 tests be removed? Predictive validation of the CLASH reliability and
complexity ratings and counterfactual assessment of data-selection policies for wave-overtopping models

**Central contribution.** A validation of the CLASH data-quality ratings as decision tools, not a new
predictor. (i) Do RF and CF measure what they are used for? (ii) What do the conventional exclusion and
weighting policies actually change on the same held-out tests? (iii) Can test-level and geometry-level error
be separated, and do RF and CF map onto them? (iv) Are concept-based explanations of the learned
representations faithful, and do they depend on data quality?

## Highlights (draft, ≤ 85 characters each)
* All 10 532 CLASH tests, RF and CF 1–4, audited and retained.
* RF = 4 tests carry 2.3 times the residual variance of RF = 1; RF 2–3 do not differ.
* CF shows no ordered link to geometry-level error or generalisation.
* Excluding category 4 leaves other tests unchanged and degrades category-4 predictions by 0.22–0.31.
* Keeping all tests and widening intervals by RF/CF equalises coverage.

## Abstract (structure, ~250 words)
Problem (ratings used as filters and weights without validation); data (original database, all
categories); approach (leakage-free out-of-fold residuals, ordinal and hierarchical statistics, nine
counterfactual training policies, regression CCW-Net as an explanation audit); main numbers (Section 3 of
the interpretation report); conclusion (retain, do not exclude; use RF for uncertainty, not for the mean).
Define "counterfactual" as an alternative training-data policy in the first use.

## 1 Introduction
1.1 The CLASH database and its two ratings; their role in neural-network training (weights
(4−RF)(4−CF), removal of category 4 and of remarked tests) and in later data-driven models.
1.2 The gap: the ratings are used, not tested (Section 6 report). Why it matters: removing tests removes
coverage of conditions (depth-limited waves, unusual sections).
1.3 Scope relative to previous work. Grouped validation, monotonic constraints, conformal intervals and
physical causal-effect estimation are not contributions here (cite Paper 1 and Paper 2).
1.4 Research questions RQ1–RQ6.

## 2 Data
2.1 Database, variables, response classes (Table 1). 2.2 RF and CF: documented meaning, direction, 4 × 4
matrix (Table 2, Figure 2). 2.3 Internal consistency: CF varying within reconstructed structures;
indicators of estimated parameters. 2.4 Identifiers available, reconstructed, inferred.

## 3 Methods
3.1 Leakage-free out-of-fold residuals: geometry-grouped nested CV, Optuna, five diagnostic models
(Table 4). 3.2 Ordinal validation of RF and CF, repeatability from near-replicates, geometry-level
analysis. 3.3 Hierarchical decomposition. 3.4 Counterfactual data-selection policies P1–P6 with a fixed
evaluation design. 3.5 Regression CCW-Net and explanation-quality measures; comparison with grouped
TreeSHAP. 3.6 Statistics.

## 4 Results
4.1 Diagnostic models (Table 3; Figures 3–7) – brief.
4.2 RF validity (Tables 5a–h; Figures 8, 9).
4.3 CF validity (Tables 6a–b).
4.4 Error decomposition and the levels of RF and CF (Table 8; Figure 10).
4.5 Data-selection policies (Tables 13–15; Figures 13–15) – main result.
4.6 Concept explanations and data quality (Tables 9–12; Figures 11, 12).

## 5 Discussion – by evidential status
**5.1 Demonstrated by the data.** Effect of each policy on identical held-out tests; repeatability of
near-replicates; variance shares; CF inconsistency; thin support of CF = 4.
**5.2 Predictive association only.** RF 4 vs 1 variance ratio; no ordered CF effect; CF = 3 excess; coverage
equalisation. These describe how the ratings relate to error, not whether they are correct.
**5.3 Model-internal explanations.** Alignment criteria, entanglement, disagreement with SHAP, explanation
quality by RF. Statements concern network representations only.
**5.4 Not identifiable without independent evidence.** Correctness of individual ratings; pure measurement
error; laboratory effects; inter-rater agreement. Proposed re-rating design.
**5.5 Engineering recommendations.**
  * Retain category-4 tests in the mean model.
  * Do not use (4−RF)(4−CF) weights.
  * Use RF (and CF = 3) to widen prediction intervals.
  * Report errors for RF = 4 and CF = 3 separately.
  * Treat CF as a flag, not as an ordered weight.

## 6 Limitations (from `07_limitations.md`)
## 7 Conclusions – one paragraph per RQ, with the numbers and the acceptance-criteria table.

## Main-text tables and figures
Tables 2, 3, 5a, 6a, 8, 9, 13 (with Table 4 as optimisation summary). Figures 1, 2, 3, 8 or 9, 10, 12, 13, 14.
The remaining tables and figures go to the supplementary material (`05_supplementary_material_outline.md`).

## Before submission
* Verify the RF/CF category wording against the CLASH WP2 report.
* Compare against the full texts of Paper 1 and Paper 2 (Section 6 report, 6.7).
* Complete a systematic literature search.
* Consider the blinded re-rating study (L2).
