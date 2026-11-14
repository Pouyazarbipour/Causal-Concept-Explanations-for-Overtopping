# Overlap with the two previous papers (Phase XII)

**What could be checked.** Neither previous paper is available in this environment:
- no full text;
- no repository, figures, out-of-fold predictions or configuration.

The assessment below therefore rests on:
- the contribution lists in the revision brief;
- the content of this repository.

**It is not a check of duplicated text, figures or numbers.** The authors must compare the final manuscript
against both full texts and both code bases before submission. This document extends report 06
(`results/reports/06_originality_and_overlap_assessment.md`).

## 1. Distinct contribution of the present paper

The empirical validity of the CLASH reliability (RF) and complexity (CF) ratings, and the counterfactual
consequences of RF/CF-based data-curation decisions. The paper consists of:
- predictive-dispersion, repeatability and generalisation evidence for RF and CF;
- a hierarchical error decomposition;
- nine training-data policies on identical held-out tests, with fixed and with policy-specific tuned
  learners;
- quality-aware conformal intervals as the alternative to exclusion;
- an engineering decision-impact analysis;
- an audit of concept-based explanations (CCW-Net).

## 2. Shared data, methods and outputs

| Element | Paper 1 | Paper 2 | Present paper | Overlap and how it is handled |
|---|---|---|---|---|
| CLASH database | yes (assumed) | yes (assumed) | yes, all 10 532 records audited | shared data; legitimate. Record accounting here is new (S_accounting tables) |
| Geometry reconstruction | yes (geometry-grouped validation) | unknown | yes, used to build leakage-free folds and as a variance level | **possible methodological overlap**. If identical to Paper 1, cite Paper 1 for the procedure; the tolerance sensitivity (G1) is specific to this paper |
| Geometry-grouped CV | central contribution | unknown | instrument only (not a highlight, not claimed) | moderate; framed as a means to obtain residuals |
| Conformal prediction | central contribution (coverage) | uncertainty quantification | quality-aware scaling by RF/CF; coverage by quality category; sensitivity to calibration size | moderate. New only if Paper 1 does not stratify by RF/CF; the authors must check |
| Weighting / scaling | not known | scaling and weighting discussed | conventional (4−RF)(4−CF) and empirical inverse-variance weights evaluated as **policies** | generic sample weighting is NOT claimed as a contribution; the contribution is measuring what RF/CF-based weighting changes |
| XGBoost, RF, CatBoost, MLP | likely | likely (classical baselines) | diagnostic instruments | no accuracy claim; v2 optimisation reported as a robustness check, not as a contribution |
| SHAP / Sobol | — | yes | grouped TreeSHAP only as the comparator for CCW-Net | low; no Sobol, no driver analysis |
| DML / causal forests | — | yes (physical causal effects) | none; CCW-Net effects are model-internal | terminology: the manuscript states that no physical causal effect is estimated |
| Quantum ML | — | yes | none | none |
| Freeboard monotonicity, freeboard response curves | yes | — | none | none |
| Figures and tables | unknown | unknown | all generated from this repository's outputs | duplication cannot be checked here; the authors must compare |
| Out-of-fold predictions | unknown | unknown | `results/predictions`, `revision/outputs` | the authors must confirm that no prediction file is reused across papers |

## 3. What could not be checked

- Duplicated figures, tables, numbers or text (no access to the full texts).
- Whether Paper 1 reports errors or coverage by RF/CF, or any category-4 sensitivity. If it does, the
  corresponding material here must be cut to a citation.
- Whether Paper 2's weighting or scaling is RF/CF-based.
- Bibliographic details of both papers (placeholders in the reference list).

## 4. Claims that must not be made

- Complete novelty, or absence of overlap.
- That geometry-grouped validation, conformal coverage or sample weighting is introduced here.
- That RF/CF ratings are shown to be correct or incorrect.
- That CCW-Net or any analysis identifies physical causal effects.
- That any model here is more accurate than existing overtopping predictors in general. The EurOtop
  comparison is restricted to its stated domain and is not independent, because EurOtop was partly
  calibrated on CLASH.
