# Revised manuscript structure

Main paper: 13 sections, about 11 figures and 9 tables. Every number comes from `results/tables` (original
analyses, re-verified) or from `revision/tables` (new analyses).

## Main manuscript

1. **Introduction.** How the ratings are used (CLASH NN weights, later exclusions), why that use is untested,
   six research questions. RQ3 is reworded: laboratory effects are not identifiable.
2. **CLASH data and quality ratings.** Mutually exclusive record accounting (Table 1, Fig. 1). Input
   unusability confined to RF = 4 (emergent toes). RF/CF distributions (Table 2). Identifiers, reconstructed
   geometries, CF variation within geometries.
3. **Methodology.**
   - 3.1 Response and predictors.
   - 3.2 Validation and optimisation (baseline reproduced; v2 budgets and learners).
   - 3.3 Two-stage occurrence and discharge model.
   - 3.4 RF/CF validity.
   - 3.5 REML decomposition.
   - 3.6 Policies (Table 3; fixed and policy-specific tuned).
   - 3.7 Quality-aware conformal intervals.
   - 3.8 EurOtop and decision analysis.
   - 3.9 CCW-Net (short).
4. **Predictive model diagnostics.** Baseline vs optimised learners (Table 4, Fig. 2). Budget and
   partition sensitivity. Validation unit. Two-stage occurrence model (Fig. 3).
5. **RF and CF validity.**
   - RF table (Table 5) and robustness across learners, schemes and contrasts (Fig. 4).
   - Repeatability tolerance sensitivity.
   - CF geometry table (Table 6), power, leave-one-series-out, CF = 3 series concentration, series-held-out
     trend.
   - Residual distributions (Fig. 5).
6. **Hierarchical error decomposition.** REML table (Table 7, Fig. 6), geometry-definition sensitivity,
   replicate ratio with CI, laboratory component not identifiable.
7. **Counterfactual data-selection policies.** Fixed vs tuned (Table 8, Fig. 7), ESS and weights,
   stability, two-stage occurrence under exclusion.
8. **Quality-aware uncertainty quantification.** Six scale functions (Table 9, Fig. 8), calibration
   sensitivity, CF = 3 under-coverage explained, retrospective vs prospective availability of RF and CF.
9. **Engineering decision implications.**
   - EurOtop benchmark within its domain (Table 10).
   - Exceedance decisions on the q_allow × Hm0 grid.
   - Safety-class changes.
   - Regimes removed by exclusion (Fig. 9).
10. **CCW-Net concept-explanation assessment.** Summary of the negative results. Full analysis in the SI.
11. **Discussion.** Demonstrated / predictive association / model-internal / not identifiable.
12. **Limitations.**
13. **Conclusions.**

Placement recommendation. Keep the full CCW-Net analysis (ablations, attribution, interaction terms,
reference-code check) in the Supplementary Information. It answers RQ5, but it gives no engineering
evidence about the ratings, and placing it there protects the main narrative. All results stay available
and reproducible.

# Supplementary Information outline

- **S1 Record accounting.**
  - Overlapping exclusion reasons (`S_accounting_overlap`).
  - Reconciliation with the manuscript counts (`S_accounting_reconciliation`).
  - Missingness by RF, CF, series and response.
  - Representativeness of RF = 4 records with unusable inputs.
  - Deep-water substitution sensitivity (R7).
- **S2 Variable audit.** Source, unit, transformation, range, missingness. Recomputation checks of q*, Y,
  L0 and s0. Physical flags (depth-limited, Ac > Rc, default period ratios).
- **S3 Models.**
  - Search spaces and per-fold selections (baseline and v2).
  - Fold-mean vs pooled RMSE (explains the original Table 2 / Table 4 difference).
  - Budget sensitivity (M3).
  - Repeated partitions and series-held-out (M4).
  - RMSE by RF/CF for every learner (M2).
- **S4 Two-stage model.** Stage-A metrics by RF/CF, confusion matrices, reliability (W1–W2). Combined
  expected discharge (W3). Stage-B reproduction check (W5).
- **S5 RF/CF validity.**
  - Ordinal models, trend tests, Gamma GLM with interaction (original tables 05b–05g).
  - Variance ratios for every learner and scheme (R1).
  - RF contrasts (R2).
  - Repeatability tolerance and disjoint pairs (R3).
  - CF leave-one-series-out (R4).
  - Minimum detectable ratios (R5).
  - CF-inconsistent geometries (R6).
  - RF × CF joint table.
- **S6 Decomposition.** Grouping and model sensitivity (original Table 8). Geometry-definition sensitivity
  (G1). Replicate ratio CI (R3b).
- **S7 Policies.**
  - Original fixed-parameter results (Tables 13–15, Random Forest).
  - v2 fixed and tuned results (P1).
  - Coverage by category (P2).
  - Policy hyper-parameters, ESS, weights.
  - Occurrence model under exclusion (W4).
- **S8 CCW-Net.**
  - Architecture (fig11), ablations (Table 9), attribution (Tables 10–11, fig12).
  - Explanation quality vs RF/CF (Table 12).
  - Quality layer.
  - Interaction pairs.
  - Reference-implementation check (report 08).
- **S9 Conformal.** Ten original methods (tableS_conformal). Six v2 scales × three calibration fractions
  (C1–C3). CF = 3 errors by series (C4). Coverage by geometry and series.
- **S10 Engineering.**
  - EurOtop domains and counts (E1, E1b).
  - Exceedance decisions on the full threshold grid (E2).
  - Risk ranking and safety classes (E3).
  - Regimes and input-space coverage (E4–E5).
- **S11 Audit.** Repository audit, numerical-consistency audit, reference audit, overlap with previous papers.
