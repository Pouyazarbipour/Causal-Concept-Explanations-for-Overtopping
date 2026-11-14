# Revision log: original draft → revised manuscript

The original draft (`original/original_manuscript/Manuscript_CLASH_data_quality.docx`, commit in
`original/ORIGINAL_COMMIT.txt`) is preserved. The revised manuscript is built by
`manuscript/revision/build_revised.py` from verified outputs only.

**Rule followed throughout.** No test was removed from any analysis to lower an error. The train/test split
was not altered: the pre-specified geometry folds are used for every primary result, and alternative
partitions are reported only as sensitivity analyses. All 9 333 regression tests (RF and CF 1–4) and all
10 389 wet/dry tests are retained in every primary analysis.

## 1. Structure

| Original | Revised |
|---|---|
| 1 Introduction, 2 Data, 3 Methods, 4 Results (4.1–4.6), 5 Discussion, 6 Limitations, 7 Conclusions | 1 Introduction, 2 CLASH data and quality ratings, 3 Methodology, 4 Predictive model diagnostics, 5 RF and CF validity, 6 Hierarchical error decomposition, 7 Counterfactual policies, 8 Quality-aware UQ, 9 Engineering decision implications, 10 CCW-Net assessment, 11 Discussion, 12 Limitations, 13 Conclusions |
| Title "Should category-4 tests be removed? …" | Title of the revision brief |
| CCW-Net results in the main Results section | One-paragraph summary in Section 10. The full analysis moves to SI Section S8 and is retained unchanged |

## 2. Corrections of statements

| # | Original statement | Problem | Revised |
|---|---|---|---|
| 1 | Highlight "All 10 532 CLASH tests … audited and retained" | implies that all 10 532 enter the regression | audit (10 532), occurrence model (10 389) and regression (9 333) are distinguished |
| 2 | "all 143 tests with unusable inputs have RF = 4" | the 143 are all records outside the wet/dry set: 21 invalid responses plus 122 tests with unusable inputs (123 in total, one overlapping) | exact accounting (Table 1, Fig. 1) |
| 3 | "32 geometries (311 tests) carry more than one CF; RF varies within 166 geometries", next to the 2 046 regression geometries | these counts refer to all 10 532 records (2 209 geometries); the regression set has 27 / 283 / 140 | population stated; causes of CF variation analysed |
| 4 | Table 2 outer RMSE (fold mean) vs Table 4 RMSE (pooled) | two definitions; MLP 0.547 vs 0.576 also differs in seed averaging | pooled OOF RMSE everywhere; fold mean moved to the SI and labelled |
| 5 | CF = 3 "geometry RMSE 0.391" vs record-level 0.71 | different aggregation (unweighted geometry mean vs pooled records) and population (935 vs 825 tests) | both defined in the text; concentration in series 506 shown |
| 6 | Abstract/highlight: intervals "equalise coverage" | CF = 3 stays under-covered | "removes the RF under-coverage; CF = 3 remains under-covered" |
| 7 | RQ1 "measurement-associated uncertainty" | not identifiable from residuals | "predictive residual dispersion and near-replicate repeatability" |
| 8 | RQ2 "difficult to represent" | representation quality not verified | "predictive difficulty or poorer generalisation" |
| 9 | RQ3 "source effects" | laboratory effects not identifiable | test-series, geometry and within-geometry levels; laboratory stated as not identifiable |
| 10 | Cross-references "Fig. 4" (REML), "Fig. 5" and "Fig. 6" (policies) | wrong numbers | numbers generated at build time |
| 11 | CCW-Net cited as arXiv:2611.14407 | identifier not verifiable | NeurIPS 2026 citation (details to verify) |
| 12 | "Interval width 1.55 vs 1.54" | correct (P1 1.55, P6b 1.54); the brief listed the values in reverse order | unchanged values; v2 widths reported |

## 3. Method changes

| Area | Original | Revised (original kept in parallel) |
|---|---|---|
| Interval scale for P6 policies | σ²(RF, CF) fitted on residuals that include the calibration geometries | fitted on proper-training geometries only (`src/counterfactual_policies.py`, `src/conformal_uncertainty.py`) |
| Policies | fixed hyper-parameters only | plus policy-specific nested tuning; ESS and weight distribution; CF = 3 RMSE; Winkler score; partially pooled cell variances (P6b′) |
| Wet/dry | fixed hyper-parameters, own folds, AUC/Brier only | nested two-stage model on the regression folds; inner-CV threshold; calibration, confusion, combined E(q); exclusion effects |
| Models | budgets 8–30 trials | budgets 24–100 trials with pruning; LightGBM, histogram GB, ExtraTrees, physical features; three refit seeds; paired bootstrap vs baseline; budget and partition sensitivity |
| RF/CF validity | XGBoost Gamma GLM, trend tests across models | plus variance ratios for every learner and scheme, RF contrasts, power, CF leave-one-series-out, CF inconsistency causes, missing-input sensitivity |
| Repeatability | 1 % tolerance | plus 0.5–5 % tolerances, disjoint pairs, CI of the replicate / within ratio |
| Geometry | single key | plus 2 and 4 significant digits and coarse key (stability of folds, RMSE, shares, RF ratio) |
| Conformal | ten methods, one calibration fraction | six scale functions × three calibration fractions × five seeds; coverage by geometry and series; Winkler; CF = 3 diagnosis; prospective availability of RF/CF |
| Engineering | — | EurOtop benchmark within its validity domain; exceedance decisions on a q_allow × Hm0 grid; safety classes; regimes removed by exclusion |

## 4. Changed numbers

The changed numbers are in `revision/final_report.md` (Section A), side by side with the original values and
the reason for each change.
