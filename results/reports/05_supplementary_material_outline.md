# 5. Supplementary-material outline

**S1 Data audit.** File inventory; variable dictionary with ranges and missingness (Table 1); response
classes and unusable records with reasons; duplicate analysis; flag definitions and counts; RF × CF matrix
for all records and for the regression set; within-group consistency of RF and CF at five grouping levels;
list of geometries with more than one CF value; identifiers available, reconstructed and inferred.
Files: `table01*`, `table02*`, `diagnostics/data_audit_summary.json`, `cf_discrepancies_within_geometry.csv`.

**S2 Models and optimisation.** Search spaces and their justification; nested design diagram; per-fold best
hyper-parameters, inner objective and outer error (Table 4); Optuna histories (Figure S1); early-stopping
rules; performance on the sensitivity schemes (random records, series held out); wet/dry model (Table S).

**S3 Regression CCW-Net.** Mapping to the original method and list of adaptations; concept ontology with
thresholds and counts (Table 3); head equations and the zero-reference property; derivation of each loss
term; mediator model; training schedule; evaluation estimators (necessity, sufficiency, independence,
zero-ablation shares, matched label-swap Shapley estimator); interaction pairs (Table S); variance branch
(Table S); convergence diagnostics.

**S4 Additional RF/CF validity results.** Trend statistics for every model and validation scheme
(Table 5b); ordinal models (5c); variance model with interaction (5e); completeness indicators (5f);
zero-overtopping share (5g); repeatability by RF and CF (5h); geometry-level CF results (6a, 6b); RF × CF
cells with support flags (7); scatter plots stratified by RF and CF for all models (Figures 4, 5).

**S5 Error-decomposition robustness.** Grouping definitions; source and inferred-laboratory levels; other
models; random and series-held-out residuals; strata of RF and CF (Table 8, 8b).

**S6 Data-selection policies.** Full policy table for both learners (Table 13); calibration by category
(14); bias, RMSE and coverage by structural family (15); training variance model per fold (Table S);
alternative conformal constructions by quality category (Tables S: scale models, Mondrian, CQR).

**S7 Reproducibility.** Software versions, seeds, fold assignments, checksum of the raw workbook, unit
tests, execution guide.

**S8 Originality statement** (from `06_originality_and_overlap_assessment.md`).
