# 5. Unresolved methodological limitations

**Data and documentation**

L1. No CLASH documentation is in the repository. RF/CF category wording is from memory of the published
reports and must be verified; variable definitions follow the workbook header and standard CLASH usage.

L2. The workbook cannot be proven identical to the 2005 release.

L3. No independent quality information exists (laboratory reports, protocols, second ratings). Inter-rater
reliability (weighted kappa, Krippendorff's alpha) and criterion validity against an external standard
could not be computed; the functions exist (`statistical_analysis.weighted_kappa`) but were not applied to
invented data. A feasible design: stratified sample of about 40 series over the RF x CF cells with
references available, two or more experts rating blind to the original values from the source reports.

L4. No detection limit is recorded; zero and below-detection overtopping are indistinguishable. A censored
model was not fitted. The wet/dry model is evaluated but not combined with the regression into an
expected-discharge predictor.

**Grouping**

L5. Geometry is reconstructed from rounded descriptors within a series. It can split one model or merge
two. Half of the fine groups contain one test, so the geometry and within-geometry components are
identified mainly by the larger groups.

L6. Geometry-grouped folds still share test series and laboratories between training and test data. The
series-held-out scheme is the stricter estimate of transfer to a new investigation (RMSE 0.76).

L7. Laboratories are inferred from reference keywords for 70 % of records; the laboratory variance
component is not estimable (12 units, confounded with structure type).

L8. CF varies inside 32 reconstructed geometries; geometry-level analyses use the maximum, which is a
choice. CF = 4 comes from 7 series; six RF x CF cells are too thin to interpret.

**Modelling**

L9. Predictors are Froude-scaled; model and prototype tests are pooled and scale effects are not modelled.
Wind tests and prototype measurements are retained (a sensitivity run without the 509 "do not use" records
is in Table 12).

L10. The concept ontology is one defensible choice. Bin edges are round engineering values; three states
have fewer than 310 records. Roughness gamma_f is itself an assigned value.

L11. The CCW-Net does not disentangle concepts (matched routing 14 % against 10 % chance). All slots are
computed from one shared latent; per-concept input masking or an explicit disentanglement term would be
needed and was not tried. The necessity criterion covers only the six null-capable concepts.

L12. Alignment training was not run to convergence (60 alignment epochs); loss weights were examined at
three levels with one seed. Ablations use two seeds (ten runs), the main variants three (fifteen runs);
fold-by-seed runs are not independent, so SDs and paired intervals in Table 5 are descriptive.

L13. The concept models use one pre-specified architecture (no nested tuning), unlike the tree models and
the MLP. The 0.10 RMSE gap to XGBoost may partly reflect that.

L14. The mediator model's counterfactual draws for inputs whose concept is absent are extrapolations by
construction, as in the original method.

**Auditing and statistics**

L15. Counterfactual audits evaluate fitted models at unobserved inputs. Support is checked by range and
k-nearest-neighbour distance (99th percentile), which is a heuristic. The 0.02 tolerance is a choice;
rates change with it. Curves for neural models are averaged over three seeds' individual curves.

L16. The "rougher" and "crest wider" interventions assume a sign that holds for sloping structures in
general but not necessarily in every regime.

L17. Near-replicate pairs are defined by a 1 % tolerance on five recorded variables; unrecorded
differences make them an upper bound on repeatability. Only 66 pairs involve RF = 4.

L18. Residual-based RF/CF analyses depend on the reference model. Trends were repeated for four learners
with the same conclusions, but a common misspecification would affect all.

L19. Bootstrap intervals resample test series (162) or geometries; with few series in some strata
(7 for CF = 4) the intervals are themselves unstable. Bootstrap p-values have a resolution of 1/B.
A mixed-effects ordinal model was not available; cluster-bootstrapped proportional-odds models were used,
and the proportional-odds assumption was not tested.

L20. Conformal coverage is guaranteed marginally over new geometries (equal weight per geometry), not per
record, category or structural family; calibration and test geometries share series.

L21. The filtering experiment shows that category-4 records help predict category-4 records; part of that
is shared series. It does not show that those records are accurate.
