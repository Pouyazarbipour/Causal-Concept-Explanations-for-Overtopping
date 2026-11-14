# 7. Unresolved methodological limitations

**Evidence base**

L1. No CLASH documentation is in the repository. The RF/CF scale direction (1 best, 4 worst) is confirmed
from secondary sources; the per-category wording follows the published CLASH reports as recalled and must
be checked against the CLASH WP2 report.

L2. No independent quality evidence exists: no laboratory reports, measurement protocols, second ratings or
expert re-assessment. Criterion validity and inter-rater reliability (weighted kappa, Krippendorff's alpha)
could not be measured; the functions exist but were not applied to invented data. The study establishes
predictive validity and internal consistency only. A feasible design for the missing part: a stratified
sample of about 40 series across the RF × CF cells with available reports, rated blind by at least two
coastal-laboratory experts.

L3. The two previous papers were not available; the overlap assessment rests on the contribution lists in
the brief. The literature screening used three web searches (the scholarly-search quota was exhausted) and
is not a systematic review.

L4. No detection limit is recorded; zero and below-detection overtopping cannot be separated, and no
censored model was fitted. Policies are evaluated on positive discharges; the wet/dry model is reported
separately.

**Grouping and identifiers**

L5. Geometry is reconstructed from rounded descriptors within a series. It can split one model or merge
two; half of the fine groups contain one test.

L6. Geometry-grouped folds share test series and laboratories between training and test sets. The
series-held-out sensitivity (RMSE 0.75) is the better guide to transfer to a new investigation.

L7. There is no laboratory identifier. An inferred laboratory exists for 70 % of records; its variance
component is not estimable.

L8. CF varies inside 32 reconstructed geometries. CF = 4 comes from 7 series (one chute investigation
dominates); six RF × CF cells are too thin to interpret.

**Models and optimisation**

L9. Optuna budgets are modest (8–30 trials per fold; 6 for CCW-Net, on a shortened schedule), chosen for
runtime. The optimisation histories (Figure S1) flatten for the tree models but not fully for the MLP and
CCW-Net. The policies reuse the hyper-parameters tuned on the full training partition; re-tuning per policy
was not done, which could slightly favour the retain-all policy.

L10. Model and prototype tests are pooled under Froude scaling; wind tests and prototype records are kept
(excluding the 509 "do not use" records is policy P5b).

L11. The concept ontology is one defensible choice with round-number thresholds; gamma_f is itself an
assigned value. The CCW-Net does not disentangle its concept slots (see interpretation report).

**Statistics**

L12. Residual-based RF/CF analyses depend on the reference model; trends were repeated for four learners
and three validation schemes, but a misspecification common to all would affect all.

L13. Near-replicate pairs (1 % tolerance on five recorded variables) give an upper bound on repeatability;
only 66 pairs involve an RF = 4 test, and none support a CF = 4 estimate.

L14. Bootstrap intervals resample test series or geometries; with 7 series in CF = 4 they are unstable.
A mixed-effects ordinal model was not available; cluster-bootstrapped proportional-odds models were used
without testing the proportional-odds assumption.

L15. Conformal coverage is guaranteed only marginally over new geometries; per-category coverage is an
empirical result, not a guarantee. Calibration and test geometries share series.

L16. Policy effects on category-4 tests partly reflect shared series: a category-4 test is predicted better
when other tests of its series are in the training data. This shows that the records are informative for
similar records, not that they are accurate.
