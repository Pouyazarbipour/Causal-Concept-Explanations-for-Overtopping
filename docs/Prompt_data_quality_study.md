# Research Prompt: An Independent Research Study on CLASH Data-Quality Validation

## Research objective

Develop a completely independent research paper for potential submission to *Ocean Engineering*, using the original, unfiltered CLASH wave-overtopping database, including all observations with Reliability Factor (RF) and Complexity Factor (CF) values from 1 to 4.

The research must investigate whether the original CLASH RF and CF ratings genuinely represent experimental reliability and structural complexity, respectively, and whether conventional exclusion or weighting policies based on these ratings lead to unnecessary loss of useful engineering information.

The central research contribution must be:

**Causal-concept-based validation of CLASH data-quality ratings, counterfactual assessment of data-selection policies, and separation of test-level uncertainty from geometry-level representation error.**

### 1. Mandatory separation from previous publications

Before implementation, inspect and document the research contributions of the following two papers:

**Paper 1 — Current wave-overtopping manuscript:**

*Machine-learning prediction of wave overtopping at untested structures: geometry-grouped validation, freeboard monotonicity and conformal coverage.*

Its contributions include:
- Geometry-grouped validation and generalization to unseen structures.
- Quantification of random-split optimism.
- Freeboard monotonicity constraints.
- Counterfactual freeboard-response diagnostics.
- Conformal prediction and uncertainty coverage.
- Crest-level design applications.

**Paper 2 — Previously published Ocean Engineering article:**

*Quantum machine learning for wave overtopping estimation: Integrating with causal inference and uncertainty quantification.*

Its contributions include:
- Quantum machine-learning prediction.
- SHAP and global sensitivity analysis.
- LinearDML and CausalForestDML.
- Causal analysis of hydraulic and structural predictors.
- Prediction uncertainty and model comparisons.

These two papers define strict research boundaries.

Do NOT reproduce their main contributions.

In particular:
- Do not make geometry-grouped validation itself a novelty claim.
- Do not develop another freeboard-monotonic prediction framework.
- Do not repeat crest-level design or conformal coverage analyses as the main contribution.
- Do not present SHAP, Sobol, DML, or CausalForestDML as the principal novelty.
- Do not merely introduce another machine-learning model with improved RMSE.

Geometry-grouped validation may be used only as a necessary methodological procedure to obtain leakage-free out-of-fold predictions for the new data-quality investigation.

The new research must answer a fundamentally different engineering question:

**Are CLASH data-quality ratings valid and useful for deciding which experimental observations should be retained, down-weighted, or excluded when developing engineering prediction models?**

Conduct a literature-based novelty audit before proceeding. Identify directly overlapping studies, distinguish the proposed contribution from existing CLASH filtering and weighting practices, and report any unresolved overlap. If a sufficiently distinct and defensible research gap cannot be established, revise the research scope rather than claiming artificial novelty.

### 2. Original CLASH database audit

Inspect the entire repository and identify the original, unfiltered CLASH database.

Retain all available observations, including:
- RF = 1, 2, 3, and 4.
- CF = 1, 2, 3, and 4.
- Observations with both RF = 4 and CF = 4.
- Zero-overtopping and positive-overtopping records, with appropriate response treatment.

Do not silently remove category-4 observations.

Audit missing values, duplicate records, data quality, laboratory identifiers, test-series identifiers, geometry descriptors, and available experimental documentation.

Verify the documented meaning and direction of RF and CF. Treat both as ordinal ratings, not continuous variables with necessarily equal category spacing.

Establish whether CF is consistent within repeated tests of the same physical geometry and whether RF varies across individual experimental tests.

Produce a complete RF × CF 4 × 4 frequency matrix, including observation counts and percentages.

### 3. Core research questions

Investigate the following:

**RQ1 — RF validity:** Does RF reliably distinguish experimental observations with different levels of measurement-associated uncertainty, residual dispersion, and repeatability?

**RQ2 — CF validity:** Does CF identify geometries that are difficult to represent using the standard CLASH parameters and that produce larger geometry-level generalization errors?

**RQ3 — Error decomposition:** Can test-level uncertainty, geometry-level representation discrepancy, and identifiable laboratory/source effects be distinguished statistically?

**RQ4 — Counterfactual data-selection policies:** How do retaining, excluding, or reweighting observations according to RF and CF change predictive performance, calibration, and engineering reliability?

**RQ5 — Causal concept explanations:** Can a regression-adapted Causal Concept Wrapper Network (CCW-Net), inspired by arXiv:2611.14407, provide more faithful explanations of learned engineering concepts and their contributions to overtopping predictions?

**RQ6 — Engineering decision-making:** Can RF/CF-informed data-quality assessment provide a more defensible alternative to conventional automatic exclusion of category-4 observations?

### 4. Predictive models and hyperparameter optimization

Implement a controlled set of regression models as diagnostic tools, not as the central novelty claim.

Include at least:
- Random Forest.
- XGBoost.
- CatBoost.
- A multilayer perceptron.
- The proposed regression-adapted CCW-Net.

Use Optuna for hyperparameter optimization.

For every model:
- Define a justified hyperparameter search space.
- Use nested geometry-grouped validation.
- Tune only within training data.
- Apply identical outer folds for fair model comparison.
- Use early stopping where appropriate.
- Record the best hyperparameters, objective values, and optimization history.
- Use reproducible seeds.
- Save out-of-fold predictions for every observation.

Do not select hyperparameters based on the final held-out test results.

Use appropriate regression metrics, including RMSE, MAE, R², bias, and F10.

Prediction comparisons must be based primarily on out-of-fold predictions for held-out geometries.

#### Required prediction scatter plots

Generate actual-versus-predicted scatter plots for every model.

Each model must have:
- Measured values on the horizontal axis.
- Predicted values on the vertical axis.
- A 1:1 reference line.
- Clearly reported RMSE, MAE, R², and sample size.
- Consistent axis limits across comparable models.
- Appropriate point transparency or density visualization to prevent overplotting.

Produce:
1. An overall multi-panel comparison of all models.
2. Separate plots stratified by RF.
3. Separate plots stratified by CF.
4. A selected comparison for category-4 observations.
5. Residual-versus-predicted plots to identify systematic error patterns.

Do not use training-set scatter plots as evidence of generalization performance.

### 5. RF and CF independent validation

This is the principal empirical contribution.

#### RF: test-level reliability

Evaluate whether increasing RF categories are associated with:
- Larger absolute out-of-fold residuals.
- Greater within-geometry residual dispersion.
- Higher predictive uncertainty.
- Lower repeatability where repeated measurements exist.
- Higher measurement-associated uncertainty.
- Reduced predictive calibration.

Use ordinal statistical models, mixed-effects models, heteroscedastic regression, and geometry-clustered bootstrap confidence intervals as appropriate.

#### CF: geometry-level complexity

Evaluate whether increasing CF categories are associated with:
- Larger geometry-level generalization errors.
- Greater unexplained geometry effects.
- Larger errors on held-out geometries.
- More difficult structural representation using the CLASH descriptors.
- Greater uncertainty in poorly represented structural configurations.

Treat geometry as the statistical grouping unit.

#### Independent evidence

Search the repository for original experimental reports, laboratory information, repeated tests, and measurement documentation.

Where possible, conduct a blinded independent expert reassessment of a representative subset of RF and CF ratings.

Use suitable ordinal inter-rater agreement statistics.

Clearly distinguish predictive validity from independent criterion validity.

If independent laboratory evidence or expert reassessments are unavailable, do not claim that the correctness of the original RF/CF assignments has been established.

### 6. Error decomposition

Use out-of-fold residuals to investigate separate sources of unexplained error.

Develop a hierarchical model that distinguishes, where identifiers permit:

- Test-level residual variation.
- Geometry-level representation discrepancy.
- Laboratory or experimental-source effects.
- Remaining model discrepancy.

Investigate whether RF mainly explains test-level uncertainty and whether CF mainly explains geometry-level difficulty.

Do not label all remaining residual variation as measurement error.

Use the term measurement-associated uncertainty unless independent repeatability information supports a stronger measurement-error interpretation.

### 7. Causal concept learning

Adapt the CCW-Net methodology from:

https://arxiv.org/abs/2611.14407

Use physically meaningful, documented engineering concepts derived from CLASH variables.

Develop vector-valued concept representations and a continuous regression output for log-transformed dimensionless overtopping discharge.

Evaluate:
- Concept necessity.
- Concept sufficiency.
- Concept independence.
- Interactions between physically meaningful concepts.
- Concept-level interventional mediation within the model.

Keep RF and CF as separate data-quality metadata, not ordinary hydraulic forcing variables.

Do not equate causal effects within the neural network with real-world physical causal effects.

Do not repeat the direct hydraulic causal-effect analyses of the previous Ocean Engineering paper.

The distinctive application of CCW-Net must be to investigate the reliability of concept-based explanations and the relationship between learned engineering representations and CLASH data-quality categories.

Compare the full architecture against an unaligned concept model and appropriate loss-function ablations.

### 8. Counterfactual data-selection auditing

This must be a central engineering component.

Compare alternative policies using the same validation framework:

1. Retain all RF/CF categories.
2. Exclude RF = 4.
3. Exclude CF = 4.
4. Exclude observations with RF = 4 or CF = 4.
5. Apply conventional RF/CF weighting.
6. Develop an empirically justified quality-aware weighting or uncertainty strategy.

For each policy, evaluate:
- Predictive accuracy on common held-out observations.
- Predictive accuracy on RF = 4 and CF = 4 observations.
- Geometry-level generalization.
- Calibration and prediction uncertainty.
- Bias across structure families.
- Stability of prediction and engineering conclusions.

Use counterfactual comparisons to establish how the same engineering prediction task changes under alternative data-retention policies.

Distinguish the effect of changing the training-data policy from the effect of changing a physical input.

Do not use freeboard intervention or physical monotonicity analysis as a principal contribution, because these have already been investigated in the first paper.

### 9. Required figures and tables

Generate publication-quality figures, including:

- RF and CF frequency distributions.
- RF × CF 4 × 4 heatmap.
- Actual-versus-predicted scatter plots for all models.
- Residual distributions by RF and CF.
- Geometry-level error decomposition.
- CCW-Net architecture and concept attribution.
- RF/CF ordinal-validation plots.
- Counterfactual data-selection policy comparisons.
- Uncertainty calibration by quality category.
- Full-data versus category-4-filtered model comparisons.

Generate tables covering data description, model performance, Optuna results, RF/CF validity, variance components, causal concept analysis, and data-selection sensitivity.

### 10. Formatting and figure standards

Use **Times New Roman throughout all figures, tables, and manuscript-outline outputs**.

This includes:
- Plot titles.
- Axis labels.
- Tick labels.
- Legends.
- Annotations.
- Table text.
- Captions.

Use consistent font sizes, scientific notation, engineering units, and mathematical symbols.

Figures must be high-resolution and suitable for an Elsevier journal submission. Prefer vector PDF/SVG and high-resolution TIFF/PNG exports where appropriate.

### 11. Final research outputs

Produce:
- Complete reproducible Python implementation.
- Original-data audit.
- Optuna optimization results.
- Out-of-fold prediction files.
- Actual-versus-predicted scatter plots for all models.
- Statistical validation and sensitivity results.
- All final figures and tables.
- A structured manuscript outline for *Ocean Engineering*.
- A separate supplementary-material outline containing detailed methodology, additional results, robustness checks, and supporting figures/tables.
- A critical originality and overlap assessment against both previous publications.

The manuscript outline must clearly identify what constitutes the new scientific contribution and must not reuse the abstracts, conclusions, or novelty claims of the previous papers.

### 12. Final acceptance criteria

The project should only be considered successful if it establishes, with appropriate statistical evidence:

1. Whether RF and CF have measurable predictive validity.
2. Whether test-level reliability can be distinguished from geometry-level representation difficulty.
3. Whether conventional category-4 exclusion is justified or unnecessarily discards useful information.
4. Whether quality-aware data selection improves engineering prediction reliability.
5. Whether causal concept learning adds explanatory value beyond conventional feature attribution.
6. Whether the resulting contribution is clearly distinct from the two previous publications.

If the results do not support one of these claims, report the negative or inconclusive finding honestly.

Prioritize a defensible engineering contribution and independent scientific novelty over maximizing model accuracy.