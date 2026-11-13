# Master Research and Implementation Prompt
## Causal Concept Learning, Counterfactual Auditing, and Independent Validation of CLASH Reliability and Complexity Ratings for Wave Overtopping

### 1. Research objective

I will provide access to the original, unfiltered CLASH wave-overtopping database through this repository. Unlike some commonly used processed versions, the original dataset is expected to retain observations across all four Reliability Factor (RF = 1–4) and Complexity Factor (CF = 1–4) categories, including records with RF = 4 and/or CF = 4.

Your task is to conduct a comprehensive, reproducible research study and implement the complete computational framework described below.

The proposed research combines four methodological components:

1. **Causal Concept Learning:** Adapt the Causal Concept Wrapper Network (CCW-Net) introduced in Rountree et al. (2026), arXiv:2611.14407, to continuous wave-overtopping regression.
2. **Counterfactual Physical Auditing:** Identify predictions that contradict physically defensible directional relationships and investigate the conditions under which these inconsistencies occur.
3. **Independent Validation of CLASH RF and CF:** Evaluate whether the original reliability and complexity ratings agree with independently measurable indicators of data quality, model error, and structural representation difficulty.
4. **Error Decomposition:** Distinguish measurement-associated uncertainty, geometry-related representation error, laboratory/source effects, and model-induced inconsistencies as far as the available data permit.

The final objective is not merely to improve prediction accuracy. It is to establish whether the existing CLASH quality indicators can reliably distinguish measurement-related problems from structural representation limitations and model deficiencies.

The research must be scientifically defensible, use geometry-aware validation, retain all original observations in the primary analysis, and avoid unjustified causal claims.

### 2. Reference methodology and methodological adaptation

Use the following paper as the principal methodological inspiration:

Rountree et al. (2026), *Causal Concept Explanations for Deep Neural Models*, arXiv:2611.14407.

Reference:
https://arxiv.org/abs/2611.14407

Relevant elements to study and adapt include:

- Concept encoder and vector-valued concept representations.
- Concept-dependent prediction head.
- Conditional counterfactual mediator modeling.
- Necessity, sufficiency, and independence alignment criteria.
- Interventional mediation analysis.
- Concept-level causal attribution.
- Ablation and robustness analysis.

Do not simply reproduce the original architecture without considering the characteristics of CLASH.

The original CCW-Net was developed for concept-labeled neural prediction problems. The present application is continuous regression with mixed numerical, ordinal, and categorical engineering variables. Therefore, any modifications required for continuous wave-overtopping prediction must be explicitly justified, implemented, and documented.

In particular:

- Adapt the output head to predict continuous log-transformed dimensionless overtopping discharge.
- Use vector-valued concept representations.
- Handle multi-class and ordinal physical concepts without arbitrarily forcing every concept into a binary present/absent representation.
- Preserve physically meaningful interactions between concepts.
- Do not interpret CCW-Net's internal concept effects as automatically identified physical causal effects.
- Report model-level causal alignment separately from physical input-intervention effects.

Do not claim that the proposed architecture is the first causal concept model in engineering unless this is supported by a dedicated literature review.

### 3. Phase I — Repository inspection and original data audit

Before writing any model, inspect the entire repository.

Identify:

- Original CLASH database files.
- Processed or previously filtered CLASH datasets.
- Data dictionaries and variable definitions.
- Existing scripts and previous model implementations.
- Existing geometry-grouping procedures.
- Test identifiers, structure identifiers, laboratory identifiers, test-series identifiers, and literature/reference metadata.
- Any available information about measurement procedures, estimated parameters, experimental conditions, and data-quality flags.
- Existing preprocessing and exclusion criteria used in previous studies.

Do not assume that a file is the original complete CLASH dataset merely because it contains a large number of rows.

Create an auditable data inventory containing:

- File name and format.
- Number of rows and columns.
- Available variables.
- Missing-value percentages.
- Duplicate records.
- Numerical ranges and units.
- RF and CF category frequencies.
- Number of unique structures and candidate geometry groups.
- Available laboratory, source, and test-series identifiers.
- Potential data inconsistencies.

#### 3.1. RF and CF verification

Locate the original RF and CF columns using the data dictionary and source documentation.

Verify that:

- RF contains the expected categories 1, 2, 3, and 4.
- CF contains the expected categories 1, 2, 3, and 4.
- Values outside the documented range are flagged.
- Missing values are reported separately.
- CF consistency within repeated tests of the same structure is examined.
- RF variation across different tests of the same geometry is examined.

Do not silently convert missing RF or CF values into category 1.

Do not assume that CF and RF have equal numerical intervals. Treat them as ordinal variables in statistical modeling.

Establish the exact meaning of each category using the CLASH documentation. In particular, verify the interpretation of RF = 4 and CF = 4 rather than relying only on assumptions.

#### 3.2. Preserve the original observations

This is a strict requirement.

Do not remove records merely because:

- RF = 4.
- CF = 4.
- Both RF and CF equal 4.
- An observation has a large prediction residual.
- A measurement appears unusual.

The primary dataset must retain all original records that can be read and interpreted.

Create separate analysis flags for:

- RF = 4.
- CF = 4.
- RF = 4 and CF = 4.
- Missing quality ratings.
- Potential outliers.
- Potentially censored or zero-overtopping observations.
- Inconsistent or physically questionable records.

If a record cannot be used for a particular model because of missing inputs or an undefined response transformation, document the exact reason and retain it in the master dataset.

Never overwrite the raw database.

### 4. Phase II — Physical variables and response preparation

Identify the exact CLASH response variable and the variables required to calculate dimensionless overtopping discharge.

Use the established CLASH/overtopping definition and verify all units before deriving the target.

Where supported by the data, use:

\[
q^*=\frac{q}{\sqrt{gH_{m0}^{3}}}
\]

and:

\[
Y=\log_{10}(q^*)
\]

Verify that the measured discharge \(q\) and wave height \(H_{m0}\) are expressed in compatible units.

Do not use arbitrary epsilon addition to transform zero observations.

Distinguish between:

1. Positive measured overtopping.
2. Zero measured overtopping.
3. Below-detection-limit observations.
4. Missing or invalid response values.

If these categories exist, investigate a suitable two-part or censored-response modeling strategy.

The primary positive-discharge regression should be clearly distinguished from the wet/dry or detection model.

All preprocessing, imputation, standardization, feature selection, and target transformations must be fitted using training data only within each validation fold.

Create a complete variable dictionary, including:

- Original variable name.
- Engineering meaning.
- Units.
- Physical role.
- Data type.
- Missing-value frequency.
- Transformation applied.
- Whether it is eligible as a physical predictor.
- Whether it is used only for data-quality evaluation.

Do not invent variables or infer missing engineering parameters without documentation.

### 5. Phase III — Geometry-aware validation and baseline models

Prevent leakage between related observations.

Use geometry-aware nested cross-validation as the primary validation strategy.

First investigate whether original test-series, structure, laboratory, or experimental identifiers are available.

If reliable identifiers exist, use them to construct additional grouping strategies.

If they are unavailable, reconstruct geometry groups from documented geometry descriptors and explicitly identify the limitations of this approximation.

At minimum, implement:

1. Random record-level cross-validation as an optimistic reference.
2. Geometry-grouped cross-validation as the primary evaluation.
3. Grouping sensitivity analysis using progressively coarser structure definitions where defensible.
4. Leave-one-structure-family-out validation.
5. Laboratory- or source-level validation if sufficient identifiers are available.

All model comparisons must use the same fold assignments.

Hyperparameter tuning must occur inside the training portion of each outer fold.

Do not use test-fold observations to determine:

- Feature scaling.
- Imputation values.
- Hyperparameters.
- Concept definitions.
- Concept thresholds.
- Counterfactual distributions.
- Uncertainty calibration.

#### 5.1. Baseline models

Implement and compare:

- Random Forest regression.
- XGBoost regression.
- Monotonic XGBoost where appropriate.
- Multilayer Perceptron regression.
- A concept-bottleneck regression baseline.
- An unaligned concept-wrapper model.
- The proposed CCW-Net-based model.

The primary deep-learning backbone should be suitable for tabular engineering data.

Use a pretrained tabular MLP or another justified tabular neural architecture as the CCW-Net backbone. Do not apply an image-based architecture to CLASH.

For fair comparison, all models must use identical outer validation folds and the same available training observations.

Report performance both overall and stratified by RF and CF.

#### 5.2. Treatment of RF and CF in predictive models

The primary physical mean-prediction model must not automatically use RF and CF as ordinary hydraulic or structural input variables.

They are database quality/complexity metadata, not direct hydraulic forcing variables.

Instead, investigate their roles in:

- Conditional prediction uncertainty.
- Residual variance.
- Geometry-level generalization error.
- Calibration quality.
- Data-quality-aware training.
- Sensitivity analyses.

You may implement an explicitly labelled secondary model that uses RF and CF to estimate predictive uncertainty or a heteroscedastic variance component.

A separate ablation may investigate their use in the prediction model, but this must not replace the primary physical model or lead to claims that RF or CF directly changes the physical overtopping process.

### 6. Phase IV — Engineering concept ontology

Develop a physically meaningful concept ontology before training the proposed model.

Candidate concept groups should be selected after inspecting the actual CLASH variables.

Consider the following groups where the available variables support them:

**A. Hydraulic concepts**

- Relative freeboard and overtopping protection.
- Wave forcing and wave steepness.
- Breaking and non-breaking regimes.
- Wave incidence and obliquity.
- Water depth and submergence conditions.

**B. Structural concepts**

- Structural family: slope, bermed slope, vertical wall, composite structure, or other documented families.
- Crest and berm geometry.
- Structural roughness.
- Permeability and armour-related descriptors, where available.
- Complexity of the cross-section representation.

**C. Data-quality concepts**

- Reliability category.
- Measurement-information completeness.
- Presence of estimated or indirectly determined parameters.
- Availability of laboratory or source metadata.

Important requirements:

1. Define each concept using engineering knowledge and documented CLASH variables.
2. Document every threshold used to assign concept categories.
3. Avoid thresholds selected solely to maximize predictive performance.
4. Do not use the measured overtopping response to construct predictor-side concept labels.
5. Avoid duplicating the same underlying information across multiple concept slots.
6. Document mutual exclusivity or overlap between concept categories.
7. Preserve important physical interactions.

For continuous physical properties, consider justified ordinal or multi-class representations rather than arbitrary binary labels.

The physical concept ontology must be separate from RF and CF.

RF and CF should be treated as a separate quality/complexity assessment layer, not as ordinary physical concepts.

Create an ontology table with:

- Concept identifier.
- Concept name.
- Engineering definition.
- Input variables.
- Label-generation rule.
- Number of observations per category.
- Physical interpretation.
- Expected interactions.
- Limitations and assumptions.

If a reliable concept cannot be constructed from the available variables, do not fabricate it.

### 7. Phase V — Proposed Causal Concept Wrapper Network

Implement a regression-oriented CCW-Net adaptation.

The model should contain:

1. A tabular feature encoder/backbone.
2. A vector-valued concept encoder.
3. One concept representation slot per defined physical concept group.
4. An attention-based or otherwise justified concept aggregation head.
5. A scalar regression output for \(\log_{10}(q^*)\).
6. An auxiliary concept-label prediction component.
7. A conditional counterfactual mediator model.

The backbone must be trained only using the training partition of the respective validation fold.

The concept encoder should generate vector representations rather than relying exclusively on scalar feature-importance scores.

#### 7.1. Counterfactual mediator modeling

Investigate a conditional normalizing flow for modeling the distribution of concept representations:

\[
p(M_j \mid C_j,X)
\]

where:

- \(M_j\) is the latent representation of concept \(j\).
- \(C_j\) is the corresponding concept label.
- \(X\) represents the available conditioning information.

A RealNVP or another justified conditional invertible-flow architecture may be used.

A conditional Gaussian mixture or another statistically defensible alternative may be implemented as a benchmark.

The counterfactual representation should be sampled from a conditional distribution corresponding to an alternative concept state.

Only use alternative concept states that are adequately represented in the observed data.

For multi-category or ordinal concepts, adapt the original binary concept-flipping procedure carefully.

Do not force an artificial present/absent interpretation onto every continuous engineering variable.

#### 7.2. Causal alignment criteria

Adapt and implement the three CCW-Net criteria:

**Necessity**

An inactive or irrelevant concept should not produce an artificial change in the prediction.

For multi-category concepts, formulate an appropriate alternative-state comparison that respects the concept's engineering meaning.

**Sufficiency**

The active concept representations should collectively retain enough information to reproduce the target regression output.

**Independence**

Concept contributions should be distinguishable and should not be attributed incorrectly to unrelated concepts.

However, do not force strict additive independence when the underlying hydraulic process involves meaningful physical interactions.

Where justified, include explicit concept-interaction terms or a residual interaction component.

The adapted loss function should include:

\[
\mathcal{L} =
\mathcal{L}_{task}
+\lambda_{cls}\mathcal{L}_{concept}
+\lambda_{nec}\mathcal{L}_{nec}
+\lambda_{suf}\mathcal{L}_{suf}
+\lambda_{ind}\mathcal{L}_{ind}
+\lambda_{int}\mathcal{L}_{interaction}
\]

The exact formulation of each term must be derived and documented.

Do not use arbitrary loss weights without sensitivity analysis.

Implement the training procedure in stages:

1. Supervised backbone and concept warm-up.
2. Conditional mediator-model fitting.
3. Causal alignment training.
4. Validation and early stopping.
5. Independent evaluation on held-out geometries.

Use multiple random seeds.

Report training stability and convergence diagnostics.

#### 7.3. Required CCW-Net ablations

At minimum compare:

- Full CCW-Net.
- Same architecture without causal alignment.
- Without necessity.
- Without sufficiency.
- Without independence.
- Without interaction modeling.
- Scalar concepts versus vector-valued concepts.
- Without conditional counterfactual generation.
- Physical concepts only.
- Physical concepts plus the separate quality-assessment layer.

The goal is to demonstrate what each component contributes, not simply to report that the full model performs best.

### 8. Phase VI — Counterfactual physical auditing

This is a central component of the study.

Identify model predictions that contradict physically defensible directional relationships.

Distinguish three different types of audit:

**A. Physical input interventions**

Modify the original engineering inputs while keeping the remaining conditions physically consistent.

Examples include:

- Increasing crest freeboard while holding the other relevant physical conditions fixed.
- Increasing armour crest elevation where the relevant geometry permits a meaningful intervention.
- Examining other directional relationships only when their expected sign is supported by engineering theory and applicable conditions.

For a freeboard intervention:

\[
\Delta_{R_c,i}
=
\hat{Y}(R_{c,i}+\delta)-\hat{Y}(R_{c,i})
\]

A positive value indicates a prediction that contradicts the expected non-increasing freeboard relationship.

Evaluate several intervention magnitudes, including local increments.

**B. Model representation interventions**

Modify a learned concept representation while holding other learned representations fixed.

These interventions are used to examine the internal causal alignment of the CCW-Net.

They must not be presented as equivalent to physically changing a structure.

**C. Empirical matched comparisons**

Identify comparable measured experiments with:

- The same or closely matching geometry.
- Similar incident wave conditions.
- Similar wave periods and heights.
- Comparable wave incidence.
- Different crest freeboards or other specifically investigated parameters.

Use matching methods with explicit tolerances and common-support diagnostics.

Do not interpret a difference between two experiments as a pure freeboard effect if water depth or other important physical conditions also differ.

#### 8.1. Directional inconsistency metrics

Implement:

- Global reversal rate.
- Local reversal rate.
- Percentage of affected records.
- Number of upward steps per response curve.
- Median magnitude of the reversal.
- Maximum local reversal.
- Reversal rate by structural family.
- Reversal rate by RF category.
- Reversal rate by CF category.
- Reversal rate by the joint RF × CF categories.

For freeboard:

\[
I_i(\delta)=
\mathbb{1}
\left[
\hat{Y}(R_{c,i}+\delta)>
\hat{Y}(R_{c,i})
\right]
\]

Use a documented numerical tolerance and a minimum meaningful increase to distinguish numerical noise from meaningful physical inconsistency.

Report both the number of affected curves and the number of affected intervention steps.

Do not confuse these statistics.

#### 8.2. Physical feasibility

Counterfactual changes must remain within the supported physical domain.

Before evaluating an intervention:

- Check the range of the relevant original variable.
- Check consistency with dependent geometric variables.
- Check physically meaningful combinations.
- Check common support in the training data.
- Flag extrapolative counterfactuals.
- Avoid impossible or unsupported combinations.

Provide a counterfactual support diagnostic for every major intervention.

### 9. Phase VII — Independent validation of RF and CF

This is one of the most important research contributions.

Evaluate RF and CF separately because they represent different properties and operate at different levels.

#### 9.1. Reliability Factor (RF)

RF is primarily a test-level quality/reliability rating.

Investigate whether increasing RF categories are associated with:

- Larger absolute out-of-fold prediction residuals.
- Greater conditional residual dispersion.
- Larger test-level uncertainty.
- Lower predictive calibration.
- Reduced consistency among comparable experimental observations.
- More frequent missing or estimated input parameters, where documented.

Treat RF as an ordinal variable.

Use suitable statistical approaches, including:

- Ordinal logistic regression.
- Mixed-effects ordinal regression.
- Generalized additive or monotonic trend models where appropriate.
- Heteroscedastic regression.
- Cluster bootstrap.
- Calibration analysis.

Do not assume that the difference between RF = 1 and RF = 2 is equivalent to the difference between RF = 3 and RF = 4.

#### 9.2. Complexity Factor (CF)

CF is primarily a structure-level complexity rating.

Investigate whether increasing CF categories are associated with:

- Greater geometry-grouped prediction error.
- Larger geometry-level residual effects.
- Higher variance in predictions across related tests.
- Greater difficulty of generalization to unseen structures.
- Higher uncertainty in regions with sparse structural representation.
- Increased physical inconsistency in unconstrained model predictions.

Use geometry-level analysis.

If multiple tests belong to the same physical geometry, do not treat their CF values as independent observations.

Where multiple CF values are assigned to records that appear to represent the same structure, investigate the discrepancy instead of silently resolving it.

#### 9.3. Joint RF × CF analysis

Investigate the interaction between reliability and complexity:

\[
E_{\mathrm{error}} =
f(RF,CF,RF\times CF,\mathbf{X})
\]

Test whether structures with high CF and tests with high RF jointly produce more difficult prediction conditions.

Be careful with terminology:

- High RF should be interpreted according to the documented direction of the RF scale.
- High CF should be interpreted according to the documented direction of the CF scale.

In particular, do not accidentally reverse the interpretation of the reliability scale.

Report the number of observations in every RF × CF combination.

Avoid conclusions from extremely small categories without adequate uncertainty intervals.

#### 9.4. Independent validity versus association

The analysis must distinguish between:

1. Predictive validity.
2. Construct validity.
3. Criterion validity.
4. Inter-rater reliability.

Residuals and predictive errors can provide evidence about predictive validity.

However, prediction errors alone cannot prove that the original RF or CF assignments were objectively correct.

If the repository contains original laboratory reports, experimental descriptions, measurement protocols, or other independent quality indicators, use them to construct an external validation set.

Where possible, implement a blinded expert re-rating of a representative subset of experiments and structures.

Compare independent ratings with original RF/CF using:

- Weighted Cohen's kappa for suitable paired ratings.
- Krippendorff's ordinal alpha where multiple reviewers are involved.
- Confusion matrices.
- Category-specific agreement.
- Confidence intervals.

If independent reports or expert ratings are unavailable, clearly state that the study evaluates the predictive validity and consistency of RF/CF, not their ground-truth correctness.

Never fabricate independent validation data.

### 10. Phase VIII — Decomposition of measurement-associated and structural representation errors

Develop a hierarchical error-decomposition framework.

The primary objective is to distinguish:

1. Measurement-associated or test-level uncertainty.
2. Geometry-related representation error.
3. Laboratory/source or experimental-protocol effects.
4. Model-induced prediction error.

Use out-of-fold residuals:

\[
r_i = y_i-\hat{y}_{i}^{OOF}
\]

Investigate a hierarchical structure such as:

\[
r_{ig\ell} =
u_g + v_\ell + \epsilon_i
\]

where:

- \(u_g\) represents unexplained geometry-level effects.
- \(v_\ell\) represents identifiable laboratory/source effects.
- \(\epsilon_i\) represents remaining test-level residual variation.

The exact model must be adapted to the identifiers actually available.

Investigate the relationship of these components with CF and RF.

Expected interpretation:

- RF should primarily be evaluated against test-level reliability and within-geometry residual variation.
- CF should primarily be evaluated against geometry-level representation difficulty and unseen-geometry generalization error.

However, do not label \(\epsilon_i\) as pure measurement error without independent repeated measurements or a suitable experimental reliability design.

Repeated tests under identical or nearly identical conditions may help estimate repeatability.

If no suitable repeated measurements exist, explicitly state that measurement error cannot be fully separated from natural experimental variability, input uncertainty, and residual model discrepancy.

Use the terms:

- Measurement-associated uncertainty.
- Within-geometry residual variation.
- Geometry-level representation discrepancy.
- Source/laboratory effects.

Only use the stronger term "measurement error" when the data support it.

Report variance components, uncertainty intervals, and sensitivity to the grouping definition.

### 11. Phase IX — Quality-aware uncertainty modeling

Investigate whether RF and CF can improve uncertainty estimation without contaminating the primary physical mean predictor.

Possible approaches include:

- Heteroscedastic regression.
- Conditional variance modeling.
- RF/CF-aware uncertainty heads.
- Quantile regression.
- Conformal prediction with geometry-grouped calibration.
- Novelty-aware uncertainty estimation.

Evaluate:

1. Overall interval coverage.
2. Coverage by RF.
3. Coverage by CF.
4. Coverage by RF × CF.
5. Interval width.
6. Calibration error.
7. Worst-performing structure family.
8. Coverage on held-out geometry groups.

Use the correct exchangeable unit for calibration.

Where geometry is the unit of generalization, calibrate and evaluate accordingly.

Do not claim that conformal guarantees hold for every individual RF/CF category unless the calibration design actually establishes that result.

### 12. Phase X — Statistical validation and robustness

Use geometry-clustered bootstrap for confidence intervals wherever the geometry is the independent sampling unit.

Use multiple seeds for deep-learning models.

Apply multiple-comparison corrections when testing many physical concepts or directional relationships.

At minimum, report:

**Prediction metrics**
- RMSE.
- MAE.
- \(R^2\).
- Bias.
- F10.
- Factor-of-2 and factor-of-3 accuracy where appropriate.

**Concept metrics**
- Concept classification accuracy.
- Macro-F1.
- Ordinal error.
- Calibration of concept categories.
- Necessity score.
- Sufficiency score.
- Independence/alignment score.
- Concept-level effect contributions.
- Interaction contribution.

**Physical audit metrics**
- Global reversal rate.
- Local reversal rate.
- Number of affected response curves.
- Reversal magnitude.
- Percentage of predictions outside the physically supported domain.

**Quality metrics**
- RF-stratified residual statistics.
- CF-stratified geometry-level errors.
- RF × CF interaction.
- Conditional residual variance.
- Uncertainty calibration by quality category.

**Uncertainty metrics**
- Nominal versus empirical coverage.
- Average interval width.
- Group-wise simultaneous coverage.
- Worst-group coverage.

Use appropriate statistical assumptions and report uncertainty rather than relying solely on p-values.

### 13. Phase XI — Essential experiments and ablations

The final research must include, at minimum, the following experiments.

**Experiment 1 — Original CLASH quality-rating audit**

Analyze all RF and CF categories without excluding category 4.

**Experiment 2 — Geometry-grouped predictive performance**

Compare all baseline learners and the proposed CCW-Net adaptation.

**Experiment 3 — Causal concept alignment**

Compare the full CCW-Net with its unaligned and loss-ablation variants.

**Experiment 4 — Counterfactual physical auditing**

Identify where and under what conditions models produce physically inconsistent predictions.

**Experiment 5 — Reliability-factor validity**

Investigate whether RF is associated with test-level reliability and measurement-associated uncertainty.

**Experiment 6 — Complexity-factor validity**

Investigate whether CF is associated with geometry-level representation difficulty and out-of-geometry generalization.

**Experiment 7 — RF × CF interaction**

Determine whether reliability and geometry complexity jointly influence predictive performance and uncertainty.

**Experiment 8 — Error decomposition**

Quantify geometry-level, source-level, and test-level residual components to the extent supported by the data.

**Experiment 9 — Quality-aware uncertainty**

Determine whether RF/CF improve uncertainty estimation and calibration.

**Experiment 10 — Sensitivity to category-4 filtering**

Compare:

- Full dataset: RF and CF from 1 to 4.
- Dataset excluding RF = 4.
- Dataset excluding CF = 4.
- Dataset excluding records with RF = 4 or CF = 4.

The full dataset must remain the primary analysis.

The filtered datasets are sensitivity analyses only.

This experiment is essential because it directly evaluates the implications of conventional CLASH filtering.

### 14. Required figures

Generate publication-quality scientific figures.

Use consistent figure numbering, variable notation, units, and color scales.

Recommended figures:

**Figure 1. Research framework**

Overall workflow linking original CLASH data, concept learning, physical counterfactual auditing, RF/CF validation, error decomposition, and uncertainty modeling.

**Figure 2. CLASH RF and CF distribution**

Separate distributions and joint 4 × 4 frequency matrix, with the number and percentage of observations in each category.

**Figure 3. RF/CF versus prediction error**

Residual distributions, absolute error, and uncertainty across RF and CF categories.

**Figure 4. Geometry-level error structure**

Variance-component decomposition and geometry-level residual heterogeneity.

**Figure 5. CCW-Net architecture**

Tabular backbone, vector concept representation, counterfactual mediator, alignment losses, regression output, and separate quality-audit branch.

**Figure 6. Causal concept attribution**

Concept-level effect decomposition and interaction contribution.

**Figure 7. Counterfactual physical auditing**

Prediction-response curves for selected representative cases, including monotonicity violations and physically consistent responses.

**Figure 8. Directional inconsistencies by RF and CF**

Global and local reversal rates stratified by RF, CF, and their interaction.

**Figure 9. RF/CF validity**

Ordinal trend, calibration, effect estimates, and uncertainty intervals.

**Figure 10. Error decomposition**

Measurement-associated/test-level residual variation versus geometry-level representation discrepancy and laboratory/source contributions.

**Figure 11. Quality-aware uncertainty**

Prediction intervals and calibration across RF and CF categories.

**Figure 12. Robustness and filtering sensitivity**

Comparison of the full dataset and the conventional category-4 filtering scenarios.

Use high-resolution output suitable for Ocean Engineering.

Prefer clean scientific layouts, readable typography, correct mathematical notation, and vector PDF/SVG formats where possible.

Do not use decorative plots without a clear analytical purpose.

### 15. Required tables

Generate the following tables.

**Table 1.** CLASH data inventory and variable dictionary.

**Table 2.** RF and CF distribution and data completeness.

**Table 3.** Geometry-grouped predictive performance of all baseline models.

**Table 4.** CCW-Net concept ontology and label definitions.

**Table 5.** CCW-Net causal alignment and ablation results.

**Table 6.** Concept-level causal attribution and interaction effects.

**Table 7.** Counterfactual physical inconsistency metrics.

**Table 8.** RF and CF ordinal validation models.

**Table 9.** Hierarchical error-decomposition results.

**Table 10.** RF × CF joint effect and uncertainty analysis.

**Table 11.** Quality-aware uncertainty and conformal coverage.

**Table 12.** Sensitivity to category-4 filtering.

Every table must report sample counts and uncertainty information where applicable.

### 16. Software engineering and repository requirements

Use Python as the primary implementation language.

Follow modular and reproducible software design.

Recommended project structure:

```text
project/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── configs/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── experiment_config.yaml
│
├── src/
│   ├── data_audit.py
│   ├── data_preprocessing.py
│   ├── geometry_grouping.py
│   ├── baseline_models.py
│   ├── concept_ontology.py
│   ├── ccw_net.py
│   ├── mediator_model.py
│   ├── counterfactual_audit.py
│   ├── rf_cf_validation.py
│   ├── error_decomposition.py
│   ├── uncertainty_modeling.py
│   ├── statistical_analysis.py
│   ├── plotting.py
│   └── evaluation.py
│
├── experiments/
│   ├── baseline/
│   ├── ccw_net/
│   ├── counterfactual/
│   ├── quality_validation/
│   └── ablation/
│
├── results/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   ├── predictions/
│   ├── diagnostics/
│   └── reports/
│
├── tests/
├── requirements.txt
├── README.md
└── run_all.py
```

Adapt this structure to the actual repository rather than creating unnecessary duplicate files.

Requirements:

- Do not overwrite original data.
- Use configuration files for experiment parameters.
- Use fixed random seeds and record software versions.
- Save fold assignments.
- Save out-of-fold predictions for every model.
- Save model configurations and hyperparameters.
- Record excluded or unusable observations with explicit reasons.
- Implement automated checks for data leakage.
- Include unit tests for target calculations, transformations, group assignments, and counterfactual feasibility.
- Make each experiment reproducible independently.
- Provide a master execution script.
- Avoid unnecessary dependencies.
- Handle CPU/GPU availability gracefully.
- Use batch processing and efficient data loading.
- Avoid excessive memory usage.
- Provide clear logs and error messages.

Do not generate synthetic CLASH observations to fill missing categories.

Do not fabricate empirical results, effect sizes, confidence intervals, or validation metrics.

### 17. Scientific interpretation and restrictions

The following restrictions are mandatory.

1. Do not equate correlation between RF/CF and prediction error with a physical causal effect.
2. Do not claim that RF measures pure measurement-error variance unless independent repeatability evidence exists.
3. Do not claim that CF directly causes overtopping.
4. Do not interpret model-internal concept mediation as real-world physical intervention effects.
5. Do not interpret a physically inconsistent model prediction as proof that the corresponding experimental observation is wrong.
6. Do not treat all residual variation as measurement error.
7. Do not assume that all concept effects are additive.
8. Do not claim external validity for structural families that are not represented in CLASH.
9. Do not make conclusions from poorly supported RF/CF category combinations.
10. Do not hide negative results or unsuccessful ablations.

Every important conclusion must be traceable to a defined analysis, an output table, a figure, and an uncertainty estimate.

Distinguish findings from hypotheses and interpretation.

### 18. Final deliverables

The final implementation must produce:

1. A complete repository/data audit report.
2. A verified variable dictionary.
3. A complete RF/CF distribution report.
4. A reproducible geometry-aware validation framework.
5. Baseline model implementations and predictions.
6. A regression-adapted CCW-Net implementation.
7. Counterfactual mediator modeling.
8. Physical counterfactual auditing.
9. Independent RF/CF predictive-validity analyses.
10. Hierarchical error decomposition.
11. Quality-aware uncertainty analysis.
12. All required tables and figures.
13. Ablation and sensitivity analyses.
14. A scientific interpretation report.
15. A proposed manuscript outline for submission to *Ocean Engineering*.
16. A list of unresolved methodological limitations.
17. A fully reproducible execution guide.

The manuscript outline must clearly distinguish:

- What is directly demonstrated by the data.
- What is supported only by predictive association.
- What concerns model-internal causal explanations.
- What concerns physically meaningful counterfactual interventions.
- What cannot be identified without independent experimental evidence.

### 19. Execution strategy

Work in the following order:

**Stage A — Inspect**

Inspect the complete repository and identify all available original data, documentation, and existing code.

**Stage B — Audit**

Produce the original-data audit, variable dictionary, RF/CF validation checks, and proposed analysis plan.

**Stage C — Validate the data structure**

Determine which geometry, test-series, laboratory, and source identifiers are genuinely available.

**Stage D — Implement baseline analyses**

Build the leakage-safe geometry-grouped validation and save out-of-fold predictions.

**Stage E — Implement the proposed causal concept architecture**

Adapt CCW-Net for continuous tabular overtopping regression and demonstrate the effects of the causal alignment objectives.

**Stage F — Conduct counterfactual and RF/CF analyses**

Implement physical audits, ordinal validation, hierarchical error decomposition, and uncertainty analysis.

**Stage G — Perform ablations and robustness tests**

Evaluate all major modeling assumptions and alternative data-filtering scenarios.

**Stage H — Generate research outputs**

Produce the final figures, tables, diagnostics, and manuscript outline.

Do not stop after merely generating an analysis plan. Once the data and repository have been inspected, proceed with implementation and testing wherever the available files permit.

If a critical data source, identifier, experimental report, or definition is unavailable, document the limitation and continue with the analyses that remain statistically defensible.

Do not make unsupported assumptions simply to complete a pipeline.

### 20. Final scientific objective

The study should ultimately answer the following research questions:

**RQ1.** Do the original CLASH RF ratings reliably distinguish tests with different levels of predictive and measurement-associated uncertainty?

**RQ2.** Do the original CF ratings reliably identify structural configurations that are difficult to represent and generalize using the standard CLASH parameters?

**RQ3.** Can causal concept learning provide more faithful and physically interpretable wave-overtopping predictions than conventional black-box regression models?

**RQ4.** Are physically inconsistent counterfactual predictions concentrated in particular data-quality and structural-complexity categories?

**RQ5.** Can hierarchical analysis distinguish test-level uncertainty, geometry-level representation error, and laboratory/source effects?

**RQ6.** Can RF/CF-informed uncertainty modeling improve predictive calibration without introducing misleading physical dependencies into the mean overtopping predictor?

**RQ7.** Does retaining RF = 4 and CF = 4 observations, with suitable uncertainty handling, provide useful information that conventional filtering discards?

The final contribution should be a scientifically rigorous framework for **causal concept-based auditing and data-quality-aware wave-overtopping prediction**, rather than another conventional machine-learning model comparison.

Prioritize methodological validity, engineering interpretation, reproducibility, and defensible conclusions over artificially maximizing model performance.


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
