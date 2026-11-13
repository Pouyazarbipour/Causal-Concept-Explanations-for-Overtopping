# 2. Methods

Code: `src/` and `experiments/`. Configuration: `configs/`. All steps are deterministic given the seeds.

## 2.1 Response, predictors, grouping

* Response Y = log10 q*, q* = q / sqrt(g Hm0,toe^3), positive discharges only (9 333 tests). Zero
  discharge (1 056 usable tests) is modelled separately as wet/dry; no epsilon is added.
* Fifteen dimensionless physical predictors (Froude scaling with toe conditions): Rc/Hm0, Ac/Hm0, Gc/L0,
  Hm0/L0, h/L0, ht/Hm0, Bt/L0, B/L0, hb/Hm0, tan(alpha_B), cot(alpha_d), cot(alpha_u), gamma_f, beta, 1/m.
  RF and CF are never predictors of the mean.
* Geometry groups are reconstructed within each test series from water-level-invariant descriptors
  (h+Rc, h+Ac, h−ht, berm level, widths, slopes, gamma_f, foreshore). Grouped validation is used solely
  to obtain leakage-free out-of-fold predictions; it is not a contribution of this study.

## 2.2 Diagnostic regression models and hyper-parameter optimisation

Random Forest, XGBoost, CatBoost, MLP and the regression CCW-Net. Five outer geometry-grouped folds,
identical for all models. Inside each outer-training partition, Optuna (TPE sampler, fixed seed) minimises
the RMSE of a grouped inner 3-fold cross-validation:

| Model | Search space | Trials per fold |
|---|---|---|
| Random Forest | max_features 0.2–1; min_samples_leaf 1–10; max_depth 6–30; 300 trees | 12 |
| XGBoost | learning rate 0.01–0.2 (log); depth 3–10; min_child_weight 1–20 (log); subsample 0.5–1; colsample 0.4–1; lambda 0.1–20 (log); gamma 0–1 | 30 |
| CatBoost | learning rate 0.02–0.3 (log); depth 4–9; l2_leaf_reg 1–20 (log); random_strength 0–2; rsm 0.5–1 | 15 |
| MLP | width 64/128/256; depth 2–5; dropout 0–0.2; learning rate 3e-4–5e-3 (log); weight decay 1e-6–1e-3 (log) | 8 |
| CCW-Net wrapper | concept dimension 4/8/16; encoder width 64/128/256; learning rate 3e-4–3e-3 (log); interaction rank 2/4/8; alignment-weight multiplier 0.1–10 (log) | 6 |

Early stopping: boosted trees stop on the inner validation fold (50 rounds); the final model uses 1.1 x the
mean best iteration. Networks stop on a grouped validation split of the training partition. Imputation and
scaling are fitted on training rows only. Test folds are never used for selection. Best parameters,
objective values and all trials are stored in `results/models/optuna/` (Table 4, Figure S1). The CCW-Net
objective is the validation RMSE of the aligned wrapper on a shortened schedule; the selected
configuration is then used for all its ablations so that they differ only in the loss terms.

Out-of-fold predictions for every record and model: `results/predictions/geometry__<model>.npz`.
The reference model for residual-based analyses is XGBoost (pre-specified); conclusions are repeated with
the other models, and with random-record and series-held-out residuals, as sensitivity.

## 2.3 RF and CF validity

RF and CF are ordinal; category effects are estimated per category or by rank statistics.

* Residual statistics per category (MAE, RMSE, bias, within-geometry SD) with test-series cluster bootstrap.
* Ordinal trend: Spearman correlation of |residual| with the category; category 4 vs 1 contrasts.
* Proportional-odds models of RF and CF on |residual|, within/between-geometry parts and completeness
  indicators (cluster bootstrap; a mixed-effects ordinal model is not available in the software stack).
* Heteroscedastic model: Gamma GLM (log link) of squared residuals on RF and CF categories, with and
  without a linear-by-linear interaction.
* Repeatability without any model: near-replicate pairs (same geometry; Hm0, Tm-1,0, h, Rc within 1 %; same
  beta), repeat SD = sqrt(mean(d^2)/2), by the larger RF of the pair.
* CF at geometry level: |geometry mean residual|, geometry RMSE and the difference between grouped and
  random-CV error per geometry; CF aggregated by its maximum within the geometry, inconsistencies reported.
* Construct-related evidence: association of RF/CF with inferred completeness indicators (default period
  ratios, missing offshore data, missing reference) and with zero-overtopping share.
* No independent re-rating or laboratory documentation exists; inter-rater statistics are not computed.

## 2.4 Error decomposition

Exact REML for nested random intercepts, r = mu + u_series + u_geometry + e, with series-cluster bootstrap.
Sensitivities: coarser geometry definitions, reference-level source, records with an inferred laboratory,
other models, random-CV and series-held-out residuals, and strata of RF and CF. The residual e is called
within-geometry residual variation; it is compared with the model-free near-replicate variance.

## 2.5 Counterfactual data-selection policies

Same outer folds, learner, per-fold hyper-parameters, seeds and calibration geometries; only the training
policy changes (Section 3.5 of the results). For each policy: a model fitted on all policy-eligible training
rows gives the accuracy; a model fitted on 75 % of the training geometries, calibrated on the remaining 25 %
(policy-eligible rows only, group-weighted split conformal, 90 %), gives the intervals. The empirical
quality model sigma^2(RF, CF) is a main-effects log-linear fit to grouped inner out-of-fold squared residuals
of the training partition. Outcomes: RMSE on all, common (RF<4 and CF<4), RF = 4 and CF = 4 held-out tests
with paired geometry-cluster bootstrap differences to the retain-all policy; geometry-level bias; coverage
and width by category; bias by structural family; share of predictions changed by more than a factor 2.

## 2.6 Regression CCW-Net

Adaptation of Rountree et al. (2026, arXiv:2611.14407); details and derivation in the supplementary
outline (S3). Frozen tabular backbone; encoder producing one squashed vector per concept group (ten groups,
Table S); bias-free attention read-out A(M) plus explicit low-rank pairwise interaction I(M); linear
output on the standardised target. Loss

    L = L_task + l_cls L_concept + l_nec L_nec + l_suf L_suf + l_ind L_ind + l_int L_int

with necessity applied to the six concept groups that have a physical null state, sufficiency to the
active slots including their mutual interactions, and independence to the additive read-out only.
Counterfactual slot values come from a per-concept conditional RealNVP. Comparators: unaligned wrapper
(same architecture, task + concept loss), loss ablations (no necessity / sufficiency / independence /
interaction / conditional counterfactual), concept-bottleneck regression.

Evaluation on held-out geometries: necessity effect, sufficiency error, additivity residual, effect share
through absent concepts, matched label-swap routing share, concept read-out accuracy. New in this study:
(i) the per-record explanation-quality quantities are stratified by RF and CF; (ii) a linear probe tests
whether RF/CF categories can be recovered from the learned physical concept vectors; (iii) concept
attribution is compared with grouped TreeSHAP of the reference model (rank agreement, attribution assigned
to physically absent concepts). All effects are effects of model representations on model output.

## 2.7 Statistics

Cluster bootstrap (1 000 resamples; test series for RF/CF statistics, geometry for prediction and
coverage); Holm correction within families of tests; no conclusion from RF x CF cells with fewer than 50
records or three series.
