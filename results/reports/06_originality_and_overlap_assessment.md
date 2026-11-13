# 6. Originality and overlap assessment

## 6.1 Basis and limits of this assessment

* Neither previous paper is in the repository. **Paper 1** (*Machine-learning prediction of wave overtopping at
  untested structures: geometry-grouped validation, freeboard monotonicity and conformal coverage*, current
  manuscript) and **Paper 2** (*Quantum machine learning for wave overtopping estimation: Integrating with
  causal inference and uncertainty quantification*, Ocean Engineering) were assessed from the contribution
  lists supplied in the research brief. A web search did not retrieve Paper 2, so its text could not be
  checked. **The authors must compare the final manuscript against both full texts before submission.**
* The literature check was limited: the scholarly-search connector's quota was exhausted, so three web
  searches were used. It is a screening, not a systematic review. Items marked "to verify" need a proper
  database search (Scopus/Web of Science: CLASH, EurOtop, "reliability factor", "complexity factor",
  "weight factor", data quality, overtopping).

## 6.2 What the literature already does with RF and CF

| Practice | Source found | Relation to this study |
|---|---|---|
| RF and CF assigned on a 1–4 scale to every CLASH test | CLASH WP2 (Verhaeghe et al.); Steendam et al. 2004; Van der Meer et al. 2009 | Object of study |
| Weight factor WF = (4−RF)(4−CF) in neural-network training; RF = 4 or CF = 4 receive zero weight | Van Gent et al. 2007 (CLASH neural network), described in later reviews | Policy P5; here its effect is measured on common held-out tests |
| Removal of records remarked "do not use for neural network"; training set of 8 372 tests | WL Delft Hydraulics NN report / CLASH NN documentation | Policy P5b |
| RF/CF used to select "reliable" data, or to set bootstrap resampling probability, in later ANN and gradient-boosting models (EurOtop ANN, XGBoost models) | Zanuttigh et al. 2016; den Bieman et al. 2020–2021 (to verify details) | Same family of practice; these papers use the ratings, they do not test them |

**Gap identified.** In the sources retrieved, RF and CF are *used* as weights or filters; no study was
found that (i) tests whether the ratings predict out-of-sample error, residual dispersion or repeatability,
(ii) separates a test-level from a geometry-level error component and relates each to RF and CF, or
(iii) measures on common held-out tests what exclusion or weighting actually changes. This is the
contribution claimed. It is stated as "not found in a limited search", not as "does not exist".

## 6.3 Boundary against Paper 1 (geometry-grouped validation, freeboard monotonicity, conformal coverage)

| Paper 1 contribution | Treatment here | Residual overlap |
|---|---|---|
| Geometry-grouped validation; generalisation to unseen structures | Used only as the procedure that produces leakage-free out-of-fold residuals. Not claimed, not a highlight. Random-split results appear once, as a sensitivity row | **Moderate.** Both papers reconstruct geometry groups from CLASH descriptors. The grouping definition should be cited to Paper 1 if identical, or its difference stated |
| Random-split optimism | Not analysed as a result. One sentence and one table row at most | Low |
| Freeboard monotonicity constraints | Removed. No monotone model, no constraint | None |
| Counterfactual freeboard-response diagnostics | Removed from code and results (the input-intervention audit of the earlier version of this repository was deleted). "Counterfactual" here refers only to **training-data policies** | **Terminology.** Define the term in the abstract to avoid confusion with Paper 1 |
| Conformal prediction and coverage | Not a contribution. Split-conformal intervals are an evaluation instrument for policies (does a policy keep nominal coverage for category-4 tests?) | **Moderate.** Coverage by RF/CF is new only if Paper 1 does not stratify coverage by RF/CF. To check in Paper 1 |
| Crest-level design application | Absent | None |

## 6.4 Boundary against Paper 2 (quantum ML, SHAP/Sobol, LinearDML, CausalForestDML)

| Paper 2 contribution | Treatment here | Residual overlap |
|---|---|---|
| Quantum ML prediction; model comparison by accuracy | No quantum model. Five conventional models as diagnostic tools; no accuracy claim | None |
| SHAP and global sensitivity analysis | TreeSHAP appears once, as the *comparator* against which concept attribution is judged (Table 11). No Sobol analysis | Low; must stay a comparator |
| LinearDML, CausalForestDML: causal effects of hydraulic and structural predictors | No treatment-effect estimation of physical predictors. CCW-Net effects are explicitly model-internal (effects of concept representations on the model output) | **Terminology.** Both papers use "causal". The manuscript must say in the introduction that no physical causal effect is estimated |
| Prediction uncertainty | Intervals used as a policy-evaluation instrument only | Low |

## 6.5 What is new in this study (claims that the results support)

1. Predictive-validity audit of the CLASH RF and CF ratings on the complete database, treating them as
   ordinal and using clustered uncertainty.
2. Model-free repeatability evidence from near-replicate tests, by RF.
3. Hierarchical separation of series-, geometry- and within-geometry residual components and their relation
   to RF and CF.
4. Counterfactual comparison of nine retention/weighting policies on common held-out tests, including the
   conventional (4−RF)(4−CF) weighting, with accuracy, calibration, family bias and stability.
5. An empirically derived quality-aware alternative (retain all, RF/CF-aware interval width).
6. A regression adaptation of CCW-Net used to test whether concept explanations are faithful and whether
   explanation quality differs across RF/CF categories; compared with conventional feature attribution.

## 6.6 Claims that must NOT be made

* That geometry-grouped validation, monotonicity or conformal coverage is introduced here.
* That RF or CF ratings are shown to be correct or incorrect (no independent re-rating exists).
* That CCW-Net estimates physical causal effects, or that it is the first causal concept model in
  engineering.
* That any model here is more accurate than existing overtopping predictors.
* That concept learning outperforms feature attribution, unless Table 11 supports the specific statement.

## 6.7 Unresolved overlap and recommended actions

1. Check whether Paper 1 already reports errors or coverage stratified by RF/CF, or any category-4
   sensitivity. If it does, cut the corresponding material here to a citation.
2. Use a different figure set from Paper 1: no freeboard-response curves, no coverage-versus-width figure
   as a main figure.
3. Run a systematic database search for prior evaluations of RF/CF (Section 6.1) and for the exact
   weighting schemes in Zanuttigh et al. (2016) and den Bieman et al. (2021).
4. Verify the RF/CF category wording against the CLASH WP2 report.

Sources consulted in the screening:
[CLASH database description, TU Delft repository](https://repository.tudelft.nl/file/File_04262eab-a3f1-4bd3-9ecd-1ecf6a7626a1?preview=1);
[Coastal Flooding Hazard by Wave Overtopping, Phase 2 report](https://assets.publishing.service.gov.uk/media/602d17d38fa8f503896d6c7f/Coastal_Flooding_Hazard_by_Wave_Overtopping__Phase_2__Final_Report.pdf);
[Etemad-Shahidi, MODSIM 2015](https://www.mssanz.org.au/modsim2015/L16/etemad-shahidi.pdf);
[ICCE paper on the CLASH neural network](https://icce-ojs-tamu.tdl.org/icce/article/download/7826/pdf_909/31445);
[den Bieman et al., Journal of Hydroinformatics 2021 (DOAJ record)](https://doaj.org/article/afcfc04513604982a298bbaa3f70596c);
[Zanuttigh et al., prediction of extreme and tolerable overtopping (Unibo repository)](https://cris.unibo.it/bitstream/11585/566757/4/PP_Prediction%20of%20extreme%20and%20tolerable.pdf).
