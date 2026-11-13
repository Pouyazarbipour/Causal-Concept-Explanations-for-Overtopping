# Are the CLASH data-quality ratings valid and useful?

Reproducible study of the original, unfiltered CLASH wave-overtopping database (10 532 tests, Reliability
Factor RF and Complexity Factor CF 1–4). It asks whether RF and CF behave as measures of test reliability and
structural complexity, and whether excluding or down-weighting tests by these ratings helps or harms
engineering prediction. Tools: Optuna-tuned diagnostic regression models with leakage-free geometry-grouped
out-of-fold predictions, ordinal and hierarchical statistics, a counterfactual comparison of nine
data-selection policies, and a regression adaptation of the Causal Concept Wrapper Network (CCW-Net,
Rountree et al. 2026, arXiv:2611.14407) used to test explanation faithfulness.

**All 10 532 records are retained** in the master table. No record is removed because of its rating, its
residual or its appearance; records that cannot enter a model carry an explicit reason.

Reports (`results/reports/`):

| Report | Content |
|---|---|
| `01_data_audit_report.md` | repository/data inventory, RF/CF verification, flags, identifiers |
| `02_methods.md` | models, Optuna search spaces, statistics, policies, CCW-Net |
| `03_scientific_interpretation.md` | results for RQ1–RQ6 and the acceptance criteria |
| `04_manuscript_outline_ocean_engineering.md` | manuscript structure |
| `05_supplementary_material_outline.md` | supplementary material structure |
| `06_originality_and_overlap_assessment.md` | boundaries against the two previous papers |
| `07_limitations.md` | unresolved methodological limitations |

## Layout

```
data/raw/            original workbook (read-only, SHA-256 checked on every read)
data/processed/      clash_master.csv (all 10 532 rows + flags), fold_assignments.csv
configs/             data_config.yaml, model_config.yaml, experiment_config.yaml
src/                 data_audit, data_preprocessing, features, geometry_grouping, concept_ontology, baseline_models,
                     ccw_net, mediator_model, data_selection, rf_cf_validation, error_decomposition,
                     uncertainty_modeling, statistical_analysis, evaluation, plotting, pipeline, utils
experiments/         run_models.py, run_ccw.py, run_policies.py, run_analysis.py
results/             tables/ (CSV + Markdown), figures/ (PDF + PNG), predictions/ (out-of-fold, all models),
                     models/ (selected hyper-parameters per fold), diagnostics/, reports/
tests/               unit tests (target, transforms, grouping, leakage, counterfactual feasibility, statistics)
docs/                CCW-Net reference paper, first research brief
```

## Execution guide

```bash
pip install -r requirements.txt
```

```bash
python run_all.py
```

| Stage | Command | Writes | Time (4 CPU cores) |
|---|---|---|---|
| audit | `python -m src.data_audit` | master table, folds, Tables 1–2 | < 1 min |
| tests | `python -m pytest -q tests` | – | < 1 min |
| models | `python -m experiments.run_models` | Optuna-tuned out-of-fold predictions (RF, XGBoost, CatBoost, MLP); wet/dry model | ~ 1 h |
| ccw | `python -m experiments.run_ccw` | CCW-Net (Optuna-tuned), ablations, unaligned wrapper, concept bottleneck | ~ 2–3 h |
| policies | `python -m experiments.run_policies` | nine data-selection policies (needs `models`) | ~ 10 min |
| analysis | `python -m experiments.run_analysis [--only perf rfcf decomp ccw policy unc]` | Tables 3–15 and supplementary tables | ~ 30 min |
| figures | `python -m src.plotting` | Figures 1–15, S1 (PDF + 600 dpi PNG) | ~ 2 min |

Every fold of `models`, `ccw` and `policies` is checkpointed in `results/models/*_cache/` (not versioned),
so an interrupted run resumes. Figures use Times New Roman; where it is not installed, the metrically
identical Liberation Serif is used automatically.

## Guard rails implemented in code

* `read_raw` refuses to run if the workbook checksum changes; nothing writes to `data/raw`.
* `check_no_leakage` is called for every outer and inner split; imputation, scaling, target
  standardisation, hyper-parameters, mediator models and conformal calibration use training rows only.
* RF and CF never enter a mean predictor. They are used only to select or weight training records
  (policies) and to scale prediction intervals.
* Zero overtopping is never transformed with an epsilon; it is handled by a separate wet/dry model.
* Concept labels use predictor variables only (never q, RF or CF); thresholds are fixed constants.
