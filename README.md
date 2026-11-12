# Causal concept auditing of CLASH wave-overtopping data

Reproducible framework for (i) a regression-oriented Causal Concept Wrapper Network (CCW-Net, after
Rountree et al. 2026, arXiv:2611.14407), (ii) counterfactual physical auditing of overtopping
predictors, (iii) predictive-validity analysis of the CLASH Reliability (RF) and Complexity (CF)
factors, (iv) hierarchical error decomposition and (v) quality-aware, geometry-calibrated uncertainty.

**All 10 532 original CLASH records are retained** (RF and CF 1–4). No record is removed because of its
rating, its residual or its appearance; records that cannot enter a model carry an explicit reason.

Start with `results/reports/`:

| Report | Content |
|---|---|
| `01_data_audit_report.md` | repository/data inventory, RF/CF verification, flags, identifiers |
| `02_methods_ccw_net_adaptation.md` | derivation of the adapted architecture and loss terms |
| `03_scientific_interpretation.md` | results for RQ1–RQ7, with table/figure references and uncertainty |
| `04_manuscript_outline_ocean_engineering.md` | proposed manuscript structure and evidence classes |
| `05_limitations.md` | unresolved methodological limitations |

## Layout

```
data/raw/            original workbook (read-only, SHA-256 checked on every read)
data/processed/      clash_master.csv (all 10 532 rows + flags), fold_assignments.csv
configs/             data_config.yaml, model_config.yaml, experiment_config.yaml
src/                 data_audit, data_preprocessing, geometry_grouping, concept_ontology, baseline_models,
                     ccw_net, mediator_model, counterfactual_audit, rf_cf_validation, error_decomposition,
                     uncertainty_modeling, statistical_analysis, evaluation, plotting, pipeline, utils
experiments/         run_baselines.py, run_ccw.py, run_analysis.py
results/             tables/ (CSV + Markdown), figures/ (PDF + PNG), predictions/ (out-of-fold, all models),
                     models/ (selected hyper-parameters per fold), diagnostics/, reports/
tests/               unit tests (target, transforms, grouping, leakage, counterfactual feasibility, statistics)
docs/                CCW-Net reference paper, original research brief
```

## Execution guide

```bash
pip install -r requirements.txt
```

```bash
python run_all.py
```

Stages can be run one at a time; each later stage only needs the files written by the earlier ones.

| Stage | Command | Writes | Time (4 CPU cores) |
|---|---|---|---|
| audit | `python -m src.data_audit` | master table, folds, Tables 1, 2, 4 | < 1 min |
| tests | `python -m pytest -q tests` | – | < 1 min |
| baselines | `python -m experiments.run_baselines` | OOF predictions of RF, XGBoost, monotone XGBoost, MLP on six validation schemes; filtering runs; wet/dry model | ~ 40 min |
| ccw | `python -m experiments.run_ccw` | CCW-Net, 10 ablations, unaligned wrapper, concept bottleneck (3 seeds x 5 folds) | ~ 2–3 h |
| analysis | `python -m experiments.run_analysis [--only perf ccw audit rfcf decomp unc filter]` | Tables 3, 5–12 | ~ 30 min |
| figures | `python -m src.plotting` | Figures 1–12 | < 1 min |

No GPU is needed (the networks are small; PyTorch runs single-threaded per worker).
Seeds, folds and software versions are recorded (`configs/experiment_config.yaml`,
`data/processed/fold_assignments.csv`, `results/diagnostics/software_versions.json`).

## Guard rails implemented in code

* `read_raw` refuses to run if the workbook checksum changes; nothing writes to `data/raw`.
* `check_no_leakage` is called for every outer and inner split; imputation, scaling, target
  standardisation, hyper-parameters, mediator models and conformal calibration use training rows only.
* RF and CF never enter the primary mean model. The one model that uses them as inputs is saved under
  the name `xgb_rfcf_secondary` and reported as a secondary ablation.
* Zero overtopping is never transformed with an epsilon; it is handled by a separate wet/dry model.
* Concept labels use predictor variables only (never q, RF or CF); thresholds are fixed constants.
