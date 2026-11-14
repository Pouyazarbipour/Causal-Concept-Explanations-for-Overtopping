# Reproducibility status

| Component | How it was checked | Status |
|---|---|---|
| Raw data integrity | SHA-256 checksum checked at every load (`src/data_preprocessing.read_raw`) | reproduced |
| Record counts (10 532 / 9 442 / 1 069 / 21 / 10 389 / 9 333 / 163 / 162 / 2 046) | independent recomputation (`src/data_validation.py`, `revision/tables/S_accounting_reconciliation.csv`) | reproduced. Labels of the "143" and "32/311/166" counts corrected (audit A2, A3) |
| Physical transformations (q*, Y, L0, s0) | recomputed from raw columns | max relative difference < 1e-12 |
| Outer folds | `data/processed/fold_assignments.csv` reused unchanged | identical |
| XGBoost baseline out-of-fold predictions | refitted from the stored per-fold hyper-parameters: (i) in the two-stage module (sequential), (ii) in the geometry module (4 threads) | identical (maximum absolute difference 0.0); pooled RMSE 0.5015 |
| RF, CatBoost, MLP, CCW-Net out-of-fold predictions | loaded from the original predictions; metrics recomputed independently | metrics reproduced (`numerical_consistency_audit.csv`). Not refitted from scratch for runtime reasons; the checkpointed fold caches are in `results/models` |
| Optuna studies | stored trial histories (`results/models/optuna`); TPE sampler seeded per fold | deterministic given the seed. Not re-run end to end |
| RF/CF statistics (variance ratios, strata, trends, CF geometry level, replicate pairs) | independent recomputation of point estimates | reproduced |
| Policies (v1) | recomputed from `policy__xgb.npz` | reproduced (ΔRMSE, coverage, widths) |
| Wet/dry AUC (v1) | recomputed from `wetdry__xgb_geometry.csv` | reproduced (0.92 / 0.81) |
| CCW-Net | retrained after the reference-code check (report 08); tables regenerated | reproduced in the previous revision step |

## How to reproduce the revision

```bash
python -m src.data_validation              # accounting, missingness, variable audit
python -m src.repository_inventory         # manifest and inventory
python -m src.consistency_audit            # numerical consistency
python -m experiments.run_optimized --partitions
python -m src.wet_dry_model
python -m src.geometry_sensitivity
python -m src.residual_robustness
python -m experiments.run_policies_v2 --mode fixed tuned
python -m src.conformal_uncertainty
python -m src.engineering_impact
python -m src.model_comparison
python -m src.plotting_revision
python manuscript/revision/build_revised.py
node manuscript/build_docx.js manuscript/revision/revised_manuscript.json manuscript/revision/revised_manuscript.docx
```

- Seeds: base seed 20261112; per-fold seed = base seed + fold; calibration seeds = base seed + 101 × repetition.
- Every runner checkpoints per fold (`revision/outputs/fold_cache`, `revision/outputs/policies/cache`,
  `revision/outputs/wet_dry/cache`), so an interrupted run resumes.
- Environment: Python 3.13, xgboost, catboost, lightgbm (installed for the revision), scikit-learn, optuna,
  statsmodels, torch. Minimum versions are listed in `requirements.txt` (lightgbm added).
