"""Phase I (revision): manifest of the original results and inventory of manuscript items -> code -> data.

Writes revision/audit/original_results_manifest.csv and revision/audit/existing_analysis_inventory.csv.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from .utils import ROOT

AUD = ROOT / "revision" / "audit"

# generating function for each original result table / figure (from experiments/run_analysis.py, src/plotting.py)
GEN = {
    "table01": ("src/data_audit.py", "variable dictionary and file inventory"),
    "table02": ("src/data_audit.py", "RF/CF distributions, joint tables, flag counts"),
    "table03": ("experiments/run_analysis.py::exp_perf", "out-of-fold accuracy (pooled), by RF/CF"),
    "table04": ("experiments/run_analysis.py::exp_perf", "Optuna best hyper-parameters; inner CV and fold-mean outer RMSE"),
    "table05": ("experiments/run_analysis.py::exp_rfcf", "RF/CF residual strata, trends, ordinal models, Gamma GLM, repeatability"),
    "table06": ("experiments/run_analysis.py::exp_rfcf", "geometry-level CF analysis"),
    "table07": ("experiments/run_analysis.py::exp_rfcf", "RF x CF joint table"),
    "table08": ("experiments/run_analysis.py::exp_decomp", "REML decomposition; replicate comparison"),
    "table09": ("experiments/run_analysis.py::exp_ccw", "CCW-Net alignment and ablation"),
    "table10": ("experiments/run_analysis.py::exp_ccw", "CCW-Net concept attribution"),
    "table11": ("experiments/run_analysis.py::exp_ccw", "concept vs feature attribution"),
    "table12": ("experiments/run_analysis.py::exp_ccw", "explanation quality vs data quality"),
    "table13": ("experiments/run_analysis.py::exp_policy", "data-selection policies"),
    "table14": ("experiments/run_analysis.py::exp_policy", "policy calibration by quality category"),
    "table15": ("experiments/run_analysis.py::exp_policy", "policy bias by structure family"),
    "tableS_ccw": ("experiments/run_analysis.py::exp_ccw", "CCW-Net supplementary"),
    "tableS_interaction": ("experiments/run_analysis.py::exp_ccw", "CCW-Net interaction pairs"),
    "tableS_concept": ("src/data_audit.py", "concept ontology"),
    "tableS_conformal": ("experiments/run_analysis.py::exp_unc", "conformal methods by quality"),
    "tableS_coverage": ("experiments/run_analysis.py::exp_unc", "coverage by RF x CF cell"),
    "tableS_interval": ("experiments/run_analysis.py::exp_unc", "interval width by RF/CF"),
    "tableS_policy": ("experiments/run_analysis.py::exp_policy", "policy training variance model"),
    "tableS_wet": ("experiments/run_analysis.py::exp_perf + experiments/run_models.py::wet_dry", "wet/dry model AUC/Brier"),
}

# manuscript items of the original draft (manuscript/build_content.py, commit in original/ORIGINAL_COMMIT.txt)
MANUSCRIPT = [
    ("Table 1", "RF and CF distribution", "table02_rf_cf_distribution", "src/data_audit.py", "data/processed/clash_master.csv"),
    ("Table 2", "Optuna optimisation summary (inner CV and FOLD-MEAN outer RMSE)", "table04_optuna_best_hyperparameters", "experiments/run_analysis.py::exp_perf", "results/models/optuna/*"),
    ("Table 3", "Policy definitions", "static text in build_content.py", "src/data_selection.py", "-"),
    ("Table 4", "Out-of-fold accuracy (POOLED RMSE; per-seed mean for networks)", "table03_model_performance", "experiments/run_analysis.py::exp_perf", "results/predictions/geometry__*.npz"),
    ("Table 5", "Residuals, repeatability, variance ratios by RF", "table05a, table05h, table05d", "experiments/run_analysis.py::exp_rfcf", "results/predictions/geometry__xgb.npz"),
    ("Table 6", "Geometry-level errors by CF", "table06a_cf_geometry_level", "experiments/run_analysis.py::exp_rfcf", "results/predictions/geometry__xgb.npz, random__xgb.npz"),
    ("Table 7", "REML decomposition", "table08_hierarchical_error_decomposition", "experiments/run_analysis.py::exp_decomp", "results/predictions/geometry__xgb.npz"),
    ("Table 8", "Policies on identical held-out tests", "table13_data_selection_policies", "experiments/run_analysis.py::exp_policy", "results/predictions/policy__xgb.npz"),
    ("Table 9", "CCW-Net alignment and ablation", "table09_ccw_alignment_and_ablation", "experiments/run_analysis.py::exp_ccw", "results/diagnostics/ccw_metrics.csv"),
    ("Fig. 1", "RF/CF distribution", "fig02_rf_cf_distribution", "src/plotting.py::fig02", "table02"),
    ("Fig. 2", "Measured vs predicted, all models (seed ensemble)", "fig03_measured_vs_predicted_all_models", "src/plotting.py::fig03", "results/predictions"),
    ("Fig. 3", "Residuals by RF and CF", "fig08_residuals_by_rf_cf", "src/plotting.py::fig08", "table05a"),
    ("Fig. 4", "RF/CF validity panels", "fig09_rf_cf_ordinal_validation", "src/plotting.py::fig09", "table05*, table06*, table07"),
    ("Fig. 5", "Error decomposition", "fig10_error_decomposition", "src/plotting.py::fig10", "table08*"),
    ("Fig. 6", "Policies", "fig13_data_selection_policies", "src/plotting.py::fig13", "table13"),
    ("Fig. 7", "Coverage by RF/CF under policies", "fig14_calibration_by_quality_category", "src/plotting.py::fig14", "table14"),
    ("Fig. 8", "Concept explanations", "fig12_concept_attribution_and_alignment", "src/plotting.py::fig12", "table09-12"),
    ("Text 2.1", "9 442 / 1 069 / 21 / 10 389 / 143 counts", "HARD-CODED in build_content.py", "-", "verified in revision/tables/S_accounting_reconciliation.csv"),
    ("Text 2.2", "CF=4: 254 tests, 9 series; 212 tests, 7 series; 32 geometries (311 tests); 166 geometries", "HARD-CODED in build_content.py", "-", "verified in S_accounting_reconciliation.csv (32/311/166 refer to ALL records)"),
    ("Text 2.3", "162 series, 2 046 geometries, 70 % labs", "HARD-CODED in build_content.py", "-", "verified"),
    ("Text 3.2", "1 136 near-replicate pairs", "HARD-CODED in build_content.py", "experiments/run_analysis.py::exp_rfcf", "table05h"),
    ("Text 4.2", "zero share 16.5 % vs 5.8 %", "HARD-CODED in build_content.py", "experiments/run_analysis.py::exp_rfcf", "table05g"),
    ("Text 4.5", "8 494 common tests", "HARD-CODED in build_content.py", "-", "verified"),
    ("Not in manuscript", "fig01, fig04-07, fig11, fig15, figS1 and tables 03b, 05b-c, 05e-g, 06b, 07, 08b, 10-12, 14-15, S*", "results/figures, results/tables", "src/plotting.py, experiments/run_analysis.py", "-"),
]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def main():
    AUD.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted((ROOT / "original" / "original_results").rglob("*")):
        if p.is_file():
            stem = p.stem
            gen = next((v for k, v in GEN.items() if stem.startswith(k)), ("experiments/run_models.py / run_ccw.py / run_policies.py", "model predictions or Optuna output"))
            info = ""
            if p.suffix == ".csv":
                try:
                    d = pd.read_csv(p)
                    info = f"{d.shape[0]} rows x {d.shape[1]} cols"
                except Exception:  # noqa: BLE001
                    info = "unreadable"
            rows.append({"file": str(p.relative_to(ROOT)), "sha256_16": sha(p), "size_bytes": p.stat().st_size, "shape": info,
                         "generated_by": gen[0], "content": gen[1]})
    pd.DataFrame(rows).to_csv(AUD / "original_results_manifest.csv", index=False)
    pd.DataFrame(MANUSCRIPT, columns=["manuscript_item", "content", "source_output", "generating_code", "data_source"]).to_csv(
        AUD / "existing_analysis_inventory.csv", index=False)
    print(len(rows), "files in manifest")


if __name__ == "__main__":
    main()
