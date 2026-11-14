"""Stage A/B: repository inventory, raw-data audit, RF/CF verification, master table and folds."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .concept_ontology import concept_labels, ontology_table
from .data_preprocessing import NUMERIC_RAW, build_master, read_raw, sha256, variable_dictionary
from .geometry_grouping import add_groups, make_folds
from .utils import ROOT, get_logger, load_config, out, record_versions, save_table

log = get_logger("audit")

RF_MEANING = {1: "very reliable: all needed information available, measurements and analysis reliable",
              2: "reliable: some estimations/calculations and/or some uncertainty, overall still reliable",
              3: "less reliable: estimations/calculations and uncertainties lead to the class 'less reliable'",
              4: "not reliable: no acceptable estimations possible and/or faulty measurements or calculations"}
CF_MEANING = {1: "simple: the parameters describe the cross-section exactly or very well",
              2: "quite simple: the parameters describe the cross-section very well, although not exactly",
              3: "quite complicated: the parameters describe the cross-section appreciably, but not exactly",
              4: "very complicated: the cross-section is too complex for the chosen parameters; the representation is unreliable"}


def file_inventory() -> pd.DataFrame:
    rows = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and rel.parts[0] in ("data", "docs") and "processed" not in rel.parts and "interim" not in rel.parts:
            rows.append({"file": str(rel), "format": p.suffix, "bytes": p.stat().st_size, "sha256": sha256(p)[:16]})
    return pd.DataFrame(rows)


def run() -> None:
    cfg = load_config()
    record_versions()
    raw, types, units = read_raw(cfg)
    inv = file_inventory()
    inv["role"] = inv["file"].map(lambda f: "ORIGINAL CLASH database (sheet '20050101': 10532 tests; sheet 'cross-section': plotting helper)"
                                  if f.endswith(".xls") else ("CCW-Net reference paper" if "2611" in f else
                                                              ("checksum of raw workbook" if "SHA256" in f else "research brief")))
    save_table(inv, "table01a_file_inventory")

    master = add_groups(build_master(cfg), cfg["data"]["geometry_sig_digits"])
    labels = concept_labels(master)
    labels.loc[~master["usable_inputs"]] = -1
    master = pd.concat([master, labels.add_prefix("concept_")], axis=1)
    folds = make_folds(master[master["use_regression"]], cfg["experiment"]["n_outer_folds"], cfg["experiment"]["seed"])
    folds_all = pd.DataFrame({"name": master["name"]})
    for c in folds.columns:
        folds_all[c] = -1
        folds_all.loc[folds.index, c] = folds[c]
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    master.to_csv(ROOT / cfg["data"]["master_file"], index=False)
    folds_all.to_csv(ROOT / cfg["data"]["folds_file"], index=False)
    assert len(master) == len(raw), "master table must retain every raw record"

    # ---- Table 1: variable dictionary + ranges ------------------------------------------------
    vd = variable_dictionary(master)
    num = raw[NUMERIC_RAW].apply(pd.to_numeric, errors="coerce")
    rng = num.agg(["min", "median", "max"]).T
    rng["n_text_cells"] = (types[NUMERIC_RAW] == 1).sum()
    rng["n_zero"] = (num == 0).sum()
    rng["n_negative"] = (num < 0).sum()
    vd = vd.merge(rng.reset_index().rename(columns={"index": "raw_variable"}), on="raw_variable", how="left")
    save_table(vd, "table01_variable_dictionary")

    # ---- Table 2: RF / CF distribution and completeness ----------------------------------------
    rows = []
    for k, meaning in (("RF", RF_MEANING), ("CF", CF_MEANING)):
        for v in (1, 2, 3, 4):
            sel = master[k] == v
            sub = master[sel]
            rows.append({"factor": k, "category": v, "documented_meaning": meaning[v], "n_all": int(sel.sum()),
                         "pct_all": 100 * sel.mean(), "n_regression": int(sub["use_regression"].sum()),
                         "n_zero_q": int((sub["response_class"] == "zero").sum()),
                         "pct_zero_q": 100 * (sub["response_class"] == "zero").mean(),
                         "n_unusable": int((~sub["usable_inputs"]).sum()),
                         "pct_reference_missing": 100 * sub["flag_reference_missing"].mean(),
                         "pct_deep_missing": 100 * sub["flag_deep_missing"].mean(),
                         "pct_Tm10_default_ratio": 100 * sub["flag_tm10_default_ratio"].mean(),
                         "pct_remark_do_not_use": 100 * sub["flag_remark_do_not_use"].mean(),
                         "n_series": sub["series"].nunique(), "n_geometries": sub["geometry"].nunique()})
        rows.append({"factor": k, "category": "missing", "n_all": int(master[k].isna().sum())})
        rows.append({"factor": k, "category": "out of range", "n_all": int(master[f"{k}_out_of_range"].sum())})
    save_table(pd.DataFrame(rows), "table02_rf_cf_distribution")
    joint = pd.crosstab(master["RF"].astype(int), master["CF"].astype(int), margins=True)
    joint.to_csv(out("tables", "table02b_rf_cf_joint_all.csv"))
    jr = pd.crosstab(master.loc[master.use_regression, "RF"].astype(int), master.loc[master.use_regression, "CF"].astype(int), margins=True)
    jr.to_csv(out("tables", "table02c_rf_cf_joint_regression.csv"))

    # ---- consistency of RF / CF within groups --------------------------------------------------
    cons = []
    for level in ("geometry", "substructure", "geometry_coarse", "series", "source"):
        g = master.groupby(level)
        multi = g.size() > 1
        cons.append({"grouping": level, "n_groups": int(g.ngroups), "n_groups_with_>1_test": int(multi.sum()),
                     "groups_with_>1_CF_value": int((g["CF"].nunique() > 1).sum()),
                     "groups_with_>1_RF_value": int((g["RF"].nunique() > 1).sum()),
                     "max_CF_range_in_group": float((g["CF"].max() - g["CF"].min()).max()),
                     "records_in_CF_inconsistent_groups": int(master[level].map(g["CF"].nunique() > 1).sum())})
    save_table(pd.DataFrame(cons), "table02d_rf_cf_within_group_consistency")
    gcf = master.groupby("geometry")["CF"].agg(["min", "max", "nunique", "size"])
    gcf[gcf["nunique"] > 1].to_csv(out("diagnostics", "cf_discrepancies_within_geometry.csv"))
    # internal consistency of ratings among physically duplicated records
    phys = ["hm0_toe", "tm10_toe", "h", "ht", "bt", "gf_raw", "cotad", "cotau", "rc", "b_berm", "hb", "tanab", "ac", "gc", "m", "beta_deg", "q"]
    dup = master[master["flag_duplicate_physical"]].groupby(phys, dropna=False)
    dup_summary = {"n_duplicate_sets": int(dup.ngroups), "n_records": int(master["flag_duplicate_physical"].sum()),
                   "sets_with_different_RF": int((dup["RF"].nunique() > 1).sum()),
                   "sets_with_different_CF": int((dup["CF"].nunique() > 1).sum()),
                   "sets_spanning_series": int((dup["series"].nunique() > 1).sum())}

    # ---- unusable / flagged records -------------------------------------------------------------
    master.loc[master["unusable_reason"] != "", ["name", "RF", "CF", "q_raw_text", "q", "hm0_toe", "tm10_toe",
                                                 "response_class", "unusable_reason"]].to_csv(out("diagnostics", "records_not_in_regression.csv"), index=False)
    flags = [c for c in master.columns if c.startswith("flag_")]
    fl = pd.DataFrame({"flag": flags, "n": [int(master[c].sum()) for c in flags],
                       "pct": [100 * master[c].mean() for c in flags],
                       "n_RF4": [int((master[c] & master.flag_rf4).sum()) for c in flags],
                       "n_CF4": [int((master[c] & master.flag_cf4).sum()) for c in flags]})
    save_table(fl, "table02e_flag_counts")
    save_table(ontology_table(labels[master["use_regression"]]), "tableS_concept_ontology")
    grp = pd.DataFrame([{"identifier": c, "origin": o, "n_unique_all": master[c].nunique(),
                         "n_unique_regression": master.loc[master.use_regression, c].nunique(),
                         "median_tests_per_group": float(master.loc[master.use_regression, c].value_counts().median()),
                         "max_tests_per_group": int(master.loc[master.use_regression, c].value_counts().max())}
                        for c, o in (("series", "original (Name prefix)"), ("source", "original Reference text; series number where blank"),
                                     ("lab", "INFERRED by keyword from Reference (approximate; 'unknown' where blank)"),
                                     ("geometry", "RECONSTRUCTED from water-level-invariant geometry descriptors"),
                                     ("substructure", "RECONSTRUCTED: geometry without crest levels"),
                                     ("geometry_coarse", "RECONSTRUCTED: slopes, roughness, berm/toe presence"),
                                     ("family", "RULE-BASED structural family"))])
    save_table(grp, "table01b_identifiers_and_groups")
    summary = {"n_raw_rows": int(len(raw)), "n_columns": int(raw.shape[1]), "units": dict(zip(raw.columns, units)),
               "response_class": master["response_class"].value_counts().to_dict(),
               "q_text_markers": master.loc[master.q_raw_text != "", "q_raw_text"].value_counts().to_dict(),
               "n_use_regression": int(master.use_regression.sum()), "n_use_wetdry": int(master.use_wetdry.sum()),
               "unusable_reasons": master["unusable_reason"].value_counts().to_dict(),
               "exact_duplicate_rows_excluding_name": int(raw.drop(columns=["Name"]).duplicated().sum()),
               "physical_duplicates": dup_summary,
               "RF_values": sorted(master["RF"].dropna().unique().tolist()), "CF_values": sorted(master["CF"].dropna().unique().tolist()),
               "family_counts_regression": master.loc[master.use_regression, "family"].value_counts().to_dict(),
               "lab_counts_regression": master.loc[master.use_regression, "lab"].value_counts().to_dict(),
               "min_positive_q": float(master.loc[master.q > 0, "q"].min())}
    with open(out("diagnostics", "data_audit_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1, default=str)
    log.info("master %s rows; regression %s; wet/dry %s", len(master), summary["n_use_regression"], summary["n_use_wetdry"])


if __name__ == "__main__":
    run()
