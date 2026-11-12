"""Read the raw CLASH workbook and build the master table.

Nothing here removes a record.  Every one of the raw rows is kept in the master
table together with flags and, where a record cannot enter a given model, the
explicit reason (`unusable_reason`).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .utils import ROOT, load_config

RAW_COLUMNS = ["Name", "Hm0 deep", "Tp deep", "Tm deep", "Tm-1,0 deep", "h deep", "m", "b", "h", "Hm0 toe",
               "Tp toe", "Tm toe", "Tm-1,0 toe", "ht", "Bt", "gf", "cotad", "cotau", "cotaexcl", "cotaincl",
               "Rc", "B", "hb", "tanaB", "Bh", "Ac", "Gc", "RF", "CF", "q", "Pow", "Remark", "Reference"]
RENAME = {"Hm0 deep": "hm0_deep", "Tp deep": "tp_deep", "Tm deep": "tm_deep", "Tm-1,0 deep": "tm10_deep",
          "h deep": "h_deep", "m": "m", "b": "beta_deg", "h": "h", "Hm0 toe": "hm0_toe", "Tp toe": "tp_toe",
          "Tm toe": "tm_toe", "Tm-1,0 toe": "tm10_toe", "ht": "ht", "Bt": "bt", "gf": "gf_raw", "cotad": "cotad",
          "cotau": "cotau", "cotaexcl": "cotaexcl", "cotaincl": "cotaincl", "Rc": "rc", "B": "b_berm", "hb": "hb",
          "tanaB": "tanab", "Bh": "bh", "Ac": "ac", "Gc": "gc", "RF": "RF", "CF": "CF", "q": "q", "Pow": "pow",
          "Remark": "remark", "Reference": "reference", "Name": "name"}
NUMERIC_RAW = [c for c in RAW_COLUMNS if c not in ("Name", "Remark", "Reference")]


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (values, cell-types, units).  Cells are returned exactly as stored."""
    import xlrd
    cfg = cfg or load_config()
    path = ROOT / cfg["data"]["raw_file"]
    expected = open(ROOT / cfg["data"]["raw_sha256"]).read().split()[0]
    if sha256(path) != expected:
        raise RuntimeError("Raw workbook checksum mismatch: the raw database must never be modified.")
    sh = xlrd.open_workbook(path).sheet_by_name(cfg["data"]["raw_sheet"])
    header = [sh.cell_value(0, c) for c in range(sh.ncols)]
    if header != RAW_COLUMNS:
        raise RuntimeError(f"Unexpected header in raw sheet: {header}")
    units = [sh.cell_value(1, c) for c in range(sh.ncols)]
    vals = [[sh.cell_value(r, c) for c in range(sh.ncols)] for r in range(2, sh.nrows)]
    typs = [[sh.cell_type(r, c) for c in range(sh.ncols)] for r in range(2, sh.nrows)]
    return pd.DataFrame(vals, columns=header), pd.DataFrame(typs, columns=header), units


def dimensionless_discharge(q, hm0, g: float = 9.81):
    """q* = q / sqrt(g Hm0^3); q in m3/s per m, Hm0 in m -> dimensionless."""
    q = np.asarray(q, float)
    hm0 = np.asarray(hm0, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return q / np.sqrt(g * hm0 ** 3)


def response_class(q_raw: pd.Series, q_num: pd.Series) -> pd.Series:
    """positive / zero / negative / missing (blank or non-numeric marker)."""
    cls = pd.Series("positive", index=q_num.index)
    cls[q_num == 0] = "zero"
    cls[q_num < 0] = "negative"
    cls[q_num.isna()] = "missing"
    return cls


def signed_log1p(x):
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def build_master(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    g = cfg["data"]["gravity"]
    raw, _, _ = read_raw(cfg)
    df = pd.DataFrame({"raw_row": np.arange(len(raw)) + 3})  # 1-based Excel row
    df["name"] = raw["Name"]
    df["q_raw_text"] = raw["q"].where(raw["q"].map(lambda v: isinstance(v, str)), "")
    for c in NUMERIC_RAW:
        df[RENAME[c]] = pd.to_numeric(raw[c], errors="coerce")
    df["remark"] = raw["Remark"].astype(str)
    df["reference"] = raw["Reference"].astype(str)
    df["series"] = df["name"].str.split("-").str[0]

    # ---- quality metadata: never imputed, never converted ------------------------------
    for k in ("RF", "CF"):
        df[f"{k}_missing"] = df[k].isna()
        df[f"{k}_out_of_range"] = df[k].notna() & ~df[k].isin([1, 2, 3, 4])
    df["flag_rf4"] = df["RF"] == 4
    df["flag_cf4"] = df["CF"] == 4
    df["flag_rf4_and_cf4"] = df["flag_rf4"] & df["flag_cf4"]
    df["flag_rf4_or_cf4"] = df["flag_rf4"] | df["flag_cf4"]

    # ---- response ----------------------------------------------------------------------
    df["response_class"] = response_class(raw["q"], df["q"])
    hm0_ok = df["hm0_toe"] > 0
    t_ok = df["tm10_toe"] > 0
    df["q_star"] = np.where((df["q"] > 0) & hm0_ok, dimensionless_discharge(df["q"], df["hm0_toe"].where(hm0_ok), g), np.nan)
    df["y"] = np.log10(df["q_star"])
    reason = pd.Series("", index=df.index)
    reason[df["response_class"] == "missing"] = "q missing or non-numeric marker"
    reason[df["response_class"] == "negative"] = "q negative"
    reason[(reason == "") & ~hm0_ok] = "Hm0,toe missing or non-positive (q* undefined)"
    reason[(reason == "") & ~t_ok] = "Tm-1,0,toe missing or non-positive (wavelength undefined)"
    reason[(reason == "") & df["h"].isna()] = "toe water depth h missing"
    reason[(reason == "") & df["rc"].isna()] = "crest freeboard Rc missing"
    df["unusable_reason"] = reason
    df["usable_inputs"] = reason == ""
    df["use_wetdry"] = df["usable_inputs"] & df["response_class"].isin(["positive", "zero"])
    df["use_regression"] = df["usable_inputs"] & (df["response_class"] == "positive")
    df.loc[df["usable_inputs"] & (df["response_class"] == "zero"), "unusable_reason"] = \
        "zero overtopping: log10(q*) undefined; used in wet/dry model only"

    # ---- dimensionless physical predictors (Froude scaling with toe conditions) --------
    H = df["hm0_toe"].where(hm0_ok)
    L0 = g * df["tm10_toe"].where(t_ok) ** 2 / (2 * np.pi)
    df["l0"] = L0
    df["rc_h"] = df["rc"] / H
    df["ac_h"] = df["ac"] / H
    df["gc_l"] = df["gc"] / L0
    df["s0"] = H / L0
    df["h_l"] = df["h"] / L0
    df["ht_h"] = df["ht"] / H
    df["bt_l"] = df["bt"] / L0
    df["b_l"] = df["b_berm"] / L0
    df["hb_h"] = df["hb"] / H
    df["gf"] = df["gf_raw"]
    df["beta"] = df["beta_deg"]
    df["fs"] = np.where(df["m"] > 0, 1.0 / df["m"], np.nan)  # foreshore tan; m<=0 is not a valid slope
    df["h_h"] = df["h"] / H

    # ---- flags used only for quality evaluation ----------------------------------------
    rem = df["remark"].str.lower()
    df["flag_remark_do_not_use"] = rem.str.contains("do not use")
    df["flag_wind"] = rem.str.contains("v =")
    df["flag_prototype"] = rem.str.contains("prototype")
    df["flag_chute"] = rem.str.contains("shute")
    df["flag_reference_missing"] = df["reference"].str.strip() == ""
    df["flag_deep_missing"] = df["hm0_deep"].isna() | df["tm10_deep"].isna()
    with np.errstate(invalid="ignore", divide="ignore"):
        df["flag_tm10_default_ratio"] = (np.abs(df["tp_toe"] / df["tm10_toe"] - 1.1) < 1e-4)
        df["flag_tm_default_ratio"] = (np.abs(df["tp_toe"] / df["tm_toe"] - 1.2) < 1e-4)
        df["flag_toe_equals_deep"] = (np.abs(df["hm0_toe"] / df["hm0_deep"] - 1) < 1e-9)
        df["flag_depth_limited_gt1"] = (df["hm0_toe"] / df["h"] > 1.0) & (df["h"] > 0)
    df["flag_m_missing"] = df["m"].isna() | (df["m"] <= 0)
    df["flag_beta_missing"] = df["beta_deg"].isna()
    df["flag_gf_missing"] = df["gf_raw"].isna()
    df["flag_toe_emergent"] = df["h"] <= 0
    df["flag_ac_gt_rc"] = df["ac"] > df["rc"] + 1e-9
    phys = ["hm0_toe", "tm10_toe", "h", "ht", "bt", "gf_raw", "cotad", "cotau", "rc", "b_berm", "hb", "tanab",
            "ac", "gc", "m", "beta_deg", "q"]
    df["flag_duplicate_physical"] = df.duplicated(subset=phys, keep=False)
    df["n_estimated_indicators"] = (df[["flag_tm10_default_ratio", "flag_tm_default_ratio", "flag_deep_missing",
                                        "flag_m_missing", "flag_beta_missing", "flag_gf_missing"]].sum(axis=1))
    df["flag_questionable"] = (df["response_class"].isin(["negative"]) | df["flag_depth_limited_gt1"]
                               | df["flag_toe_emergent"] | (df["hm0_toe"] <= 0) | (df["tm10_toe"] <= 0))
    # robust response outlier flag (descriptive only, never used for exclusion)
    y = df["y"]
    med, mad = y.median(), (y - y.median()).abs().median() * 1.4826
    df["flag_y_outlier_mad4"] = (y - med).abs() > 4 * mad
    return df


VARIABLE_DICTIONARY = [
    # raw name, column, meaning, unit, role, dtype, transformation, eligible predictor, quality-only
    ("Name", "name", "Test identifier 'SSS-TTT': SSS = test-series number, TTT = test number within the series", "-", "identifier", "text", "split into series", False, False),
    ("Hm0 deep", "hm0_deep", "Spectral significant wave height at deep water", "m", "hydraulic (offshore)", "numeric", "none (not used: 4.1 % missing; toe conditions are used)", False, True),
    ("Tp deep", "tp_deep", "Peak wave period, deep water", "s", "hydraulic (offshore)", "numeric", "none", False, True),
    ("Tm deep", "tm_deep", "Mean wave period, deep water", "s", "hydraulic (offshore)", "numeric", "none", False, True),
    ("Tm-1,0 deep", "tm10_deep", "Spectral wave period Tm-1,0, deep water", "s", "hydraulic (offshore)", "numeric", "none", False, True),
    ("h deep", "h_deep", "Water depth at deep water", "m", "hydraulic (offshore)", "numeric", "none", False, False),
    ("m", "m", "Foreshore slope 1:m (cotangent); 1000 is the conventional code for a (near-)horizontal foreshore", "-", "foreshore geometry", "numeric", "fs = 1/m", True, False),
    ("b", "beta_deg", "Angle of wave attack relative to the structure normal (beta)", "deg", "hydraulic", "numeric", "none", True, False),
    ("h", "h", "Water depth at the toe of the structure", "m", "hydraulic", "numeric", "h_l = h/L0", True, False),
    ("Hm0 toe", "hm0_toe", "Spectral significant wave height at the toe", "m", "hydraulic / scaling", "numeric", "scales q, Rc, Ac, ht, hb; s0 = Hm0/L0", True, False),
    ("Tp toe", "tp_toe", "Peak period at the toe", "s", "hydraulic", "numeric", "none (collinear with Tm-1,0)", False, True),
    ("Tm toe", "tm_toe", "Mean period at the toe", "s", "hydraulic", "numeric", "none (collinear with Tm-1,0)", False, True),
    ("Tm-1,0 toe", "tm10_toe", "Spectral period Tm-1,0 at the toe", "s", "hydraulic / scaling", "numeric", "L0 = g T^2 / (2 pi)", True, False),
    ("ht", "ht", "Water depth on the toe berm", "m", "structural", "numeric", "ht_h = ht/Hm0", True, False),
    ("Bt", "bt", "Width of the toe berm", "m", "structural", "numeric", "bt_l = Bt/L0", True, False),
    ("gf", "gf", "Roughness/permeability factor gamma_f of the armour (1 = smooth impermeable)", "-", "structural", "numeric", "none", True, False),
    ("cotad", "cotad", "Cotangent of the slope below the berm (down slope)", "-", "structural", "numeric", "none", True, False),
    ("cotau", "cotau", "Cotangent of the slope above the berm (upper slope); negative = overhanging", "-", "structural", "numeric", "none", True, False),
    ("cotaexcl", "cotaexcl", "Mean cotangent excluding the berm (derived from the other geometry and the water level)", "-", "structural (derived)", "numeric", "not used: deterministic function of other predictors and of Rc", False, False),
    ("cotaincl", "cotaincl", "Mean cotangent including the berm (derived)", "-", "structural (derived)", "numeric", "not used (as above)", False, False),
    ("Rc", "rc", "Crest freeboard relative to still water level", "m", "structural / hydraulic", "numeric", "rc_h = Rc/Hm0", True, False),
    ("B", "b_berm", "Berm width", "m", "structural", "numeric", "b_l = B/L0", True, False),
    ("hb", "hb", "Water depth on the berm (negative = berm above still water)", "m", "structural", "numeric", "hb_h = hb/Hm0", True, False),
    ("tanaB", "tanab", "Tangent of the berm slope", "-", "structural", "numeric", "none", True, False),
    ("Bh", "bh", "Horizontal berm width (derived from B and tanaB)", "m", "structural (derived)", "numeric", "not used (duplicate of B)", False, False),
    ("Ac", "ac", "Armour crest freeboard relative to still water level", "m", "structural", "numeric", "ac_h = Ac/Hm0", True, False),
    ("Gc", "gc", "Crest (armour) width", "m", "structural", "numeric", "gc_l = Gc/L0", True, False),
    ("RF", "RF", "Reliability Factor of the test, ordinal 1 (very reliable) ... 4 (not reliable)", "ordinal", "quality metadata", "ordinal", "none; never imputed", False, True),
    ("CF", "CF", "Complexity Factor of the structure section, ordinal 1 (simple) ... 4 (very complex)", "ordinal", "quality metadata", "ordinal", "none; never imputed", False, True),
    ("q", "q", "Mean overtopping discharge per metre crest width", "m3/s/m", "response", "numeric", "q* = q/sqrt(g Hm0^3); Y = log10 q* (positive q only)", False, False),
    ("Pow", "pow", "Percentage of waves overtopping", "%", "secondary response", "numeric", "not used (93 % missing; a response, not a predictor)", False, False),
    ("Remark", "remark", "Free-text remark (wind speed, prototype, chute, 'do not use for neural network')", "text", "quality metadata", "text", "parsed into flags", False, True),
    ("Reference", "reference", "Literature/report reference of the test series (blank for confidential data)", "text", "source metadata", "text", "source id; laboratory inferred by keyword (approximate)", False, True),
]


def variable_dictionary(master: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw_name, col, meaning, unit, role, dtype, transf, elig, qonly in VARIABLE_DICTIONARY:
        s = master[col]
        miss = (s.astype(str).str.strip() == "").mean() if dtype == "text" else s.isna().mean()
        rows.append({"raw_variable": raw_name, "column": col, "engineering_meaning": meaning, "unit": unit,
                     "physical_role": role, "data_type": dtype, "missing_pct": round(100 * miss, 2),
                     "transformation": transf, "eligible_physical_predictor": elig,
                     "quality_evaluation_only": qonly})
    return pd.DataFrame(rows)
