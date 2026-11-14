"""Phase II (revision): record-level data accounting, RF/CF-dependent missingness, variable audit.

Every criterion is evaluated independently on the raw-derived master table, so overlapping exclusion
reasons are visible (the original pipeline assigns only the first applicable reason).
Outputs go to revision/audit and revision/tables.  Nothing in data/ is modified.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .utils import ROOT

AUD = ROOT / "revision" / "audit"
TAB = ROOT / "revision" / "tables"


def load_master():
    m = pd.read_csv(ROOT / "data/processed/clash_master.csv", low_memory=False, dtype={"series": str, "name": str})
    for c in ("remark", "reference", "unusable_reason"):
        m[c] = m[c].fillna("")
    return m


def criteria(m: pd.DataFrame) -> pd.DataFrame:
    c = pd.DataFrame({"name": m["name"], "series": m["series"], "geometry": m["geometry"], "RF": m["RF"].astype(int), "CF": m["CF"].astype(int)})
    c["resp_positive"] = m["response_class"] == "positive"
    c["resp_zero"] = m["response_class"] == "zero"
    c["resp_missing_marker"] = m["response_class"] == "missing"
    c["resp_negative"] = m["response_class"] == "negative"
    c["resp_invalid"] = c["resp_missing_marker"] | c["resp_negative"]
    c["hm0_toe_missing_or_nonpos"] = ~(m["hm0_toe"] > 0)
    c["tm10_toe_missing_or_nonpos"] = ~(m["tm10_toe"] > 0)
    c["h_missing"] = m["h"].isna()
    c["rc_missing"] = m["rc"].isna()
    c["any_required_input_unusable"] = c[["hm0_toe_missing_or_nonpos", "tm10_toe_missing_or_nonpos", "h_missing", "rc_missing"]].any(axis=1)
    c["qstar_undefined"] = c["resp_positive"] & c["hm0_toe_missing_or_nonpos"]
    c["duplicate_name"] = m["name"].duplicated(keep=False)
    c["duplicate_physical_record"] = m["flag_duplicate_physical"]
    c["in_wetdry"] = m["use_wetdry"]
    c["in_regression"] = m["use_regression"]
    # consistency with the original pipeline's flags
    assert (c["in_wetdry"] == (~c["any_required_input_unusable"] & (c["resp_positive"] | c["resp_zero"]))).all()
    assert (c["in_regression"] == (~c["any_required_input_unusable"] & c["resp_positive"])).all()
    return c


def exclusive_accounting(c: pd.DataFrame) -> pd.DataFrame:
    cat = np.select(
        [c.resp_invalid & c.any_required_input_unusable, c.resp_invalid, c.resp_zero & c.any_required_input_unusable,
         c.resp_positive & c.any_required_input_unusable, c.resp_zero, c.resp_positive],
        ["invalid response AND unusable inputs", "invalid response only (missing marker or negative q)",
         "zero discharge, unusable inputs", "positive discharge, unusable inputs",
         "zero discharge, usable (wet/dry only)", "positive discharge, usable (wet/dry + regression)"], "other")
    c = c.assign(category=cat)
    t = c.groupby("category").agg(n=("name", "size"), n_series=("series", "nunique"),
                                  RF4=("RF", lambda s: int((s == 4).sum())), CF4=("CF", lambda s: int((s == 4).sum())),
                                  in_wetdry=("in_wetdry", "sum"), in_regression=("in_regression", "sum")).reset_index()
    t = pd.concat([t, pd.DataFrame([{"category": "TOTAL", "n": len(c), "n_series": c.series.nunique(), "RF4": int((c.RF == 4).sum()),
                                     "CF4": int((c.CF == 4).sum()), "in_wetdry": int(c.in_wetdry.sum()), "in_regression": int(c.in_regression.sum())}])])
    return t, c


def overlap_table(c: pd.DataFrame) -> pd.DataFrame:
    reasons = ["resp_missing_marker", "resp_negative", "hm0_toe_missing_or_nonpos", "tm10_toe_missing_or_nonpos", "h_missing", "rc_missing"]
    rows = []
    for a in reasons:
        r = {"reason": a, "n": int(c[a].sum())}
        for b in reasons:
            r[b] = int((c[a] & c[b]).sum())
        rows.append(r)
    return pd.DataFrame(rows)


def reconciliation(c: pd.DataFrame, m: pd.DataFrame) -> pd.DataFrame:
    excl_wd = ~c.in_wetdry
    rows = [
        ("Total CLASH tests (rows of sheet 20050101)", 10532, len(c)),
        ("Positive discharge", 9442, int(c.resp_positive.sum())),
        ("Zero discharge", 1069, int(c.resp_zero.sum())),
        ("Missing or invalid discharge (20 missing markers + 1 negative)", 21, int(c.resp_invalid.sum())),
        ("Records with unusable required inputs (Hm0,toe, Tm-1,0,toe, h or Rc), any response", None, int(c.any_required_input_unusable.sum())),
        ("  of which also invalid response", None, int((c.any_required_input_unusable & c.resp_invalid).sum())),
        ("  of which valid response (positive or zero)", None, int((c.any_required_input_unusable & ~c.resp_invalid).sum())),
        ("Records excluded from wet/dry model (manuscript: '143 tests with unusable inputs')", 143, int(excl_wd.sum())),
        ("  of which RF = 4", 143, int((excl_wd & (c.RF == 4)).sum())),
        ("Wet/dry observations", 10389, int(c.in_wetdry.sum())),
        ("Positive-discharge regression observations", 9333, int(c.in_regression.sum())),
        ("Positive discharge excluded from regression (unusable inputs)", 109, int((c.resp_positive & ~c.in_regression).sum())),
        ("Test series (all)", 163, c.series.nunique()),
        ("Test series (regression)", 162, c[c.in_regression].series.nunique()),
        ("Reconstructed geometries (regression)", 2046, c[c.in_regression].geometry.nunique()),
        ("Reconstructed geometries (all records)", None, c.geometry.nunique()),
        ("CF = 4 tests (all)", 254, int((c.CF == 4).sum())),
        ("CF = 4 series (all)", 9, c[c.CF == 4].series.nunique()),
        ("CF = 4 tests (regression)", 212, int(((c.CF == 4) & c.in_regression).sum())),
        ("CF = 4 series (regression)", 7, c[(c.CF == 4) & c.in_regression].series.nunique()),
        ("RF = 4 tests (regression)", 637, int(((c.RF == 4) & c.in_regression).sum())),
        ("Common tests RF < 4 and CF < 4 (regression)", 8494, int(((c.RF < 4) & (c.CF < 4) & c.in_regression).sum())),
    ]
    r = m[m.use_regression]
    for lab, d, mv in (("all 10 532 records", m, (32, 311, 166)), ("regression set", r, (None, None, None))):
        gcf = d.groupby("geometry")["CF"].nunique()
        grf = d.groupby("geometry")["RF"].nunique()
        rows += [(f"Geometries with >1 CF value ({lab})", mv[0], int((gcf > 1).sum())),
                 (f"Tests in geometries with >1 CF value ({lab})", mv[1], int(d.geometry.isin(gcf[gcf > 1].index).sum())),
                 (f"Geometries with >1 RF value ({lab})", mv[2], int((grf > 1).sum()))]
    rows += [
             ("Share of regression tests with an inferred laboratory", 0.70, round(float((r.lab != "unknown").mean()), 3)),
             ("Zero-discharge share among valid-response RF = 1 tests (wet/dry set)", 0.058,
              round(float(m[m.use_wetdry & (m.RF == 1)].response_class.eq("zero").mean()), 4)),
             ("Zero-discharge share among valid-response RF = 4 tests (wet/dry set)", 0.165,
              round(float(m[m.use_wetdry & (m.RF == 4)].response_class.eq("zero").mean()), 4))]
    out = pd.DataFrame(rows, columns=["quantity", "manuscript_value", "reproduced_value"])
    out["status"] = [("not reported in manuscript" if mv is None or (isinstance(mv, float) and np.isnan(mv)) else ("match" if np.isclose(float(mv), float(rv), atol=0.005) else "MISMATCH"))
                     for mv, rv in zip(out.manuscript_value, out.reproduced_value)]
    return out


def missingness(c: pd.DataFrame, m: pd.DataFrame) -> dict:
    """Is unusability of inputs related to RF, CF, series, response?  Valid-response records only."""
    v = c[~c.resp_invalid].copy()
    v["unusable"] = v.any_required_input_unusable
    out = {}
    rows = []
    for k in ("RF", "CF"):
        ct = pd.crosstab(v[k], v.unusable)
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        for cat in ct.index:
            n = int(ct.loc[cat].sum())
            u = int(ct.loc[cat].get(True, 0))
            rows.append({"factor": k, "category": cat, "n_valid_response": n, "n_unusable_inputs": u, "pct_unusable": 100 * u / n,
                         "chi2_p_overall": p})
    out["by_rf_cf"] = pd.DataFrame(rows)
    s = v.groupby("series").agg(n=("name", "size"), n_unusable=("unusable", "sum"), RF=("RF", "max"))
    out["by_series"] = s[s.n_unusable > 0].assign(pct=lambda d: 100 * d.n_unusable / d.n).sort_values("n_unusable", ascending=False).reset_index()
    rr = v.groupby(["resp_zero"]).unusable.agg(["size", "sum"]).reset_index().rename(columns={"resp_zero": "zero_discharge", "size": "n", "sum": "n_unusable"})
    out["by_response"] = rr
    # representativeness: RF = 4 records with unusable inputs vs RF = 4 records that enter the models
    mm = m.assign(unusable=c.any_required_input_unusable.values)
    r4 = mm[(mm.RF == 4) & mm.response_class.isin(["positive", "zero"])]
    desc = []
    for col, lab in (("hm0_deep", "Hm0 deep [m]"), ("tm10_deep", "Tm-1,0 deep [s]"), ("h", "h [m]"), ("rc", "Rc [m]"), ("q", "q [m3/s/m]"),
                     ("cotad", "cot alpha_d"), ("gf_raw", "gamma_f"), ("CF", "CF")):
        a, b = r4.loc[r4.unusable, col].dropna(), r4.loc[~r4.unusable, col].dropna()
        p = stats.mannwhitneyu(a, b).pvalue if len(a) > 2 and len(b) > 2 else np.nan
        desc.append({"variable": lab, "unusable_n": len(a), "unusable_median": a.median(), "unusable_iqr": f"{a.quantile(.25):.4g}–{a.quantile(.75):.4g}",
                     "usable_RF4_n": len(b), "usable_RF4_median": b.median(), "usable_RF4_iqr": f"{b.quantile(.25):.4g}–{b.quantile(.75):.4g}",
                     "mann_whitney_p": p})
    desc.append({"variable": "zero-discharge share", "unusable_n": int(r4.unusable.sum()),
                 "unusable_median": float(r4.loc[r4.unusable, "response_class"].eq("zero").mean()),
                 "usable_RF4_n": int((~r4.unusable).sum()), "usable_RF4_median": float(r4.loc[~r4.unusable, "response_class"].eq("zero").mean())})
    desc.append({"variable": "number of series", "unusable_n": r4.loc[r4.unusable, "series"].nunique(), "usable_RF4_n": r4.loc[~r4.unusable, "series"].nunique()})
    desc.append({"variable": "share from series 802", "unusable_median": float((r4.loc[r4.unusable, "series"] == "802").mean()),
                 "usable_RF4_median": float((r4.loc[~r4.unusable, "series"] == "802").mean())})
    out["rf4_representativeness"] = pd.DataFrame(desc)
    return out


def variable_audit(m: pd.DataFrame) -> pd.DataFrame:
    r = m[m.use_regression]
    g = 9.81
    spec = [  # column, source, unit, transformation, expected range, interpretation
        ("q", "q", "m3/s per m", "raw", "> 0 (regression)", "mean overtopping discharge per metre crest"),
        ("hm0_toe", "Hm0 toe", "m", "raw", "> 0", "spectral significant wave height at the toe"),
        ("tm10_toe", "Tm-1,0 toe", "s", "raw", "> 0", "spectral period at the toe"),
        ("q_star", "q, Hm0 toe", "-", "q / sqrt(g Hm0,toe^3)", "1e-9 – 1e-1", "Froude-scaled discharge"),
        ("y", "q, Hm0 toe", "-", "log10 q*", "-9 – -1", "response"),
        ("l0", "Tm-1,0 toe", "m", "g T^2 / (2 pi)", "> 0", "deep-water wavelength from the toe period"),
        ("s0", "Hm0, Tm-1,0 toe", "-", "Hm0 / L0", "0.001 – 0.08", "wave steepness"),
        ("rc_h", "Rc, Hm0", "-", "Rc / Hm0 (signed log1p in model)", "-1 – 5", "relative crest freeboard"),
        ("ac_h", "Ac, Hm0", "-", "Ac / Hm0 (signed log1p)", "-1 – 5", "relative armour-crest freeboard"),
        ("gc_l", "Gc, Tm-1,0", "-", "Gc / L0 (signed log1p)", "0 – 0.5", "relative crest width"),
        ("h_l", "h, Tm-1,0", "-", "h / L0 (signed log1p)", "0 – 1", "relative toe depth"),
        ("ht_h", "ht, Hm0", "-", "ht / Hm0 (signed log1p)", "-1 – 20", "relative toe submergence"),
        ("bt_l", "Bt, Tm-1,0", "-", "Bt / L0 (signed log1p)", "0 – 1", "relative toe width"),
        ("b_l", "B, Tm-1,0", "-", "B / L0 (signed log1p)", "0 – 1", "relative berm width"),
        ("hb_h", "hb, Hm0", "-", "hb / Hm0 (signed log1p)", "-5 – 5", "relative berm submergence"),
        ("tanab", "tan aB", "-", "raw", "0 – 0.2", "berm slope (0 = horizontal or no berm)"),
        ("cotad", "cot ad", "-", "raw", "0 – 10", "lower slope (0 = vertical)"),
        ("cotau", "cot au", "-", "raw", "0 – 10", "upper slope (0 = vertical)"),
        ("gf", "gamma_f", "-", "raw", "0.3 – 1", "roughness/permeability factor"),
        ("beta", "beta", "deg", "raw", "0 – 90", "angle of wave attack"),
        ("fs", "m", "-", "1/m (NaN if m <= 0)", "0 – 0.1", "foreshore slope"),
    ]
    rows = []
    for col, src, unit, tr_, rng, interp in spec:
        x = r[col].astype(float)
        rows.append({"variable": col, "source": src, "unit": unit, "transformation": tr_, "expected_range": rng, "interpretation": interp,
                     "n_missing_regression": int(x.isna().sum()), "min": x.min(), "p01": x.quantile(.01), "median": x.median(),
                     "p99": x.quantile(.99), "max": x.max(), "n_nonfinite": int((~np.isfinite(x.dropna())).sum())})
    va = pd.DataFrame(rows)
    # recomputation checks
    qs = r.q / np.sqrt(g * r.hm0_toe ** 3)
    l0 = g * r.tm10_toe ** 2 / (2 * np.pi)
    checks = pd.DataFrame([
        {"check": "q* recomputed = q / sqrt(g Hm0,toe^3)", "max_abs_rel_diff": float(np.max(np.abs(qs / r.q_star - 1)))},
        {"check": "Y recomputed = log10 q*", "max_abs_rel_diff": float(np.max(np.abs(np.log10(qs) - r.y)))},
        {"check": "L0 recomputed = g Tm-1,0^2 / 2pi", "max_abs_rel_diff": float(np.max(np.abs(l0 / r.l0 - 1)))},
        {"check": "s0 = Hm0/L0", "max_abs_rel_diff": float(np.max(np.abs(r.hm0_toe / l0 / r.s0 - 1)))},
        {"check": "records with Rc < 0 (submerged crest)", "max_abs_rel_diff": int((r.rc < 0).sum())},
        {"check": "records with h <= 0 (emergent toe)", "max_abs_rel_diff": int((r.h <= 0).sum())},
        {"check": "records with Hm0,toe/h > 1 (depth-limited)", "max_abs_rel_diff": int(r.flag_depth_limited_gt1.sum())},
        {"check": "records with s0 > 0.07", "max_abs_rel_diff": int((r.s0 > 0.07).sum())},
        {"check": "records with s0 < 0.002", "max_abs_rel_diff": int((r.s0 < 0.002).sum())},
        {"check": "records with Ac > Rc", "max_abs_rel_diff": int(r.flag_ac_gt_rc.sum())},
        {"check": "records with toe Hm0 = deep Hm0 (possible copy)", "max_abs_rel_diff": int(r.flag_toe_equals_deep.sum())},
        {"check": "records with Tm-1,0 = Tp/1.1 (default ratio, estimated)", "max_abs_rel_diff": int(r.flag_tm10_default_ratio.sum())},
    ])
    return va, checks


def main():
    AUD.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    m = load_master()
    c = criteria(m)
    ex, c2 = exclusive_accounting(c)
    c2.to_csv(AUD / "data_accounting.csv", index=False)
    ex.to_csv(TAB / "S_accounting_exclusive.csv", index=False)
    overlap_table(c).to_csv(TAB / "S_accounting_overlap.csv", index=False)
    rec = reconciliation(c, m)
    rec.to_csv(TAB / "S_accounting_reconciliation.csv", index=False)
    mi = missingness(c, m)
    for k, v in mi.items():
        v.to_csv(TAB / f"S_missingness_{k}.csv", index=False)
    va, chk = variable_audit(m)
    va.to_csv(AUD / "variable_audit.csv", index=False)
    chk.to_csv(AUD / "physical_transformation_checks.csv", index=False)
    print(ex.to_string()); print(rec.to_string()); print(mi["by_rf_cf"].to_string()); print(mi["rf4_representativeness"].to_string())
    print(chk.to_string())


if __name__ == "__main__":
    main()
