"""Phase X (revision): EurOtop benchmark and engineering decision-impact analysis.

EurOtop (2018) mean-value formulae, applied ONLY inside explicitly stated domains:
  S  single-slope structures (no berm, lower = upper slope, no crest wall: Ac >= Rc), 1 <= cot(alpha) <= 7,
     Rc > 0, 0.1 < Rc/Hm0 < 3.5, xi_m-1,0 < 5, beta <= 80 deg, 0.001 <= s0 <= 0.07, gamma_f documented:
       q* = 0.023/sqrt(tan a) * xi * exp(-(2.7 Rc / (xi Hm0 gb gf gbeta))^1.3)            (EurOtop eq. 5.10)
       with maximum 0.09 exp(-(1.5 Rc / (Hm0 gf gbeta))^1.3)                                 (eq. 5.11)
     gb = 1 (no berm), gbeta = 1 - 0.0033|beta|, gf from the database; rough slopes (gf < 0.9) use
     gf,surging = gf + (xi - 1.8)(1 - gf)/8.2 for xi > 1.8 and the crest-width reduction
     Cr = min(1, 3.06 exp(-1.5 Gc/Hm0)) (EurOtop eqs. 6.5, 6.8).
  V  plain vertical walls (cot = 0 for both parts, no toe mound: ht >= 0.95 h, Bt = 0) without foreshore
     influence (h/Hm0 >= 4), 0.1 < Rc/Hm0 < 3.5, normal or near-normal attack (beta <= 20 deg):
       q* = 0.047 exp(-(2.35 Rc/Hm0)^1.3)                                                      (eq. 7.1)
The formulae were themselves partly calibrated on CLASH data, so the comparison is not independent.

Decision analysis.  Allowable discharges are not universal (EurOtop 2018, Chapter 3: limits depend on use,
hazard type and wave height).  A sensitivity grid of prototype limits q_allow in {0.1, 1, 10, 50} l/s/m at
prototype wave heights Hm0,p in {1, 2, 3} m is converted to log10 q* thresholds by Froude similarity
(q*_allow = q_allow / sqrt(g Hm0,p^3)); every test is then judged against each threshold.
Outputs: revision/tables/E*.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from .pipeline import load_model_data, load_pred
from .utils import ROOT

TAB = ROOT / "revision" / "tables"
G = 9.81


def eurotop(df: pd.DataFrame):
    H, T = df.hm0_toe.to_numpy(float), df.tm10_toe.to_numpy(float)
    L0 = G * T ** 2 / (2 * np.pi)
    s0 = H / L0
    rc, ac, gc, h = df.rc.to_numpy(float), df.ac.to_numpy(float), df.gc.to_numpy(float), df.h.to_numpy(float)
    cd, cu = df.cotad.to_numpy(float), df.cotau.to_numpy(float)
    gf = df.gf_raw.to_numpy(float)
    beta = np.abs(np.nan_to_num(df.beta_deg.to_numpy(float)))
    gb_ = 1 - 0.0033 * np.minimum(beta, 80)
    rch = rc / H
    no_berm = ~(df.b_berm.to_numpy(float) > 0)
    single = np.abs(cd - cu) < 0.05
    slope_ok = (cd >= 1) & (cd <= 7)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan = 1 / cd
        xi = tan / np.sqrt(s0)
    dom_s = no_berm & single & slope_ok & (rc > 0) & (rch > 0.1) & (rch < 3.5) & (xi < 5) & (beta <= 80) & (s0 >= 0.001) & (s0 <= 0.07) \
        & np.isfinite(gf) & (ac >= rc - 1e-9)
    gfs = np.where((gf < 0.9) & (xi > 1.8), gf + (xi - 1.8) * (1 - gf) / 8.2, gf)
    gfs = np.minimum(gfs, 1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        q1 = 0.023 / np.sqrt(tan) * xi * np.exp(-(2.7 * rch / (xi * gfs * gb_)) ** 1.3)
        q2 = 0.09 * np.exp(-(1.5 * rch / (gfs * gb_)) ** 1.3)
        cr = np.where(gf < 0.9, np.minimum(1.0, 3.06 * np.exp(-1.5 * np.nan_to_num(gc) / H)), 1.0)
        qs = np.minimum(q1, q2) * cr
    vert = (cd == 0) & (cu == 0)
    no_mound = (df.ht.to_numpy(float) >= 0.95 * h) & ~(df.bt.to_numpy(float) > 0)
    dom_v = vert & no_mound & (h / H >= 4) & (rch > 0.1) & (rch < 3.5) & (beta <= 20)
    qv = 0.047 * np.exp(-(2.35 * rch) ** 1.3)
    q = np.where(dom_s, qs, np.where(dom_v, qv, np.nan))
    domain = np.where(dom_s & (gf >= 0.9), "S smooth single slope", np.where(dom_s, "S rough single slope", np.where(dom_v, "V plain vertical wall", "outside")))
    return np.log10(q), domain


def thresholds():
    rows = []
    for qa in (0.1, 1.0, 10.0, 50.0):
        for hp in (1.0, 2.0, 3.0):
            rows.append({"q_allow_l_per_s_per_m": qa, "Hm0_prototype_m": hp, "log10_qstar_allow": np.log10(qa / 1000 / np.sqrt(G * hp ** 3))})
    return pd.DataFrame(rows)


def benchmark(D):
    df, y = D["df"], D["y"]
    le, dom = eurotop(df)
    preds = {"EurOtop 2018 (mean value)": le, "XGBoost baseline (original OOF)": load_pred("geometry__xgb")[0]}
    v2 = ROOT / "revision/outputs/models/geometry__xgb.npz"
    if v2.exists():
        preds["XGBoost v2 (optimised, seed mean)"] = np.load(v2)["pred"].mean(0)
    for m, lab in (("catboost", "CatBoost baseline"), ("rf", "Random Forest baseline")):
        preds[lab] = load_pred(f"geometry__{m}").mean(0)
    rows = []
    for d in ["all applicable"] + sorted(set(dom) - {"outside"}):
        mk = (dom != "outside") if d == "all applicable" else (dom == d)
        mk &= np.isfinite(le)
        for sub, sm in (("all RF/CF", np.ones(len(y), bool)), ("RF=4", df.RF.to_numpy() == 4), ("CF>=3", df.CF.to_numpy() >= 3)):
            mm = mk & sm
            if mm.sum() < 10:
                continue
            for lab, p in preds.items():
                e = p[mm] - y[mm]
                rows.append({"domain": d, "subset": sub, "model": lab, "n": int(mm.sum()), "n_series": df.series[mm].nunique(),
                             "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)), "mae": float(np.mean(np.abs(e))),
                             "pct_within_factor10": float(100 * np.mean(np.abs(e) <= 1))})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "E1_eurotop_benchmark.csv", index=False)
    pd.DataFrame({"domain": dom}).value_counts().rename("n").reset_index().to_csv(TAB / "E1b_eurotop_domain_counts.csv", index=False)
    return out, le, dom


def decisions(D):
    """Exceedance decisions under training-data policies (original v1 policy predictions and intervals)."""
    df, y = D["df"], D["y"]
    rf, cf = df.RF.to_numpy(), df.CF.to_numpy()
    Z = np.load(ROOT / "results/predictions/policy__xgb.npz")
    pols = ["P1_retain_all", "P2_excl_RF4", "P3_excl_CF4", "P4_excl_RF4_or_CF4", "P5_CLASH_weights", "P6a_inverse_variance_weights", "P6b_quality_aware_intervals"]
    subs = {"all": np.ones(len(y), bool), "common": (rf < 4) & (cf < 4), "RF4": rf == 4, "CF4": cf == 4}
    th = thresholds()
    rows = []
    base = Z["P1_retain_all__pred"]
    for _, t in th.iterrows():
        thr = t.log10_qstar_allow
        meas = y > thr
        for p in pols:
            mu, lo, hi = Z[f"{p}__pred"], Z[f"{p}__lo"], Z[f"{p}__hi"]
            sig = (hi - lo) / (2 * norm.ppf(0.95))
            pex = 1 - norm.cdf((thr - mu) / sig)
            pred_ex = mu > thr
            for s, mk in subs.items():
                ms = meas[mk]
                rows.append({**t.to_dict(), "policy": p, "subset": s, "n": int(mk.sum()), "pct_measured_exceed": 100 * ms.mean(),
                             "pct_false_safe": 100 * float(np.mean(ms & ~pred_ex[mk])), "pct_false_alarm": 100 * float(np.mean(~ms & pred_ex[mk])),
                             "false_safe_rate_among_exceed": 100 * float(np.mean(~pred_ex[mk][ms])) if ms.sum() else np.nan,
                             "pct_decision_changed_vs_P1": 100 * float(np.mean(pred_ex[mk] != (base[mk] > thr))),
                             "mean_P_exceed_measured_exceed": float(pex[mk][ms].mean()) if ms.sum() else np.nan,
                             "pct_upper_bound_below_threshold_but_exceeds": 100 * float(np.mean(ms & (hi[mk] < thr))),
                             "brier_exceedance": float(np.mean((pex[mk] - ms) ** 2))})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "E2_exceedance_decisions_by_policy.csv", index=False)
    # risk ranking and safety-class changes for category-4 tests
    edges = np.sort(th[th.Hm0_prototype_m == 2.0].log10_qstar_allow.to_numpy())
    rk = []
    for p in pols[1:]:
        for s in ("RF4", "CF4", "common"):
            mk = subs[s]
            c0, c1 = np.digitize(base[mk], edges), np.digitize(Z[f"{p}__pred"][mk], edges)
            rk.append({"policy": p, "subset": s, "spearman_with_P1": spearmanr(base[mk], Z[f"{p}__pred"][mk])[0],
                       "pct_safety_class_changed": 100 * float(np.mean(c0 != c1)),
                       "pct_class_lowered (less severe)": 100 * float(np.mean(c1 < c0)), "pct_class_raised": 100 * float(np.mean(c1 > c0))})
    pd.DataFrame(rk).to_csv(TAB / "E3_risk_ranking_and_safety_class.csv", index=False)
    return out


def regimes(D):
    """Do category-4 tests represent particular engineering regimes, and is coverage of those regimes lost?"""
    from sklearn.neighbors import NearestNeighbors
    df, X = D["df"], D["X"]
    rf, cf = df.RF.to_numpy(), df.CF.to_numpy()
    feats = {"depth-limited (Hm0,toe/h > 1)": df.flag_depth_limited_gt1.to_numpy(bool),
             "shallow (Hm0,toe/h > 0.6)": (df.hm0_toe / df.h > 0.6).to_numpy(),
             "vertical or steep wall family": df.family.isin(["vertical_wall", "slope_with_wall", "wall_with_slope_above"]).to_numpy(),
             "bermed slope": (df.family == "bermed_slope").to_numpy(),
             "oblique attack (beta > 30 deg)": (df.beta_deg.abs() > 30).to_numpy(),
             "low freeboard (Rc/Hm0 < 0.5)": (df.rc / df.hm0_toe < 0.5).to_numpy(),
             "steep foreshore (1/m >= 1/20)": (df.m.between(0.01, 20)).to_numpy(),
             "chute structures (remark)": df.flag_chute.to_numpy(bool)}
    rows = []
    for lab, f in feats.items():
        rows.append({"regime": lab, "n_tests": int(f.sum()), "pct_RF4_in_regime": 100 * float(np.mean(rf[f] == 4)) if f.sum() else np.nan,
                     "pct_CF4_in_regime": 100 * float(np.mean(cf[f] == 4)) if f.sum() else np.nan,
                     "pct_regime_lost_by_P4": 100 * float(np.mean((rf[f] == 4) | (cf[f] == 4))) if f.sum() else np.nan,
                     "pct_of_RF4_tests_in_regime": 100 * float(np.mean(f[rf == 4])), "pct_of_common_tests_in_regime": 100 * float(np.mean(f[(rf < 4) & (cf < 4)]))})
    R = pd.DataFrame(rows)
    R.to_csv(TAB / "E4_regimes_of_category4_tests.csv", index=False)
    # input-space coverage: nearest-neighbour distance (standardised predictors) to a retained test of ANOTHER series
    from .baseline_models import FoldPrep
    Z = FoldPrep(X).scale(X)
    ser = df.series.to_numpy()
    keep = (rf < 4) & (cf < 4)
    rows = []
    for lab, pool in (("all tests", np.ones(len(df), bool)), ("tests kept by P4 (RF<4, CF<4)", keep)):
        nn = NearestNeighbors(n_neighbors=60).fit(Z[pool])
        dist, idx = nn.kneighbors(Z)
        pidx = np.flatnonzero(pool)
        d_other = np.array([next((dd for dd, ii in zip(dist[i], idx[i]) if ser[pidx[ii]] != ser[i]), np.nan) for i in range(len(df))])
        for s, mk in (("RF4", rf == 4), ("CF4", cf == 4), ("common", keep)):
            rows.append({"reference pool": lab, "subset": s, "median_nn_distance_other_series": float(np.nanmedian(d_other[mk])),
                         "p90_nn_distance": float(np.nanpercentile(d_other[mk], 90))})
    C = pd.DataFrame(rows)
    C.to_csv(TAB / "E5_input_space_coverage.csv", index=False)
    return R, C


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    D = load_model_data()
    b, _, dom = benchmark(D)
    print(pd.Series(dom).value_counts()); print(b[b.subset == "all RF/CF"].to_string())
    dd = decisions(D)
    print(dd[(dd.Hm0_prototype_m == 2.0) & (dd.subset.isin(["RF4", "CF4"]))][["q_allow_l_per_s_per_m", "policy", "subset", "pct_false_safe", "pct_decision_changed_vs_P1"]].to_string())
    R, C = regimes(D)
    print(R.to_string()); print(C.to_string())


if __name__ == "__main__":
    main()
