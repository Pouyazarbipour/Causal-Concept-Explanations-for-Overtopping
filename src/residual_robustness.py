"""Phase VI-VII (revision): robustness of the RF/CF validity findings and the repeatability comparison.

Adds to the original analyses (experiments/run_analysis.py, unchanged):
  R1  variance ratios (Gamma GLM, series-cluster bootstrap) for every learner and validation scheme
  R2  specificity of the RF = 4 effect: RF4/RF3 and RF4/RF2 contrasts; RF 1-3 trend without RF = 4
  R3  repeatability: tolerance sensitivity, disjoint (one pair per record) estimator, CI of the
      replicate / within-geometry variance ratio
  R4  leave-one-series-out for CF = 4 and CF = 3
  R5  minimum detectable variance ratio for each CF/RF contrast (power)
  R6  why CF varies within reconstructed geometries
  R7  RF = 4 records with unusable inputs: representativeness and a deep-water substitution sensitivity
Outputs: revision/tables/R*.csv
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .error_decomposition import replicate_pairs
from .pipeline import groups_for, have, load_model_data, load_pred
from .rf_cf_validation import variance_model
from .statistical_analysis import cluster_bootstrap
from .utils import ROOT

TAB = ROOT / "revision" / "tables"
NB = 500
SEED = 20261112


def _pred(tag):
    p = ROOT / "revision" / "outputs" / "models" / f"{tag}.npz"
    return np.load(p)["pred"] if p.exists() else None


def r1_variance_ratios(D):
    df, y = D["df"], D["y"]
    rows = []
    srcs = [(f"geometry__{m}", f"{m}, geometry-grouped (original)") for m in ("xgb", "rf", "catboost", "mlp", "ccw_full")]
    srcs += [("random__xgb", "xgb, random records (original)"), ("series__xgb", "xgb, series held out (original)")]
    for tag, lab in srcs:
        if not have(tag):
            continue
        r = y - load_pred(tag).mean(0)
        t = variance_model(df, r, "series", NB, SEED, interaction=False)
        rows.append(t.assign(residuals=lab))
    for tag, lab in (("geometry__xgb", "xgb v2 (optimised), geometry-grouped"), ("series__xgb", "xgb v2 (optimised), series held out"),
                     ("geometry__catboost", "catboost v2, geometry-grouped"), ("geometry__lgbm", "lightgbm v2, geometry-grouped")):
        P = _pred(tag)
        if P is not None:
            rows.append(variance_model(df, y - P.mean(0), "series", NB, SEED, interaction=False).assign(residuals=lab))
    out = pd.concat(rows)
    out.to_csv(TAB / "R1_variance_ratios_by_learner_and_scheme.csv", index=False)
    return out


def r2_specificity(D):
    import statsmodels.api as sm
    df, y = D["df"], D["y"]
    r = y - load_pred("geometry__xgb")[0]
    rf, cf = df["RF"].to_numpy(int), df["CF"].to_numpy(int)
    X = np.column_stack([np.ones(len(df))] + [(rf == v).astype(float) for v in (2, 3, 4)] + [(cf == v).astype(float) for v in (2, 3, 4)])
    r2 = r ** 2 + 1e-8

    def st(idx):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = sm.GLM(r2[idx], X[idx], family=sm.families.Gamma(sm.families.links.Log())).fit(maxiter=100).params
        return [b[3] - b[2], b[3] - b[1], b[2] - b[1], b[2], b[1]]
    pt, lo, hi, reps = cluster_bootstrap(st, df["series"].to_numpy(), NB, SEED)
    names = ["RF4 / RF3", "RF4 / RF2", "RF3 / RF2", "RF3 / RF1", "RF2 / RF1"]
    rows = [{"contrast": n, "variance_ratio": np.exp(pt[k]), "lo": np.exp(lo[k]), "hi": np.exp(hi[k]),
             "boot_p": max(2 * min((reps[:, k] <= 0).mean(), (reps[:, k] >= 0).mean()), 1 / NB)} for k, n in enumerate(names)]
    # rank trend among RF 1-3 only
    from scipy.stats import spearmanr
    m = rf < 4
    a, f_ = np.abs(r[m]), rf[m]
    p2, l2, h2, rep2 = cluster_bootstrap(lambda i: [spearmanr(f_[i], a[i])[0]], df["series"].to_numpy()[m], NB, SEED)
    rows.append({"contrast": "Spearman(|r|, RF) among RF 1-3", "variance_ratio": p2[0], "lo": l2[0], "hi": h2[0],
                 "boot_p": max(2 * min((rep2[:, 0] <= 0).mean(), (rep2[:, 0] >= 0).mean()), 1 / NB)})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "R2_rf4_specificity_contrasts.csv", index=False)
    return out


def _disjoint(p):
    """Greedy one-pair-per-record subset (independent pairs)."""
    used, keep = set(), []
    for k, (i, j) in enumerate(zip(p.i, p.j)):
        if i not in used and j not in used:
            keep.append(k)
            used.update((i, j))
    return p.iloc[keep]


def r3_repeatability(D):
    df = D["df"]
    rows = []
    within = None
    r = D["y"] - load_pred("geometry__xgb")[0]
    t = df.assign(r=r)
    # pooled within-geometry variance (method of moments; approximates the REML within component)
    g = t.groupby("geometry")["r"]
    dfree = (g.size() - 1).clip(lower=0)
    within = float((g.var(ddof=1).fillna(0) * dfree).sum() / dfree.sum())
    for tol in (0.005, 0.01, 0.02, 0.05):
        p = replicate_pairs(df, tol)
        for sub_lab, sub in (("all pairs", p), ("disjoint pairs", _disjoint(p))):
            for key in ["all"] + [f"max RF={v}" for v in (1, 2, 3, 4)]:
                s = sub if key == "all" else sub[sub.rf_max == int(key[-1])]
                row = {"rel_tol": tol, "pairs": sub_lab, "subset": key, "n_pairs": len(s), "n_records": len(set(s.i) | set(s.j)) if len(s) else 0,
                       "n_series": s.series.nunique() if len(s) else 0}
                if len(s) >= 20 and s.series.nunique() >= 3:
                    d = s.dy.to_numpy()
                    pt, lo, hi, _ = cluster_bootstrap(lambda i: [np.sqrt(np.mean(d[i] ** 2) / 2)], s.series.to_numpy(), NB, SEED)
                    row.update({"repeat_sd": pt[0], "lo": lo[0], "hi": hi[0]})
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "R3_repeatability_tolerance_sensitivity.csv", index=False)
    # ratio replicate variance / within-geometry residual variance, joint series bootstrap
    p = replicate_pairs(df, 0.01)
    ser = df["series"].to_numpy()
    pairs_by_series = {s: p.dy.to_numpy()[p.series.to_numpy() == s] for s in np.unique(ser)}
    geo_stats = t.groupby(["series", "geometry"])["r"].agg(["size", "var"]).fillna(0).reset_index()
    gs = {s: geo_stats[geo_stats.series == s] for s in np.unique(ser)}
    uniq = np.unique(ser)
    rng = np.random.RandomState(SEED)

    def ratio(sel):
        d = np.concatenate([pairs_by_series[s] for s in sel])
        G = pd.concat([gs[s] for s in sel])
        fr = (G["size"] - 1).clip(lower=0)
        w = float((G["var"] * fr).sum() / fr.sum())
        return np.mean(d ** 2) / 2 / w, w
    pt, w0 = ratio(uniq)
    reps = [ratio(uniq[rng.randint(0, len(uniq), len(uniq))])[0] for _ in range(NB)]
    rat = pd.DataFrame([{"quantity": "replicate variance / within-geometry variance (pooled within-geometry estimator)", "value": pt,
                         "lo": np.percentile(reps, 2.5), "hi": np.percentile(reps, 97.5)},
                        {"quantity": "pooled within-geometry residual variance (method of moments)", "value": w0},
                        {"quantity": "within-geometry residual variance (REML, original Table 8)", "value": 0.1384}])
    rat.to_csv(TAB / "R3b_replicate_ratio_ci.csv", index=False)
    return out, rat


def r4_leave_one_series_out(D):
    df, y = D["df"], D["y"]
    r = y - load_pred("geometry__xgb")[0]
    rows = []
    for cat in (4, 3):
        sers = sorted(df.loc[df.CF == cat, "series"].unique())
        for s in [None] + sers:
            keep = np.ones(len(df), bool) if s is None else (df["series"] != s).to_numpy()
            d, rr = df[keep].reset_index(drop=True), r[keep]
            vm = variance_model(d, rr, "series", 200, SEED, interaction=False).set_index("term")
            m = (d.CF == cat).to_numpy()
            rows.append({"CF_category": cat, "left_out_series": s or "none", "n_left_out": int((~keep).sum()),
                         "n_CF_cat_remaining": int(m.sum()), "n_series_remaining": d.loc[m, "series"].nunique(),
                         "rmse_CF_cat": float(np.sqrt(np.mean(rr[m] ** 2))), "variance_ratio_vs_CF1": vm.loc[f"CF={cat}", "variance_ratio"],
                         "lo": vm.loc[f"CF={cat}", "ratio_lo"], "hi": vm.loc[f"CF={cat}", "ratio_hi"]})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "R4_leave_one_series_out_CF.csv", index=False)
    return out


def r5_power(R1):
    """Minimum detectable variance ratio (two-sided 5 %, 80 % power) = exp(2.80 * SE(log ratio)),
    SE approximated from the bootstrap interval width: (log hi - log lo) / (2 * 1.96)."""
    t = R1[R1.residuals == "xgb, geometry-grouped (original)"].copy()
    t = t[t.term != "const"]
    se = (np.log(t.ratio_hi) - np.log(t.ratio_lo)) / (2 * 1.96)
    t["se_log_ratio"] = se
    t["minimum_detectable_ratio_up"] = np.exp(2.80 * se)
    t["minimum_detectable_ratio_down"] = np.exp(-2.80 * se)
    t.to_csv(TAB / "R5_power_minimum_detectable_ratio.csv", index=False)
    return t


def r6_cf_inconsistency(D):
    m = D["master"]
    g = m.groupby("geometry")["CF"].nunique()
    geos = g[g > 1].index
    rows = []
    for geo in geos:
        s = m[m.geometry == geo]
        lev = sorted(s.CF.unique())
        lo, hi = s[s.CF == lev[0]], s[s.CF == lev[-1]]
        diff = {}
        for c in ("h", "rc", "hm0_toe", "tm10_toe", "beta_deg", "hm0_deep"):
            a, b = lo[c].median(), hi[c].median()
            diff[c] = (np.isfinite(a) and np.isfinite(b) and abs(a - b) > 0.01 * max(abs(a), abs(b), 1e-9))
        rem_lo, rem_hi = set(lo.remark.str.lower()), set(hi.remark.str.lower())
        # contiguous in test number?  (blocks of consecutive tests suggest a change of set-up or of rating practice)
        num = s.name.str.split("-").str[1].astype(int)
        blocks = int((s.sort_values("name").CF.diff().fillna(0) != 0).sum()) + 1
        cause = []
        if diff["rc"] or diff["h"]:
            cause.append("water level / freeboard differs")
        if diff["hm0_toe"] or diff["tm10_toe"]:
            cause.append("wave conditions differ")
        if diff["beta_deg"]:
            cause.append("obliquity differs")
        if rem_lo != rem_hi:
            cause.append("remarks differ (e.g. wind)")
        if lo.RF.median() != hi.RF.median():
            cause.append("RF also differs")
        rows.append({"geometry": geo, "series": s.series.iloc[0], "n": len(s), "CF_values": "/".join(str(int(v)) for v in lev),
                     "n_contiguous_blocks": blocks, "test_numbers": f"{num.min()}–{num.max()}", "in_regression": int(s.use_regression.sum()),
                     "differences_between_lowest_and_highest_CF": "; ".join(cause) if cause else "none in documented hydraulic variables"})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "R6_cf_inconsistent_geometries.csv", index=False)
    summ = out["differences_between_lowest_and_highest_CF"].str.split("; ").explode().value_counts().rename_axis("difference").reset_index(name="n_geometries")
    summ.to_csv(TAB / "R6b_cf_inconsistency_causes.csv", index=False)
    return out, summ


def r7_missing_input_sensitivity(D):
    """Deep-water substitution for RF = 4 positive-discharge records whose toe wave height or period is
    missing/zero.  The measured q is kept; only the scaling variables are substituted.  These records are
    then predicted by a model trained on complete records of OTHER series (fixed hyper-parameters), and the
    RF = 4 residual variance ratio is recomputed with them added.  Explicitly a sensitivity scenario."""
    import pickle
    import xgboost as xgb
    from .baseline_models import FoldPrep
    from .features import features_from_raw
    m = D["master"].copy()
    cfg = D["cfg"]["data"]
    cand = m[(m.response_class == "positive") & ~m.use_regression].copy()
    sub = cand.copy()
    bad_h = ~(sub.hm0_toe > 0)
    bad_t = ~(sub.tm10_toe > 0)
    sub.loc[bad_h, "hm0_toe"] = sub.loc[bad_h, "hm0_deep"]
    sub.loc[bad_t, "tm10_toe"] = sub.loc[bad_t, "tm10_deep"]
    ok = (sub.hm0_toe > 0) & (sub.tm10_toe > 0) & sub.h.notna() & sub.rc.notna()
    sub = sub[ok]
    sub["y"] = np.log10(sub.q / np.sqrt(9.81 * sub.hm0_toe ** 3))
    Xs = features_from_raw(sub, D["features"], cfg["log1p_features"], cfg["gravity"])
    df, X, y = D["df"], D["X"], D["y"]
    best = pickle.load(open(ROOT / "results/models/fold_cache/geometry__xgb__fold0.pkl", "rb"))["best"]
    pred = np.full(len(sub), np.nan)
    for s in sub.series.unique():
        tr = (df.series != s).to_numpy()
        prep = FoldPrep(X[tr])
        mdl = xgb.XGBRegressor(n_jobs=4, tree_method="hist", random_state=SEED, **best).fit(prep.impute(X[tr]), y[tr])
        mk = (sub.series == s).to_numpy()
        pred[mk] = mdl.predict(prep.impute(Xs[mk]))
    r_new = sub.y.to_numpy() - pred
    r0 = y - load_pred("geometry__xgb")[0]
    d_aug = pd.concat([df[["RF", "CF", "series"]], sub[["RF", "CF", "series"]]]).reset_index(drop=True)
    r_aug = np.concatenate([r0, r_new])
    base = variance_model(df, r0, "series", NB, SEED, interaction=False).set_index("term")
    aug = variance_model(d_aug, r_aug, "series", NB, SEED, interaction=False).set_index("term")
    out = pd.DataFrame([{"scenario": "complete cases (primary)", "n_RF4": int((df.RF == 4).sum()), **base.loc["RF=4"].to_dict()},
                        {"scenario": f"+ {len(sub)} RF=4 records with deep-water substitution of Hm0/Tm-1,0", "n_RF4": int((d_aug.RF == 4).sum()),
                         **aug.loc["RF=4"].to_dict()}])
    out["rmse_added_records"] = [np.nan, float(np.sqrt(np.mean(r_new ** 2)))]
    out["n_candidates"], out["n_substitutable"] = len(cand), len(sub)
    out.to_csv(TAB / "R7_missing_input_sensitivity.csv", index=False)
    return out


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    D = load_model_data()
    R1 = r1_variance_ratios(D)
    print(R1[R1.term == "RF=4"].to_string())
    print(r2_specificity(D).to_string())
    a, b = r3_repeatability(D)
    print(a[a.subset == "all"].to_string()); print(b.to_string())
    print(r4_leave_one_series_out(D).to_string())
    print(r5_power(R1).to_string())
    o, s = r6_cf_inconsistency(D)
    print(s.to_string())
    print(r7_missing_input_sensitivity(D).to_string())


if __name__ == "__main__":
    main()
