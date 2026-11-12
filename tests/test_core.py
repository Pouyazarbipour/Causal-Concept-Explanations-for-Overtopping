"""Unit tests: target calculation, transformations, grouping, leakage, counterfactual feasibility, statistics."""
import numpy as np
import pandas as pd
import pytest

from src.concept_ontology import CONCEPTS, concept_labels
from src.counterfactual_audit import build_views, curve_metrics, features_from_raw, support_flags
from src.data_preprocessing import dimensionless_discharge, response_class, signed_log1p
from src.error_decomposition import reml_em
from src.evaluation import regression_metrics
from src.geometry_grouping import add_groups, check_no_leakage, make_folds
from src.statistical_analysis import cluster_bootstrap, holm, weighted_kappa, weighted_quantile
from src.uncertainty_modeling import group_quantile

FEATS = ["rc_h", "ac_h", "gc_l", "s0", "h_l", "ht_h", "bt_l", "b_l", "hb_h", "tanab", "cotad", "cotau", "gf", "beta", "fs"]
LOG1P = ["rc_h", "ac_h", "gc_l", "h_l", "ht_h", "bt_l", "b_l", "hb_h"]


def toy(n=40, seed=0):
    r = np.random.RandomState(seed)
    return pd.DataFrame({"name": [f"{i // 10 + 1:03d}-{i:03d}" for i in range(n)], "series": [f"{i // 10 + 1:03d}" for i in range(n)],
                         "hm0_toe": r.uniform(0.05, 0.2, n), "tm10_toe": r.uniform(1, 2, n), "h": r.uniform(0.3, 0.6, n),
                         "ht": r.uniform(0.2, 0.3, n), "bt": 0.0, "gf_raw": r.choice([1.0, 0.5], n), "cotad": 2.0, "cotau": 2.0,
                         "rc": r.uniform(0.05, 0.3, n), "b_berm": 0.0, "hb": 0.0, "tanab": 0.0, "ac": r.uniform(0.05, 0.3, n),
                         "gc": r.choice([0.0, 0.1], n), "m": 1000.0, "beta_deg": 0.0, "reference": "", "s0": 0.03})


def test_dimensionless_discharge_and_units():
    # q = 1e-3 m3/s/m, Hm0 = 1 m -> q* = 1e-3 / sqrt(9.81)
    assert dimensionless_discharge(1e-3, 1.0) == pytest.approx(1e-3 / np.sqrt(9.81))
    # Froude similarity: scaling lengths by s scales q by s^1.5 and leaves q* unchanged
    assert dimensionless_discharge(1e-3 * 4 ** 1.5, 4.0) == pytest.approx(dimensionless_discharge(1e-3, 1.0))


def test_response_classes_no_epsilon():
    q = pd.Series([1e-4, 0.0, -1e-5, np.nan])
    assert list(response_class(q, q)) == ["positive", "zero", "negative", "missing"]


def test_signed_log1p_monotone_and_odd():
    x = np.linspace(-3, 3, 13)
    z = signed_log1p(x)
    assert np.all(np.diff(z) > 0) and np.allclose(z, -z[::-1])


def test_geometry_key_is_water_level_invariant():
    df = toy(2)
    df.loc[1] = df.loc[0]
    df.loc[1, "name"] = "001-999"
    for c in ("h", "ht", "hb"):
        df.loc[1, c] = df.loc[0, c] + 0.05      # water level +5 cm
    for c in ("rc", "ac"):
        df.loc[1, c] = df.loc[0, c] - 0.05
    g = add_groups(df)
    assert g["geometry"].nunique() == 1
    df.loc[1, "rc"] += 0.03                      # a higher wall is another geometry, same sub-structure
    g = add_groups(df)
    assert g["geometry"].nunique() == 2 and g["substructure"].nunique() == 1


def test_folds_have_no_group_leakage():
    df = add_groups(toy(200))
    f = make_folds(df, 4, 1)
    for scheme, col in (("geometry", "geometry"), ("series", "series")):
        for k in range(4):
            check_no_leakage(df.loc[f[scheme] != k, col], df.loc[f[scheme] == k, col])
    with pytest.raises(AssertionError):
        check_no_leakage(["a", "b"], ["b"])


def test_counterfactual_views_touch_only_intended_features():
    df = toy()
    X, vt, feas = build_views(df, FEATS, LOG1P, [0.1, 0.5])
    base = X[:, 0]
    i = int(np.flatnonzero((vt.intervention == "crest_up") & (vt.delta == 0.5))[0])
    changed = np.flatnonzero(~np.isclose(X[:, i], base).all(0))
    assert set(changed) == {FEATS.index("rc_h"), FEATS.index("ac_h")}
    assert np.allclose(np.expm1(X[:, i, 0]) - np.expm1(base[:, 0]), 0.5)   # exactly +0.5 Hm0
    j = int(np.flatnonzero((vt.intervention == "wall_up") & (vt.delta == 0.1))[0])
    assert (feas[:, j] == (df.rc >= df.ac - 1e-9)).all()
    k = int(np.flatnonzero((vt.intervention == "crest_wider") & (vt.delta == 0.1))[0])
    assert (feas[:, k] == (df.gc > 0)).all()
    r = int(np.flatnonzero((vt.intervention == "rougher") & (vt.delta == 0.5))[0])
    assert not feas[df.gf_raw == 0.5, r].any()   # 0.5 - 0.20 < 0.35 is outside the documented gamma_f range


def test_support_flags_reject_extrapolation():
    r = np.random.RandomState(0)
    Xtr = r.normal(size=(500, 3))
    Xv = np.stack([Xtr[:50], Xtr[:50] + 25.0], axis=1)
    s = support_flags(Xtr, Xv)
    assert s[:, 0].mean() > 0.9 and not s[:, 1].any()


def test_curve_metrics_counts_steps_and_curves_separately():
    vt = pd.DataFrame({"intervention": ["base", "a", "a", "a"], "delta": [0, 0.05, 0.5, 1.0]})
    P = np.array([[0.0, -0.1, -0.2, -0.3], [0.0, 0.1, -0.2, -0.1], [0.0, 0.001, 0.0, -0.5]])
    m = curve_metrics(P, vt, np.ones_like(P, bool), "a", tol=0.02, local_delta=0.05)
    assert list(m.n_steps_up) == [0, 2, 0] and list(m.curve_affected) == [False, True, False]
    assert list(m.local_reversal) == [False, True, False] and list(m.global_reversal) == [False, False, False]
    assert m.max_step_rise[1] == pytest.approx(0.1)


def test_concept_labels_cover_states_and_use_no_response():
    df = toy()
    lab = concept_labels(df)
    assert list(lab.columns) == [c.name for c in CONCEPTS]
    assert "q" not in df.columns and (lab.values >= 0).all()
    assert set(lab["roughness_permeability"]) == {0, 2}


def test_metrics_and_statistics():
    y = np.array([0.0, 1.0, 2.0])
    m = regression_metrics(y, y + 0.3)
    assert m["bias"] == pytest.approx(0.3) and m["f10"] == pytest.approx(10 ** 0.3) and m["fac2"] == 1.0 and m["rmse"] == pytest.approx(0.3)
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert weighted_kappa([1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert weighted_quantile([1, 2, 3, 4], [1, 1, 1, 1], 0.5) == 2
    pt, lo, hi, _ = cluster_bootstrap(lambda i: [np.mean(np.arange(100.0)[i])], np.repeat(np.arange(10), 10), 200, 0)
    assert lo[0] < pt[0] < hi[0]


def test_group_quantile_is_group_weighted_and_conservative():
    scores = np.concatenate([np.full(1000, 10.0), np.arange(9.0)])   # one huge group + 9 singletons
    groups = np.concatenate([np.zeros(1000), np.arange(1, 10)])
    assert group_quantile(scores, groups, 0.5) < 10.0                # the big group does not dominate
    assert np.isinf(group_quantile(scores, groups, 0.99))            # too few groups for 99 %: infinite, not optimistic


def test_reml_recovers_variance_components():
    r = np.random.RandomState(0)
    g = np.repeat(np.arange(150), 8)
    y = r.normal(0, 0.5, 150)[g] + r.normal(0, 0.3, len(g))
    f = reml_em(y, [g.astype(str)])
    assert f["var"][0] == pytest.approx(0.25, rel=0.3) and f["resid_var"] == pytest.approx(0.09, rel=0.15)
