"""Unit tests of the revision modules (fast; no model fitting)."""
import numpy as np
import pandas as pd

from src.counterfactual_policies import eff_n, variance_models
from src.engineering_impact import eurotop, thresholds
from src.uncertainty_modeling import group_quantile


def test_variance_models_recover_main_effects():
    rng = np.random.default_rng(1)
    n = 20000
    rf, cf = rng.integers(1, 5, n), rng.integers(1, 5, n)
    sd = 0.4 * np.where(rf == 4, 1.5, 1.0)
    r = rng.normal(0, sd)
    _, main, shr, info = variance_models(r, rf, cf)
    ratio = main(np.array([4]), np.array([1]))[0] / main(np.array([1]), np.array([1]))[0]
    assert abs(ratio - 2.25) < 0.25
    assert info["tau2"] >= 0
    assert np.all(shr(rf, cf) > 0)


def test_effective_sample_size():
    assert eff_n(np.ones(10)) == 10
    assert eff_n(np.array([1.0, 0.0, 0.0])) == 1


def test_eurotop_vertical_wall_and_domain():
    df = pd.DataFrame({"hm0_toe": [1.0], "tm10_toe": [8.0], "rc": [1.5], "ac": [1.5], "gc": [0.0], "h": [6.0], "cotad": [0.0], "cotau": [0.0],
                       "gf_raw": [1.0], "beta_deg": [0.0], "b_berm": [0.0], "ht": [6.0], "bt": [0.0]})
    lq, dom = eurotop(df)
    assert dom[0] == "V plain vertical wall"
    assert np.isclose(lq[0], np.log10(0.047 * np.exp(-(2.35 * 1.5) ** 1.3)))


def test_eurotop_smooth_slope_bounded_by_maximum():
    df = pd.DataFrame({"hm0_toe": [1.0], "tm10_toe": [6.0], "rc": [1.0], "ac": [1.0], "gc": [0.0], "h": [5.0], "cotad": [3.0], "cotau": [3.0],
                       "gf_raw": [1.0], "beta_deg": [0.0], "b_berm": [0.0], "ht": [5.0], "bt": [0.0]})
    lq, dom = eurotop(df)
    assert dom[0] == "S smooth single slope"
    assert lq[0] <= np.log10(0.09 * np.exp(-(1.5 * 1.0) ** 1.3)) + 1e-12


def test_thresholds_froude_scaling():
    t = thresholds()
    r = t[(t.q_allow_l_per_s_per_m == 1.0) & (t.Hm0_prototype_m == 2.0)].iloc[0]
    assert np.isclose(r.log10_qstar_allow, np.log10(0.001 / np.sqrt(9.81 * 8)))


def test_group_quantile_finite_sample():
    s = np.arange(1, 11, dtype=float)
    g = np.arange(10)
    assert np.isinf(group_quantile(s, g, 0.95))          # 10 groups: level 0.95 needs the +inf mass
    assert group_quantile(s, g, 0.5) <= 10
