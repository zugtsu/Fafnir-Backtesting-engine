"""Kaufman's probability-of-target, the path version, and the shape check."""
import math

import numpy as np
import pytest

from saty_bt.stats import path_odds, prob_reach, shape


def test_prob_reach_closed_form():
    from statistics import NormalDist
    rng = np.random.default_rng(1)
    lr = rng.normal(0.01, 0.05, 20_000)
    r = np.expm1(lr)
    mu, sd = lr.mean(), lr.std(ddof=1)
    for target in (math.e, math.exp(1.5), 0.8):
        z = (math.log(target) - 100 * mu) / (sd * 10)
        assert prob_reach(r, 1000, 1000 * target, 100) == pytest.approx(1 - NormalDist().cdf(z), abs=1e-12)
    # and the sense of it: at the median growth the odds are a coin flip
    assert prob_reach(r, 1000, 1000 * math.exp(100 * mu), 100) == pytest.approx(0.5, abs=1e-9)


def test_path_odds_extremes():
    up = path_odds([0.01] * 50, 1000, 2000, 500, n_periods=100, reps=200)
    assert up["p_target_first"] == 1.0 and up["p_floor_first"] == 0.0
    down = path_odds([-0.01] * 50, 1000, 2000, 500, n_periods=100, reps=200)
    assert down["p_floor_first"] == 1.0 and down["p_end_above"] == 0.0
    wipe = path_odds([-1.2, 0.5, 0.5], 1000, 2000, 500, n_periods=30, reps=200, mean_block=1.0)
    assert wipe["median_end"] == 0.0                       # a <= -100% trade ends the path


def test_shape_reports_plain_kurtosis():
    rng = np.random.default_rng(0)
    s = shape(rng.normal(size=50_000))
    assert s["kurtosis"] == pytest.approx(3.0, abs=0.1) and abs(s["skew"]) < 0.05
    big_winners = np.r_[np.full(95, -0.01), np.full(5, 0.30)]
    assert shape(big_winners)["skew"] > 2 and shape(big_winners)["kurtosis"] > 7
