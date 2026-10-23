"""
Kaufman Ch 2 toolkit (p.56-58): information ratio, beta / alpha / Treynor,
leveraged and inverse daily-reset funds, and the synthetic dollar index.
Each is checked on a case whose answer is known by hand.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from saty_bt.indicators.dollar_index import (COMPONENTS, DXY_WEIGHTS, PAIRS, dollar_index,
                                             index_weights)
from saty_bt.leverage import (daily_reset_index, max_drawdown, static_leverage,
                              volatility_decay)
from saty_bt.stats import annualised_return, beta_alpha, information_ratio
from saty_bt.validate import assert_causal


# ── information ratio ────────────────────────────────────────────────

def test_ir_constant_return_is_infinite_and_sign_correct():
    assert information_ratio(np.full(300, 0.001), 252) == math.inf
    assert information_ratio(np.full(300, -0.001), 252) == -math.inf
    assert math.isnan(information_ratio(np.zeros(300), 252))


def test_ir_by_hand_on_alternating_returns():
    r = np.tile([0.02, -0.01], 126)                         # 252 periods = one year
    ann_geo = (1.02 * 0.99) ** 126 - 1
    vol = np.std(r, ddof=1) * math.sqrt(252)
    assert information_ratio(r, 252) == pytest.approx(ann_geo / vol, rel=1e-12)
    assert information_ratio(r, 252, geometric=False) == pytest.approx(0.005 * 252 / vol, rel=1e-12)
    assert annualised_return(r, 252) == pytest.approx(ann_geo, rel=1e-12)


# ── beta, alpha, Treynor ─────────────────────────────────────────────

def test_beta_is_two_for_a_2x_series():
    rng = np.random.default_rng(7)
    b = rng.normal(0.0005, 0.01, 2000)
    r = beta_alpha(2 * b, b, 252)
    assert r["beta"] == pytest.approx(2.0, abs=1e-12)
    assert r["alpha_ann"] == pytest.approx(0.0, abs=1e-12)
    assert r["correlation"] == pytest.approx(1.0, abs=1e-12)
    assert r["beta_share"] == pytest.approx(1.0, abs=1e-12)     # all of it is the market


def test_alpha_is_the_constant_added_and_beta_survives_noise():
    rng = np.random.default_rng(8)
    b = rng.normal(0.0004, 0.012, 5000)
    s = 0.0002 + 0.5 * b                                        # 0.02%/day of pure alpha
    r = beta_alpha(s, b, 252)
    assert r["beta"] == pytest.approx(0.5, abs=1e-12)
    assert r["alpha_ann"] == pytest.approx(0.0002 * 252, rel=1e-9)
    noisy = beta_alpha(s + rng.normal(0, 0.01, 5000), b, 252)
    assert noisy["beta"] == pytest.approx(0.5, abs=0.05)
    assert 0 < noisy["r_squared"] < 1


def test_treynor_arithmetic():
    b = np.tile([0.01, -0.005, 0.002, 0.0], 63)                # 252 periods
    s = 1.5 * b + 0.0001
    r = beta_alpha(s, b, 252, rf=0.02)
    ann = float(np.prod(1 + s) - 1)                            # exactly one year
    assert r["beta"] == pytest.approx(1.5, abs=1e-12)
    assert r["treynor"] == pytest.approx((ann - 0.02) / 1.5, rel=1e-12)
    ra = beta_alpha(s, b, 252, rf=0.02, geometric=False)
    assert ra["treynor"] == pytest.approx((s.mean() * 252 - 0.02) / 1.5, rel=1e-12)
    # Jensen with rf: the intercept of excess returns; s - rf = 1.5 (b - rf) + 0.0001 + 0.5 rf
    assert r["alpha_ann"] == pytest.approx((0.0001 + 0.5 * 0.02 / 252) * 252, rel=1e-9)


def test_beta_alpha_aligns_series_on_their_index():
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    b = pd.Series(np.random.default_rng(9).normal(0, 0.01, 100), index=idx)
    s = (3 * b).iloc[10:]                                      # shorter, still aligned
    r = beta_alpha(s, b, 365)
    assert r["n"] == 90 and r["beta"] == pytest.approx(3.0, abs=1e-12)


# ── leveraged and inverse daily-reset funds (p.56-57) ────────────────

def _hand_loop(c, L, short=False, kind="kaufman"):
    out = [100.0]
    for i in range(1, len(c)):
        if not short:
            mid = c[i] / c[i - 1]
        elif kind == "kaufman":
            mid = c[i - 1] / c[i]
        else:
            mid = 2 - c[i] / c[i - 1]
        out.append(max(out[-1] * ((mid - 1) * L + 1), 0.0))
    return np.array(out)


def test_daily_reset_matches_hand_loop():
    c = np.array([100, 103, 99, 104.5, 101, 101, 107.2, 95.0, 96.1])
    for L in (1, 2, 3, 5):
        for short in (False, True):
            for kind in ("kaufman", "etf"):
                got = daily_reset_index(c, L, short=short, inverse=kind).to_numpy()
                assert np.allclose(got, _hand_loop(c, L, short, kind), rtol=0, atol=1e-10)


def test_1x_is_the_market_and_volatility_decay_on_up_down_up():
    c = np.array([100.0, 110.0, 99.0, 108.9])                  # +10%, -10%, +10%
    assert np.allclose(daily_reset_index(c, 1).to_numpy(), c)
    two = daily_reset_index(c, 2).to_numpy()
    assert two[-1] == pytest.approx(100 * 1.2 * 0.8 * 1.2)     # 115.2
    stat = static_leverage(c, 2).to_numpy()
    assert stat[-1] == pytest.approx(100 * (1 + 2 * 0.089))    # 117.8 = 2 x the market's +8.9%
    assert two[-1] < stat[-1]                                  # the decay
    # a round trip: the market is flat, the 2x fund is not
    flat = daily_reset_index([100.0, 110.0, 100.0], 2).to_numpy()
    assert flat[-1] == pytest.approx(100 * 1.2 * (1 - 2 / 11))  # 98.18
    # decay estimate: (L^2 - L) var / 2 per period, zero at 1x
    r = np.diff(c) / c[:-1]
    assert volatility_decay(r, 1, 252) == 0.0
    assert volatility_decay(r, 3, 252) == pytest.approx(3 * np.var(r, ddof=1) * 252)


def test_inverse_fund_identity_p57():
    c = np.array([100.0, 104.0, 97.0, 120.0, 88.0, 91.0])
    # inverting the middle term makes the inverse fund the long fund on 1/C ...
    for L in (1, 2, 3):
        assert np.allclose(daily_reset_index(c, L, short=True).to_numpy(),
                           daily_reset_index(1 / c, L).to_numpy())
    # ... so at 1x it is exactly the reciprocal of the price: XC_i = 100 C_0 / C_i
    assert np.allclose(daily_reset_index(c, 1, short=True).to_numpy(), 100 * c[0] / c)
    # the ETF convention (-L x the return) decays even at 1x on a round trip
    rt = [100.0, 110.0, 100.0]
    assert daily_reset_index(rt, 1, short=True).iloc[-1] == pytest.approx(100.0)
    assert daily_reset_index(rt, 1, short=True, inverse="etf").iloc[-1] == pytest.approx(100 * 0.9 * (1 + 1 / 11))
    assert daily_reset_index(rt, 1, short=True, inverse="etf").iloc[-1] < 100.0


def test_floor_at_zero_is_absorbing():
    c = [100.0, 105.0, 83.0, 120.0, 130.0]                    # a -20.95% day
    five = daily_reset_index(c, 5).to_numpy()
    assert five[2] == 0.0 and (five[2:] == 0).all()
    st = static_leverage([100.0, 85.0, 79.0, 150.0], 5).to_numpy()   # -21% from the start
    assert st[2] == 0.0 and st[3] == 0.0                       # liquidated, no comeback
    assert max_drawdown(five) == -1.0


def test_daily_reset_rejects_gaps_and_bad_leverage():
    with pytest.raises(ValueError):
        daily_reset_index([100.0, np.nan, 101.0], 2)
    with pytest.raises(ValueError):
        daily_reset_index([100.0, 101.0], -2)


# ── synthetic dollar index ───────────────────────────────────────────

def test_weights_drop_sek_and_sum_to_one():
    w = index_weights()
    assert "SEK" not in w and set(w) == set(COMPONENTS)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-15)
    assert w["EUR"] == pytest.approx(0.576 / (1 - DXY_WEIGHTS["SEK"]))


def _closes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    base = {"EURUSD": 1.10, "GBPUSD": 1.30, "USDJPY": 140.0, "USDCAD": 1.35, "USDCHF": 0.90}
    idx = pd.date_range("2023-01-02", periods=n, freq="h", tz="UTC")
    return pd.DataFrame({p: v * np.exp(np.cumsum(rng.normal(0, 0.001, n))) for p, v in base.items()}, index=idx)


def test_dollar_index_rises_when_every_usd_leg_rises():
    c = _closes()
    inverted = dict(COMPONENTS.values())                       # pair -> quoted USD per currency?
    up = c.copy()
    for p in PAIRS:                                            # the dollar buys 1% more of everything
        up[p] = c[p] / 1.01 if inverted[p] else c[p] * 1.01
    a, b = dollar_index(c), dollar_index(up)
    # every pair moved 1% in the dollar's favour: log index up exactly log(1.01)
    assert np.allclose(b.log_index - a.log_index, math.log(1.01))
    # one leg alone moves it by that leg's weight
    jpy = c.copy()
    jpy["USDJPY"] = c["USDJPY"] * 1.01
    assert np.allclose(dollar_index(jpy).log_index - a.log_index, index_weights()["JPY"] * math.log(1.01))
    eur = c.copy()
    eur["EURUSD"] = c["EURUSD"] * 1.01                         # euro up = dollar down
    assert (dollar_index(eur).log_index < a.log_index).all()


def test_dollar_index_nan_where_a_component_is_missing_and_rebased():
    c = _closes()
    c.iloc[:5, c.columns.get_loc("GBPUSD")] = np.nan
    c.iloc[100, c.columns.get_loc("USDCHF")] = np.nan
    d = dollar_index(c)
    assert d.log_index.iloc[:5].isna().all() and np.isnan(d.log_index.iloc[100])
    assert d.dxy.iloc[5] == pytest.approx(100.0)               # anchored at the first complete row
    assert d.dxy.notna().sum() == len(c) - 6


def test_dollar_index_is_causal():
    c = _closes(n=600, seed=3)
    assert_causal(lambda b: dollar_index(b), c, ["log_index", "dxy"], name="dollar_index")
