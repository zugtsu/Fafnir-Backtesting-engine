"""
Indicator tests against hand-computed values, plus the look-ahead audit:
shifting every input by one bar must shift every output by one bar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saty_bt.indicators.atr_levels import atr_levels, period_grid
from saty_bt.indicators.core import atr, ema, rma, swing_lows, true_range
from saty_bt.indicators.phase_oscillator import phase_oscillator
from saty_bt.indicators.pivot_ribbon import pivot_ribbon, time_warp
from saty_bt.indicators.volume_stack import volume_stack

ATR_CFG = {"atr_length": 14, "smoothing": "wilder",
           "fractions": [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.236, 1.618, 2.0, 2.618]}
RIB_CFG = {"emas": [8, 13, 21, 48, 200], "fast": 8, "pivot": 21, "slow": 48,
           "conviction_fast": 13, "conviction_slow": 48, "bias_long": 200, "flat_spread_atr": 0.15}
PO_CFG = {"pivot_ema": 21, "atr_length": 14, "atr_mult": 3.0, "smooth": 3, "bb_length": 21,
          "bb_std": 2.0, "keltner_mult": 2.0, "round_2dp": False, "compression_mode": "symmetric",
          "zones": {"extreme": 100.0, "distribution": 61.8, "neutral": 23.6}}
VS_CFG = {"rel_vol_length": 20, "conviction_mult": 1.5}


def bars(n=400, seed=1, start="2024-01-01", freq="30min", vol=True):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    opn = np.r_[100, close[:-1]]
    hi = np.maximum(opn, close) + rng.uniform(0, 0.4, n)
    lo = np.minimum(opn, close) - rng.uniform(0, 0.4, n)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    df = pd.DataFrame({"open": opn, "high": hi, "low": lo, "close": close,
                       "volume": rng.uniform(50, 150, n) if vol else np.nan}, index=idx)
    df.index.name = "timestamp"
    df["bar_close"] = df.index + pd.Timedelta(freq)
    df["day_key"] = df.index.normalize().tz_localize(None)
    return df


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def test_ema_matches_recursive_definition():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    e = ema(s, 3)                      # alpha = 0.5, seeded with first value
    assert e.tolist() == pytest.approx([1.0, 1.5, 2.25, 3.125, 4.0625])


def test_rma_sma_seed_then_wilder():
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    r = rma(s, 3)
    assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[1])
    assert r.iloc[2] == pytest.approx(20.0)                     # SMA seed
    assert r.iloc[3] == pytest.approx(20.0 + (40 - 20) / 3)      # 26.667
    assert r.iloc[4] == pytest.approx(r.iloc[3] + (50 - r.iloc[3]) / 3)


def test_true_range_and_atr_by_hand():
    df = pd.DataFrame({"open": [10, 11, 12], "high": [11, 13, 12.5], "low": [9, 10.5, 11],
                       "close": [10.5, 12.5, 11.5]})
    tr = true_range(df)
    assert tr.tolist() == pytest.approx([2.0, max(2.5, 2.5, 0.0), max(1.5, 0.0, 1.5)])
    a = atr(df, 2)
    assert a.iloc[1] == pytest.approx((2.0 + 2.5) / 2)
    assert a.iloc[2] == pytest.approx(a.iloc[1] + (1.5 - a.iloc[1]) / 2)


def test_swing_low_confirmed_late():
    low = [5, 4, 3, 4, 5, 6, 2, 3, 4]
    df = pd.DataFrame({"low": low, "high": [x + 1 for x in low]})
    sl = swing_lows(df, 2)
    # pivot at index 2 (3) confirmed at index 4; pivot at 6 (2) confirmed at 8
    assert np.isnan(sl.iloc[3])
    assert sl.iloc[4] == 3 and sl.iloc[7] == 3 and sl.iloc[8] == 2


# ---------------------------------------------------------------------------
# ATR levels
# ---------------------------------------------------------------------------

def test_period_grid_uses_previous_period_only():
    d = bars(40, freq="1D")
    d["period_key"] = d["day_key"]
    g = period_grid(d, ATR_CFG)
    k = 20
    assert g["pdc"].iloc[k] == d["close"].iloc[k - 1]
    a = atr(d, 14)
    assert g["atr"].iloc[k] == pytest.approx(a.iloc[k - 1])
    assert g["call_trigger"].iloc[k] == pytest.approx(g["pdc"].iloc[k] + 0.236 * g["atr"].iloc[k])
    assert g["put_trigger"].iloc[k] == pytest.approx(g["pdc"].iloc[k] - 0.236 * g["atr"].iloc[k])
    assert g["target_up"].iloc[k] == pytest.approx(g["pdc"].iloc[k] + 0.618 * g["atr"].iloc[k])
    assert g["full_down"].iloc[k] == pytest.approx(g["pdc"].iloc[k] - 1.0 * g["atr"].iloc[k])
    assert g["l2618_up"].iloc[k] == pytest.approx(g["pdc"].iloc[k] + 2.618 * g["atr"].iloc[k])


def test_levels_projected_and_range_util():
    chart = bars(48 * 30)                      # 30 days of 30m bars
    daily = chart.groupby("day_key").agg(open=("open", "first"), high=("high", "max"),
                                         low=("low", "min"), close=("close", "last"),
                                         volume=("volume", "sum"))
    daily.index = pd.DatetimeIndex(daily.index).tz_localize("UTC")
    daily["day_key"] = daily.index.tz_localize(None)
    daily["period_key"] = daily["day_key"]
    f = atr_levels(chart, daily, "1d", ATR_CFG)
    day = f["day_key"].iloc[-1]
    rows = f[f["day_key"] == day]
    prev_day_close = daily.loc[daily["day_key"] < day, "close"].iloc[-1]
    assert (rows["pdc"] == prev_day_close).all()
    # range util is monotone non-decreasing within the day and uses only bars so far
    ru = rows["range_util"].to_numpy()
    assert (np.diff(ru) >= -1e-12).all()
    first = rows.iloc[0]
    assert first["range_util"] == pytest.approx((first["high"] - first["low"]) / first["atr"])


# ---------------------------------------------------------------------------
# ribbon
# ---------------------------------------------------------------------------

def test_ribbon_states_are_exclusive_and_arrows_fire_on_change():
    f = pivot_ribbon(bars(600), RIB_CFG)
    assert ((f["ribbon_bull"].astype(int) + f["ribbon_bear"].astype(int) + f["ribbon_mixed"].astype(int)) == 1).all()
    ups = f["conviction_cross_up"].sum()
    assert 0 < ups < len(f) / 5
    # every cross-up bar has conviction_bull True and previous bar False
    idx = f.index[f["conviction_cross_up"]]
    for t in idx[1:]:
        i = f.index.get_loc(t)
        assert f["conviction_bull"].iloc[i] and not f["conviction_bull"].iloc[i - 1]


def test_time_warp_never_sees_unfinished_htf_bar():
    chart = bars(48 * 10)                                       # 30m, 10 days
    htf = chart.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                    "close": "last", "volume": "sum"})
    htf["bar_close"] = htf.index + pd.Timedelta("4h")
    out = time_warp(chart, htf, RIB_CFG)
    # a chart bar closing at 05:00 may only see the HTF bar 00:00-04:00
    t = out.index[out["bar_close"] == pd.Timestamp("2024-01-05 05:00", tz="UTC")][0]
    assert out.loc[t, "htf_bar_time"] == pd.Timestamp("2024-01-05 00:00", tz="UTC")
    # the chart bar closing exactly at 08:00 sees the 04:00-08:00 HTF bar
    t = out.index[out["bar_close"] == pd.Timestamp("2024-01-05 08:00", tz="UTC")][0]
    assert out.loc[t, "htf_bar_time"] == pd.Timestamp("2024-01-05 04:00", tz="UTC")


# ---------------------------------------------------------------------------
# oscillator
# ---------------------------------------------------------------------------

def test_phase_oscillator_formula():
    df = bars(300)
    f = phase_oscillator(df, PO_CFG)
    pivot = ema(df["close"], 21)
    a = atr(df, 14)
    raw = (df["close"] - pivot) / (3.0 * a) * 100
    expect = ema(raw, 3)
    pd.testing.assert_series_equal(f["osc"].dropna(), expect.dropna(), check_names=False)
    z = f["osc_zone"]
    assert set(z.unique()) <= {"extended_up", "distribution", "markup", "neutral", "markdown",
                               "accumulation", "extended_down", "na"}
    o = f["osc"]
    assert (z[(o > 23.6) & (o <= 61.8)] == "markup").all()
    assert (z[(o < -61.8) & (o >= -100)] == "accumulation").all()


def test_leaving_accumulation_is_a_cross():
    df = bars(300)
    f = phase_oscillator(df, PO_CFG)
    hits = f.index[f["leaving_accumulation"]]
    for t in hits:
        i = f.index.get_loc(t)
        assert f["osc"].iloc[i] > -61.8 and f["osc"].iloc[i - 1] <= -61.8


def test_compression_definition():
    df = bars(300)
    f = phase_oscillator(df, PO_CFG)
    pivot = ema(df["close"], 21)
    a = atr(df, 14)
    sd = df["close"].rolling(21).std(ddof=0)
    exp = ((pivot + 2 * a) < (pivot + 2 * sd)) & ((pivot - 2 * a) > (pivot - 2 * sd))
    assert (f["expansion"].to_numpy()[30:] == exp.to_numpy()[30:]).all()
    assert (f["compression"].to_numpy()[30:] == (~exp).to_numpy()[30:]).all()


# ---------------------------------------------------------------------------
# volume stack
# ---------------------------------------------------------------------------

def test_volume_stack_geometry():
    df = pd.DataFrame({"open": [10, 10, 10], "high": [12, 12, 10], "low": [8, 8, 10],
                       "close": [11, 9, 10], "volume": [100.0, 100.0, 100.0]})
    f = volume_stack(df, VS_CFG)
    assert f["buy_pct"].tolist() == pytest.approx([0.75, 0.25, 0.5])
    assert f["sell_pct"].tolist() == pytest.approx([0.25, 0.75, 0.5])
    assert f["buy_vol"].tolist() == pytest.approx([75, 25, 50])
    assert f["stack_winner"].tolist() == ["buy", "sell", "tie"]
    assert f["stack_streak"].tolist() == [1, -1, 0]


def test_volume_stack_without_volume():
    f = volume_stack(bars(50, vol=False), VS_CFG)
    assert f["rel_vol"].isna().all()
    assert not f["vol_available"].any()
    assert set(f["stack_winner"]) <= {"buy", "sell", "tie"}


# ---------------------------------------------------------------------------
# look-ahead audit: shift inputs by one bar -> outputs shift by one bar
# ---------------------------------------------------------------------------

def _shift_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Same bars, but delivered one bar later (the index moves, values don't)."""
    s = df.copy()
    s.index = df.index + (df.index[1] - df.index[0])
    s["bar_close"] = s.index + (df.index[1] - df.index[0])
    return s


@pytest.mark.parametrize("fn,cfg", [
    (pivot_ribbon, RIB_CFG), (phase_oscillator, PO_CFG), (volume_stack, VS_CFG)])
def test_indicators_are_causal(fn, cfg):
    df = bars(400)
    a = fn(df, cfg)
    b = fn(df.iloc[:-1], cfg)        # drop the last bar: nothing before it may change
    cols = [c for c in a.columns if c not in ("bar_close", "day_key")]
    for c in cols:
        x, y = a[c].iloc[:-1], b[c]
        if x.dtype.kind in "fc":
            assert np.allclose(x.to_numpy(dtype=float), y.to_numpy(dtype=float), equal_nan=True), c
        else:
            assert (x.astype(str).to_numpy() == y.astype(str).to_numpy()).all(), c


def test_pine_compression_matches_bar_by_bar_recurrence():
    """Re-implement the Pine tracker literally, bar by bar, and compare."""
    df = bars(400, seed=7)
    cfg = {**PO_CFG, "compression_mode": "pine"}
    f = phase_oscillator(df, cfg)
    pivot = ema(df["close"], 21).to_numpy()
    a = atr(df, 14).to_numpy()
    sd = df["close"].rolling(21).std(ddof=0).to_numpy()
    close = df["close"].to_numpy()
    expect = []
    comp_prev = np.nan
    for i in range(len(df)):
        if np.isnan(sd[i]) or np.isnan(a[i]):
            expect.append(False)
            comp_prev = np.nan
            continue
        bb_up, bb_dn = pivot[i] + 2 * sd[i], pivot[i] - 2 * sd[i]
        above = close[i] >= pivot[i]
        comp = (bb_up - (pivot[i] + 2 * a[i])) if above else ((pivot[i] - 2 * a[i]) - bb_dn)
        in_exp = (bb_up - (pivot[i] + 1.854 * a[i])) if above else ((pivot[i] - 1.854 * a[i]) - bb_dn)
        expansion = (not np.isnan(comp_prev)) and comp_prev <= comp
        if expansion and in_exp > 0:
            tracker = False
        elif comp <= 0:
            tracker = True
        else:
            tracker = False
        expect.append(tracker)
        comp_prev = comp
    got = f["compression"].to_numpy()
    assert (got[40:] == np.array(expect)[40:]).all()


def test_leaving_signals_use_pine_equalities():
    df = bars(300)
    f = phase_oscillator(df, {**PO_CFG, "compression_mode": "pine"})
    o, p = f["osc"], f["osc"].shift(1)
    pd.testing.assert_series_equal(f["leaving_extreme_down"], ((p <= -100) & (o > -100)), check_names=False)
    pd.testing.assert_series_equal(f["leaving_distribution"], ((p >= 61.8) & (o < 61.8)), check_names=False)
