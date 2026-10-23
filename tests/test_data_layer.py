"""
Phase 1 tests: schema, calendar, resampler, store, config.

Synthetic 1-minute series are built so every expected value is known
exactly. Real-data tests run only if the HistData cache is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from saty_bt.config import (asset_config, config_hash, load_config,
                            timeframe_minutes)
from saty_bt.data import (Anchor, SessionSpec, aggregate_period, audit,
                          infer_step, normalize, period_key, resample_daily,
                          resample_intraday, session_frame, validate)
from saty_bt.data.schema import SchemaError

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def minute_bars(start: str, end: str, tz: str = "UTC", price0: float = 100.0,
                volume: float | None = 10.0) -> pd.DataFrame:
    """Continuous 1-minute bars; close = price0 + minute index, high=close+0.5, low=open-0.5."""
    idx = pd.date_range(start, end, freq="1min", tz=tz, inclusive="left").tz_convert("UTC")
    n = len(idx)
    close = price0 + np.arange(n, dtype=float)
    opn = np.r_[price0, close[:-1]]
    df = pd.DataFrame({"open": opn, "high": np.maximum(opn, close) + 0.5,
                       "low": np.minimum(opn, close) - 0.5, "close": close,
                       "volume": np.nan if volume is None else volume}, index=idx)
    df.index.name = "timestamp"
    return df


FX = SessionSpec(kind="fx", anchor=Anchor("fixed", 22, 0, "UTC"))
FX_NY = SessionSpec(kind="fx", anchor=Anchor("fixed", 17, 0, "America/New_York"))
CRYPTO = SessionSpec(kind="crypto", anchor=Anchor("fixed", 0, 0, "UTC"))
RTH = SessionSpec(kind="rth_us", anchor=Anchor("exchange"), tz="America/New_York", calendar="NYSE")


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------

def test_normalize_rejects_naive_without_tz():
    raw = pd.DataFrame({"timestamp": ["2024-01-01 00:00"], "open": [1], "high": [1], "low": [1], "close": [1]})
    with pytest.raises(SchemaError):
        normalize(raw)


def test_normalize_localizes_and_sorts_and_dedupes():
    raw = pd.DataFrame({"ts": ["2024-01-01 10:00", "2024-01-01 09:00", "2024-01-01 09:00"],
                        "Open": [2, 1, 1.5], "High": [2, 1, 1.5], "Low": [2, 1, 1.5], "Close": [2, 1, 1.5]})
    df = normalize(raw, tz_naive_as="America/New_York")
    validate(df)
    assert len(df) == 2
    assert df.index[0] == pd.Timestamp("2024-01-01 14:00", tz="UTC")
    assert df["volume"].isna().all()
    assert df["close"].iloc[0] == 1.5          # keep='last' on duplicates


def test_audit_flags_inverted_and_gaps():
    df = minute_bars("2024-01-01 00:00", "2024-01-01 01:00")
    df.loc[df.index[5], "high"] = df.loc[df.index[5], "low"] - 1     # inverted
    df = df.drop(df.index[30:40])                                     # 10-bar hole
    rep = audit(df, "t", pd.Timedelta("1min"))
    assert rep.inverted == 1
    assert len(rep.gaps) == 1 and rep.gaps[0][2] == 10
    assert len(rep.excluded) == 1


def test_infer_step():
    assert infer_step(minute_bars("2024-01-01", "2024-01-01 02:00")) == pd.Timedelta("1min")


# ---------------------------------------------------------------------------
# calendar / anchors
# ---------------------------------------------------------------------------

def test_fixed_anchor_utc_day_open():
    idx = pd.DatetimeIndex(["2024-06-03 21:59", "2024-06-03 22:00", "2024-06-04 03:00"], tz="UTC")
    sf = session_frame(idx, FX)
    assert list(sf["day_open"]) == [pd.Timestamp("2024-06-02 22:00", tz="UTC"),
                                    pd.Timestamp("2024-06-03 22:00", tz="UTC"),
                                    pd.Timestamp("2024-06-03 22:00", tz="UTC")]
    # FX day is named after the date it closes on: Sun 22:00 -> Mon 22:00 is "Monday"
    assert pd.Timestamp(sf["day_key"].iloc[0]).dayofweek == 0
    assert pd.Timestamp(sf["day_key"].iloc[1]).dayofweek == 1


def test_ny_anchor_is_dst_aware():
    winter = pd.DatetimeIndex(["2024-01-10 12:00"], tz="UTC")
    summer = pd.DatetimeIndex(["2024-07-10 12:00"], tz="UTC")
    assert session_frame(winter, FX_NY)["day_open"].iloc[0] == pd.Timestamp("2024-01-09 22:00", tz="UTC")
    assert session_frame(summer, FX_NY)["day_open"].iloc[0] == pd.Timestamp("2024-07-09 21:00", tz="UTC")


def test_fx_weekend_is_masked():
    # Friday 2024-06-07 22:00 UTC -> Sunday 2024-06-09 22:00 UTC is closed
    idx = pd.DatetimeIndex(["2024-06-07 21:59", "2024-06-07 22:00", "2024-06-08 12:00",
                            "2024-06-09 21:59", "2024-06-09 22:00"], tz="UTC")
    m = session_frame(idx, FX)["in_session"].tolist()
    assert m == [True, False, False, False, True]


def test_crypto_never_masked():
    idx = pd.date_range("2024-06-07", "2024-06-10", freq="6h", tz="UTC")
    assert session_frame(idx, CRYPTO)["in_session"].all()


def test_rth_mask_holiday_and_early_close():
    # 2024-07-03 early close 13:00, 2024-07-04 holiday, 2024-07-05 normal
    idx = pd.DatetimeIndex([
        "2024-07-03 13:29", "2024-07-03 13:30", "2024-07-03 16:59", "2024-07-03 17:00",  # 09:29,09:30,12:59,13:00 ET
        "2024-07-04 14:00",                                                              # holiday
        "2024-07-05 13:30", "2024-07-05 19:59", "2024-07-05 20:00",                      # 09:30, 15:59, 16:00 ET
    ], tz="UTC")
    sf = session_frame(idx, RTH)
    assert sf["in_session"].tolist() == [False, True, True, False, False, True, True, False]
    assert bool(sf["early_close"].iloc[1]) is True
    assert bool(sf["early_close"].iloc[5]) is False
    assert sf["day_close"].iloc[1] == pd.Timestamp("2024-07-03 17:00", tz="UTC")


def test_rth_dst_shift():
    # 09:30 ET is 14:30 UTC in January, 13:30 UTC in July
    jan = session_frame(pd.DatetimeIndex(["2024-01-10 14:30"], tz="UTC"), RTH)
    jul = session_frame(pd.DatetimeIndex(["2024-07-10 13:30"], tz="UTC"), RTH)
    assert jan["in_session"].iloc[0] and jul["in_session"].iloc[0]
    assert jan["day_open"].iloc[0] == pd.Timestamp("2024-01-10 14:30", tz="UTC")
    assert jul["day_open"].iloc[0] == pd.Timestamp("2024-07-10 13:30", tz="UTC")


def test_period_key():
    dk = pd.DatetimeIndex(["2024-06-05", "2024-06-07", "2024-06-10", "2024-08-01", "2024-11-15"])
    assert list(period_key(dk, "1w").strftime("%Y-%m-%d")) == ["2024-06-03", "2024-06-03", "2024-06-10", "2024-07-29", "2024-11-11"]
    assert list(period_key(dk, "1M").strftime("%Y-%m-%d")) == ["2024-06-01"] * 3 + ["2024-08-01", "2024-11-01"]
    assert list(period_key(dk, "3M").strftime("%Y-%m-%d")) == ["2024-04-01"] * 3 + ["2024-07-01", "2024-10-01"]
    assert list(period_key(dk, "1Y").strftime("%Y-%m-%d")) == ["2024-01-01"] * 5


# ---------------------------------------------------------------------------
# resampling - FX fixed anchor
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fx_week():
    # Sunday 2024-06-02 22:00 -> Friday 2024-06-07 22:00 UTC, no weekend bars
    return minute_bars("2024-06-02 22:00", "2024-06-07 22:00")


def test_fx_30m_bins_and_ohlc(fx_week):
    out = resample_intraday(fx_week, "30m", FX)
    assert out.index[0] == pd.Timestamp("2024-06-02 22:00", tz="UTC")
    first = out.iloc[0]
    src = fx_week.iloc[:30]
    assert first["open"] == src["open"].iloc[0]
    assert first["close"] == src["close"].iloc[-1]
    assert first["high"] == src["high"].max() and first["low"] == src["low"].min()
    assert first["volume"] == 300.0
    assert (out.index.to_series().diff().dropna() == pd.Timedelta("30min")).all()
    assert len(out) == 5 * 48


def test_fx_4h_bins_start_at_anchor(fx_week):
    out = resample_intraday(fx_week, "4h", FX)
    hours = sorted(set(out.index.hour))
    assert hours == [2, 6, 10, 14, 18, 22]
    assert len(out) == 5 * 6


def test_fx_daily_named_by_close_date(fx_week):
    d = resample_daily(fx_week, FX)
    assert len(d) == 5
    assert list(pd.DatetimeIndex(d["day_key"]).dayofweek) == [0, 1, 2, 3, 4]
    assert d.index[0] == pd.Timestamp("2024-06-02 22:00", tz="UTC")
    assert d["bar_close"].iloc[0] == pd.Timestamp("2024-06-03 22:00", tz="UTC")
    assert not d["partial"].any()
    assert d["close"].iloc[0] == fx_week["close"].iloc[24 * 60 - 1]


def test_incomplete_tail_dropped():
    df = minute_bars("2024-06-02 22:00", "2024-06-03 05:17")       # ends mid 4h bin
    out = resample_intraday(df, "4h", FX)
    # bins: 22:00-02:00 complete, 02:00-06:00 incomplete -> dropped
    assert list(out.index) == [pd.Timestamp("2024-06-02 22:00", tz="UTC")]
    d = resample_daily(df, FX)
    assert d.empty


def test_incomplete_head_dropped():
    df = minute_bars("2024-06-03 00:00", "2024-06-03 10:00")       # starts mid 4h bin
    out = resample_intraday(df, "4h", FX)
    # 22:00-02:00 only has 00:00 onward -> dropped; 02:00 and 06:00 complete
    assert [t.hour for t in out.index] == [2, 6]


def test_interior_hole_kept_but_edges_trimmed():
    df = minute_bars("2024-06-02 22:00", "2024-06-03 22:00")
    df = df.drop(df.index[200:230])                                # a 30-minute hole inside
    out = resample_intraday(df, "4h", FX)
    assert len(out) == 6                                           # interior bin survives
    d = resample_daily(df, FX)
    assert len(d) == 1 and not d["partial"].iloc[0]


def test_weekend_bars_never_contribute():
    df = minute_bars("2024-06-07 20:00", "2024-06-10 00:00")      # spans the weekend
    out = resample_intraday(df, "1h", FX)
    assert not ((out.index >= "2024-06-07 22:00") & (out.index < "2024-06-09 22:00")).any()


def test_nan_volume_stays_nan():
    df = minute_bars("2024-06-03 00:00", "2024-06-03 04:00", volume=None)
    out = resample_intraday(df, "30m", FX)
    assert out["volume"].isna().all()


def test_no_lookahead_bar_uses_only_its_own_window(fx_week):
    """Perturb a minute inside bin k+1; bin k must not change."""
    out_a = resample_intraday(fx_week, "1h", FX)
    df = fx_week.copy()
    ts = pd.Timestamp("2024-06-03 01:30", tz="UTC")               # inside the 01:00 bin
    df.loc[ts, ["high", "close"]] = 1e6
    out_b = resample_intraday(df, "1h", FX)
    k = pd.Timestamp("2024-06-03 00:00", tz="UTC")
    pd.testing.assert_series_equal(out_a.loc[k], out_b.loc[k])
    assert out_b.loc[pd.Timestamp("2024-06-03 01:00", tz="UTC"), "high"] == 1e6


# ---------------------------------------------------------------------------
# resampling - RTH
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def rth_week():
    # continuous minutes over Thanksgiving week 2024 (holiday 11/28, early close 11/29)
    return minute_bars("2024-11-25 00:00", "2024-12-03 00:00", tz="America/New_York")


def test_rth_daily_bars(rth_week):
    d = resample_daily(rth_week, RTH)
    d_local = d.index.tz_convert("America/New_York")
    assert [t.strftime("%m-%d") for t in d_local] == ["11-25", "11-26", "11-27", "11-29", "12-02"]
    assert all(t.hour == 9 and t.minute == 30 for t in d_local)
    # early close day has 3.5h of bars, normal day 6.5h
    assert d["src_bars"].tolist() == [390, 390, 390, 210, 390]
    assert d["early_close"].tolist() == [False, False, False, True, False]
    assert d["bar_close"].iloc[3].tz_convert("America/New_York").hour == 13


def test_rth_4h_bins(rth_week):
    out = resample_intraday(rth_week, "4h", RTH)
    loc = out.index.tz_convert("America/New_York")
    normal = out[loc.strftime("%m-%d") == "11-26"]
    assert [t.strftime("%H:%M") for t in normal.index.tz_convert("America/New_York")] == ["09:30", "13:30"]
    assert normal["bar_close"].iloc[1].tz_convert("America/New_York").strftime("%H:%M") == "16:00"
    early = out[loc.strftime("%m-%d") == "11-29"]
    assert [t.strftime("%H:%M") for t in early.index.tz_convert("America/New_York")] == ["09:30"]
    assert early["bar_close"].iloc[0].tz_convert("America/New_York").strftime("%H:%M") == "13:00"


def test_rth_1h_last_bar_is_half_hour(rth_week):
    out = resample_intraday(rth_week, "1h", RTH)
    loc = out.index.tz_convert("America/New_York")
    day = out[loc.strftime("%m-%d") == "11-26"]
    assert len(day) == 7
    assert day.index[-1].tz_convert("America/New_York").strftime("%H:%M") == "15:30"
    assert day["volume"].iloc[-1] == 300.0 and day["volume"].iloc[0] == 600.0


def test_rth_no_premarket_leaks(rth_week):
    out = resample_intraday(rth_week, "30m", RTH)
    loc = out.index.tz_convert("America/New_York")
    assert loc.min().strftime("%H:%M") == "09:30"
    assert (loc.hour * 60 + loc.minute).max() == 15 * 60 + 30
    assert not (loc.strftime("%m-%d") == "11-28").any()


# ---------------------------------------------------------------------------
# period aggregation
# ---------------------------------------------------------------------------

def test_weekly_from_daily_crypto():
    df = minute_bars("2024-06-03 00:00", "2024-06-24 00:00")   # 3 full Mon-Sun weeks
    d = resample_daily(df, CRYPTO)
    assert len(d) == 21
    w = aggregate_period(d, "1w")
    assert len(w) == 3
    assert w["days"].tolist() == [7, 7, 7]
    assert w["open"].iloc[0] == d["open"].iloc[0] and w["close"].iloc[0] == d["close"].iloc[6]
    assert w["high"].iloc[0] == d["high"].iloc[:7].max()


def test_monthly_drops_incomplete_periods_at_both_ends():
    df = minute_bars("2024-04-20 00:00", "2024-06-12 00:00")
    d = resample_daily(df, CRYPTO)
    m = aggregate_period(d, "1M")
    # April is entered midway, June is unfinished: only May survives
    assert list(m["period_key"].dt.strftime("%Y-%m")) == ["2024-05"]
    assert m["days"].iloc[0] == 31
    assert m["open"].iloc[0] == d.loc[pd.Timestamp("2024-05-01", tz="UTC"), "open"]


def test_yearly_with_no_complete_year_is_empty_not_error():
    df = minute_bars("2024-05-01 00:00", "2024-05-20 00:00")
    y = aggregate_period(resample_daily(df, CRYPTO), "1Y")
    assert y.empty


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

def test_config_loads_and_hash_is_stable():
    cfg = load_config(ROOT / "config.yaml")
    h1 = config_hash(cfg)
    cfg2 = load_config(ROOT / "config.yaml")
    assert h1 == config_hash(cfg2) and len(h1) == 12
    a = asset_config(cfg, "SPX")
    assert a["session"]["kind"] == "rth_us"
    assert a["volume"]["mode"] == "proxy"
    assert a["costs"]["spread"] > 0
    fx = asset_config(cfg, "EURUSD")
    assert fx["volume_gate"] == "advisory"
    assert fx["risk"]["sizing"] == "fixed_margin"


def test_costs_never_zero_by_default():
    cfg = load_config(ROOT / "config.yaml")
    for name in cfg["assets"]:
        c = asset_config(cfg, name)["costs"]
        assert (c.get("spread", 0) + c.get("slippage", 0) + c.get("slippage_bps", 0)
                + c.get("commission_bps", 0)) > 0, name


def test_timeframe_minutes():
    assert [timeframe_minutes(t) for t in ("30m", "1h", "4h", "1d")] == [30, 60, 240, 1440]


# ---------------------------------------------------------------------------
# real data (skipped when the cache is absent)
# ---------------------------------------------------------------------------

HIST = ROOT / "cache" / "histdata"


@pytest.mark.skipif(not (HIST / "EURUSD_2024.csv").exists(), reason="HistData cache absent")
def test_real_histdata_eurusd_week():
    raw = normalize(pd.read_csv(HIST / "EURUSD_2024.csv"))
    raw = raw[(raw.index >= "2024-06-02") & (raw.index < "2024-06-09")]
    d = resample_daily(raw, FX)
    assert len(d) == 5
    assert d["partial"].sum() == 0
    h4 = resample_intraday(raw, "4h", FX)
    assert sorted(set(h4.index.hour)) == [2, 6, 10, 14, 18, 22]


@pytest.mark.skipif(not (HIST / "SPXUSD_2024.csv").exists(), reason="HistData cache absent")
def test_real_spx_rth_daily_has_no_holidays():
    raw = normalize(pd.read_csv(HIST / "SPXUSD_2024.csv"))
    raw = raw[(raw.index >= "2024-11-24") & (raw.index < "2024-12-04")]
    d = resample_daily(raw, RTH)
    days = [t.strftime("%m-%d") for t in d.index.tz_convert("America/New_York")]
    assert "11-28" not in days
    assert d.loc[d.index.tz_convert("America/New_York").strftime("%m-%d") == "11-29", "early_close"].all()
