"""window_trades: fills at the open of the first bar at/after the scheduled
time, drops (never shifts) trades it cannot fill, charges the full venue
round trip plus holding."""
import numpy as np
import pandas as pd
import pytest

from saty_bt.edge import OANDA, Venue, pf, summary, window_trades


def _bars(n=48, start="2024-01-02 00:00", freq="1h", step=0.01):
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    o = 100 + step * np.arange(n)
    return pd.DataFrame({"open": o, "high": o + 0.005, "low": o - 0.005, "close": o + step}, index=idx)


FREE = Venue("free", spread={})


def test_fill_at_open_of_first_bar_at_or_after():
    b = _bars()
    s = pd.DataFrame({"entry_time": [pd.Timestamp("2024-01-02 02:30", tz="UTC")],
                      "exit_time": [pd.Timestamp("2024-01-02 05:00", tz="UTC")], "direction": [1]})
    t = window_trades(b, s, "X", FREE, max_lag="2h")
    assert t.entry_time.iloc[0] == pd.Timestamp("2024-01-02 03:00", tz="UTC")
    assert t.exit_time.iloc[0] == pd.Timestamp("2024-01-02 05:00", tz="UTC")
    assert t.entry_price.iloc[0] == pytest.approx(100.03)
    assert t.exit_price.iloc[0] == pytest.approx(100.05)
    assert t.ret.iloc[0] == pytest.approx(0.02 / 100.03)
    assert t.hours.iloc[0] == pytest.approx(2.0)


def test_gap_drops_rather_than_shifts():
    b = _bars().drop(pd.date_range("2024-01-02 03:00", periods=3, freq="1h", tz="UTC"))
    s = pd.DataFrame({"entry_time": [pd.Timestamp("2024-01-02 03:00", tz="UTC")],
                      "exit_time": [pd.Timestamp("2024-01-02 08:00", tz="UTC")], "direction": [1]})
    t = window_trades(b, s, "X", FREE, max_lag="30min")
    assert len(t) == 0 and t.attrs["dropped"] == 1


def test_short_and_costs():
    b = _bars()
    s = pd.DataFrame({"entry_time": [pd.Timestamp("2024-01-02 01:00", tz="UTC")],
                      "exit_time": [pd.Timestamp("2024-01-02 11:00", tz="UTC")], "direction": [-1]})
    v = Venue("v", spread={"X": 0.02}, fee_side=0.0001, hold_bps_hour=1.0)
    t = window_trades(b, s, "X", v)
    pe = 100.01
    exp = -0.10 / pe - 0.02 / pe - 2 * 0.0001 - 1.0 * 10 / 1e4
    assert t.ret.iloc[0] == pytest.approx(exp)
    t2 = window_trades(b, s, "X", v, cost_mult=2.0)
    assert t2.ret.iloc[0] == pytest.approx(-0.10 / pe + 2 * (exp + 0.10 / pe))


def test_mixed_time_units():
    # parquet loads as ns, pandas 3 builds new stamps as us - must not matter
    b = _bars()
    b.index = b.index.as_unit("ns")
    s = pd.DataFrame({"entry_time": pd.DatetimeIndex(["2024-01-02 02:30"], tz="UTC").as_unit("us"),
                      "exit_time": pd.DatetimeIndex(["2024-01-02 05:00"], tz="UTC").as_unit("s"),
                      "direction": [1]})
    t = window_trades(b, s, "X", FREE, max_lag="45min")
    assert len(t) == 1 and t.hours.iloc[0] == pytest.approx(2.0)
    assert len(window_trades(b, s, "X", FREE, max_lag="20min")) == 0


def test_requires_real_utc():
    b = _bars()
    b.index = b.index.tz_localize(None)
    with pytest.raises(ValueError):
        window_trades(b, pd.DataFrame({"entry_time": [], "exit_time": [], "direction": []}), "X", FREE)


def test_oanda_round_trip_is_one_spread():
    assert OANDA.round_trip_frac("USDJPY", 140.0) == pytest.approx(0.014 / 140.0)


def test_summary_splits_regimes():
    t = pd.DataFrame({"exit_time": pd.to_datetime(["2019-06-01", "2019-07-01", "2022-01-01", "2022-02-01"], utc=True),
                      "pnl_usd": [2.0, -1.0, 3.0, -1.0], "ret": [2e-4, -1e-4, 3e-4, -1e-4]})
    s = summary(t, "2021-01-01")
    assert s["pf_old"] == pytest.approx(2.0) and s["pf_new"] == pytest.approx(3.0)
    assert pf([1, -1, 2]) == pytest.approx(3.0)
