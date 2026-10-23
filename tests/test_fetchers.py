"""Fetcher tests: file-format imports run offline; network fetchers are smoke
tests that skip cleanly when there is no connectivity."""

from __future__ import annotations

import socket

import pandas as pd
import pytest

from saty_bt.data.fetchers import fetch_csv_import, fetch_ccxt, fetch_yfinance
from saty_bt.data.schema import validate


def _online(host="api.exchange.coinbase.com") -> bool:
    try:
        socket.create_connection((host, 443), timeout=3).close()
        return True
    except OSError:
        return False


def test_dukascopy_import(tmp_path):
    p = tmp_path / "EURUSD_Candlestick_1_M_BID.csv"
    p.write_text("Gmt time,Open,High,Low,Close,Volume\n"
                 "01.06.2024 22:00:00.000,1.08400,1.08410,1.08390,1.08405,123.4\n"
                 "01.06.2024 22:01:00.000,1.08405,1.08420,1.08400,1.08415,98.1\n")
    df, src = fetch_csv_import(p, fmt="dukascopy")
    validate(df)
    assert df.index[0] == pd.Timestamp("2024-06-01 22:00", tz="UTC")
    assert src["volume_kind"] == "tick"
    assert df["volume"].iloc[1] == 98.1


def test_oanda_import(tmp_path):
    p = tmp_path / "oanda.csv"
    p.write_text("time,o,h,l,c,volume\n"
                 "2024-06-02T21:00:00Z,1.084,1.0841,1.0839,1.08405,12\n"
                 "2024-06-02T21:30:00Z,1.08405,1.0842,1.084,1.08415,9\n")
    df, src = fetch_csv_import(p, fmt="oanda")
    validate(df)
    assert len(df) == 2 and src["volume_kind"] == "tick"


def test_generic_import_requires_tz(tmp_path):
    p = tmp_path / "g.csv"
    p.write_text("timestamp,open,high,low,close,volume\n2024-06-02 21:00,1,1,1,1,0\n")
    with pytest.raises(Exception):
        fetch_csv_import(p, fmt="generic")
    df, _ = fetch_csv_import(p, fmt="generic", tz="America/New_York")
    assert df.index[0] == pd.Timestamp("2024-06-03 01:00", tz="UTC")


def test_generic_parquet_roundtrip(tmp_path):
    idx = pd.date_range("2024-06-03", periods=3, freq="30min", tz="UTC")
    raw = pd.DataFrame({"timestamp": idx, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 3.0})
    p = tmp_path / "x.parquet"
    raw.to_parquet(p)
    df, _ = fetch_csv_import(p, fmt="generic")
    validate(df)
    assert len(df) == 3


@pytest.mark.skipif(not _online(), reason="offline")
def test_ccxt_smoke():
    since = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=2)).strftime("%Y-%m-%d")
    df, src = fetch_ccxt("coinbase", "BTC/USD", "1h", since)
    validate(df)
    assert len(df) > 20
    assert df.index[-1] + pd.Timedelta("1h") <= pd.Timestamp.now(tz="UTC")   # no forming candle


@pytest.mark.skipif(not _online("query1.finance.yahoo.com"), reason="offline")
def test_yfinance_daily_stamped_at_open():
    df, src = fetch_yfinance("SPY", "1d", start="2024-01-01")
    validate(df)
    local = df.index.tz_convert("America/New_York")
    assert all(t.hour == 9 and t.minute == 30 for t in local[:5])
    assert src["adjusted"] is False
