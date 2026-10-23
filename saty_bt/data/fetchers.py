"""
Fetchers. Each returns a canonical frame (see schema.py) and a `source` dict
describing exactly what was used, which build.py writes into meta.json.

    histdata_cache  - the existing cache/histdata/<PAIR>_*.csv 1-minute files
    bitstamp_cache  - the existing cache/bitstamp_<pair>_<step>.csv
    yfinance        - equities, SPY/ES proxies, ^GSPC (daily unlimited,
                      1h ~730 days, 30m ~60 days)
    ccxt            - any CCXT exchange, paged OHLCV (BTC/USD)
    csv_import      - Dukascopy / OANDA / generic CSV or Parquet exports

Nothing here touches the existing cache files except to read them.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import resolve_path
from .schema import normalize

# yfinance history caps measured against the live endpoint (Sep 2026)
YF_MAX_DAYS = {"1m": 7, "5m": 59, "15m": 59, "30m": 59, "1h": 729, "1d": None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# existing caches
# ---------------------------------------------------------------------------

def fetch_histdata_cache(cfg: dict, symbol: str) -> tuple[pd.DataFrame, dict]:
    """Concatenate cache/histdata/<SYMBOL>_*.csv (1-minute, UTC, bid, no volume)."""
    root = resolve_path(cfg, "raw_cache") / "histdata"
    files = sorted(root.glob(f"{symbol}_*.csv"))
    if not files:
        raise FileNotFoundError(f"no HistData cache for {symbol} under {root}")
    parts = [pd.read_csv(f) for f in files]
    df = normalize(pd.concat(parts, ignore_index=True))
    # HistData volume is always 0 - it is not tick volume, it is absent.
    if (df["volume"].fillna(0) == 0).all():
        df["volume"] = np.nan
    src = {"kind": "histdata_cache", "symbol": symbol, "files": [f.name for f in files],
           "price": "bid", "volume_kind": "none",
           "tz_note": "HistData EST-fixed -> UTC; week opens 22:00 UTC Sunday",
           "read_at": _now()}
    return df, src


def fetch_bitstamp_cache(cfg: dict, symbol: str = "btcusd", step: int = 1800) -> tuple[pd.DataFrame, dict]:
    p = resolve_path(cfg, "raw_cache") / f"bitstamp_{symbol}_{step}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = normalize(pd.read_csv(p))
    src = {"kind": "bitstamp_cache", "symbol": symbol, "step_seconds": step,
           "file": p.name, "volume_kind": "real", "read_at": _now()}
    return df, src


# ---------------------------------------------------------------------------
# yfinance
# ---------------------------------------------------------------------------

def fetch_yfinance(symbol: str, interval: str, start: str | None = None,
                   end: str | None = None, exchange_tz: str = "America/New_York",
                   ) -> tuple[pd.DataFrame, dict]:
    """
    One yfinance history call. Intraday bars come back tz-aware in the
    exchange zone and stamped at bar open. Daily bars come back as naive
    dates; we stamp them at 09:30 exchange time so they line up with the
    daily bars we derive from intraday data ourselves.
    """
    import yfinance as yf

    cap = YF_MAX_DAYS.get(interval)
    kwargs = {"interval": interval, "auto_adjust": False, "actions": False, "prepost": False}
    if cap is not None:
        kwargs["period"] = f"{cap}d"
    else:
        kwargs["start"] = start or "2015-01-01"
        if end:
            kwargs["end"] = end
    hist = yf.Ticker(symbol).history(**kwargs)
    if hist is None or hist.empty:
        raise RuntimeError(f"yfinance returned nothing for {symbol} {interval}")
    hist = hist.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    idx = pd.DatetimeIndex(hist.index)
    if interval == "1d":
        # daily rows are dated by session; stamp them at the 09:30 open so
        # they align with daily bars we derive from intraday data ourselves
        local = idx.tz_convert(exchange_tz) if idx.tz is not None else idx.tz_localize(exchange_tz)
        idx = (local.tz_localize(None).normalize()
               + pd.Timedelta(hours=9, minutes=30)).tz_localize(exchange_tz)
    elif idx.tz is None:
        idx = idx.tz_localize(exchange_tz)
    hist.index = idx
    df = normalize(hist)
    # yfinance daily bars are stamped at the session date; drop a forming today bar
    if interval == "1d":
        today_open = pd.Timestamp.now(tz=exchange_tz).normalize() + pd.Timedelta(hours=9, minutes=30)
        df = df[df.index < today_open.tz_convert("UTC")]
    src = {"kind": "yfinance", "symbol": symbol, "interval": interval,
           "volume_kind": "real", "adjusted": False, "prepost": False,
           "cap_days": cap, "fetched_at": _now()}
    return df, src


# ---------------------------------------------------------------------------
# CCXT
# ---------------------------------------------------------------------------

def fetch_ccxt(exchange: str, symbol: str, timeframe: str, since: str,
               limit: int = 1000, pause: float = 0.25) -> tuple[pd.DataFrame, dict]:
    import ccxt

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    ex.load_markets()
    ms = int(pd.Timestamp(since, tz="UTC").timestamp() * 1000)
    tf_ms = ex.parse_timeframe(timeframe) * 1000
    rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=ms, limit=limit)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + tf_ms
        if nxt <= ms or len(batch) < 2:
            break
        ms = nxt
        time.sleep(pause)
        if ms > int(pd.Timestamp.now(tz="UTC").timestamp() * 1000):
            break
    if not rows:
        raise RuntimeError(f"ccxt returned nothing for {exchange} {symbol}")
    raw = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], unit="ms", utc=True)
    df = normalize(raw)
    # drop the forming candle
    tf_td = pd.Timedelta(milliseconds=tf_ms)
    df = df[df.index + tf_td <= pd.Timestamp.now(tz="UTC")]
    src = {"kind": "ccxt", "exchange": exchange, "symbol": symbol, "timeframe": timeframe,
           "since": since, "volume_kind": "real", "fetched_at": _now()}
    return df, src


# ---------------------------------------------------------------------------
# CSV / Parquet imports (Dukascopy, OANDA, generic)
# ---------------------------------------------------------------------------

def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def fetch_csv_import(path: str | Path, fmt: str = "generic",
                     tz: str | None = None) -> tuple[pd.DataFrame, dict]:
    """
    fmt = "dukascopy": columns 'Gmt time,Open,High,Low,Close,Volume',
          time like '01.01.2024 00:00:00.000', GMT. Volume = tick volume.
    fmt = "oanda":     columns 'time,o,h,l,c,volume' (or open/high/...), RFC3339 UTC.
    fmt = "generic":   timestamp,open,high,low,close,volume; tz-aware or pass tz.
    """
    path = Path(path)
    raw = _read_any(path)
    vol_kind = "unknown"
    if fmt == "dukascopy":
        raw = raw.rename(columns={"Gmt time": "timestamp"})
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], format="%d.%m.%Y %H:%M:%S.%f", utc=True)
        vol_kind = "tick"
    elif fmt == "oanda":
        raw = raw.rename(columns={"time": "timestamp", "o": "open", "h": "high",
                                  "l": "low", "c": "close"})
        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        vol_kind = "tick"
    df = normalize(raw, tz_naive_as=tz or ("UTC" if fmt != "generic" else None))
    src = {"kind": "csv_import", "format": fmt, "file": str(path),
           "volume_kind": vol_kind, "read_at": _now()}
    return df, src


# ---------------------------------------------------------------------------
# dispatcher
# ---------------------------------------------------------------------------

def fetch(cfg: dict, source: dict, interval: str | None = None) -> tuple[pd.DataFrame, dict]:
    kind = source["kind"]
    if kind == "histdata_cache":
        return fetch_histdata_cache(cfg, source["symbol"])
    if kind == "bitstamp_cache":
        return fetch_bitstamp_cache(cfg, source.get("symbol", "btcusd"), int(source.get("step", 1800)))
    if kind == "yfinance":
        yf_cfg = cfg.get("fetch", {}).get("yfinance", {})
        return fetch_yfinance(source["symbol"], interval or source.get("interval", "1d"),
                              start=yf_cfg.get("daily_start"))
    if kind == "ccxt":
        c = {**cfg.get("fetch", {}).get("ccxt", {}), **source}
        return fetch_ccxt(c["exchange"], c["symbol"], interval or c.get("timeframe", "30m"), c["since"])
    if kind == "csv_import":
        return fetch_csv_import(source["path"], source.get("format", "generic"), source.get("tz"))
    raise ValueError(f"unknown source kind {kind}")
