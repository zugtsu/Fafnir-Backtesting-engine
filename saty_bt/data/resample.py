"""
Session-aware resampling.

Rules that make the output safe for a no-look-ahead engine:

  * Bars are stamped at bin OPEN. Bin k of a trading day opens at
    day_open + k*tf. For RTH that gives 09:30, 10:00, ... on 30m; 09:30,
    13:30 on 4H (the second bar is 2.5h long, matching TradingView); for FX
    with a 22:00 UTC anchor it gives 22:00, 02:00, 06:00 ... on 4H.
  * A bin is emitted only if the source covers it to the end. A trailing,
    half-formed bar is dropped rather than published with a partial close.
  * Bars outside the session (pre/post market, FX weekend) never contribute.
  * Daily bars aggregate one trading day; weekly/monthly/quarterly/yearly
    bars aggregate daily bars by `period_key`.

Volume is summed; if the source volume is entirely NaN the result is NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import timeframe_minutes
from .calendar import SessionSpec, period_key, session_frame
from .schema import infer_step

_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _agg(groups: pd.core.groupby.DataFrameGroupBy) -> pd.DataFrame:
    out = groups.agg(_AGG)
    all_nan = groups["volume"].apply(lambda s: s.isna().all())
    out.loc[all_nan[all_nan].index, "volume"] = np.nan
    return out


def resample_intraday(df: pd.DataFrame, tf: str, spec: SessionSpec,
                      step: pd.Timedelta | None = None) -> pd.DataFrame:
    """
    Source bars (any spacing that divides `tf`) -> `tf` bars, session-aware.
    Returns canonical OHLCV plus `day_key`, `day_open`, `bar_close` columns.
    """
    tf_min = timeframe_minutes(tf)
    if tf_min >= 1440:
        raise ValueError("use resample_daily for 1d")
    step = step or infer_step(df)
    if pd.Timedelta(minutes=tf_min) < step:
        raise ValueError(f"cannot build {tf} from {step} bars")

    sf = session_frame(df.index, spec)
    keep = sf["in_session"].to_numpy()
    d = df[keep].copy()
    sf = sf[keep]

    day_open = pd.DatetimeIndex(sf["day_open"])
    day_close = pd.DatetimeIndex(sf["day_close"])
    k = ((d.index - day_open) // pd.Timedelta(minutes=tf_min)).astype("int64")
    bin_open = day_open + pd.to_timedelta(k * tf_min, unit="m")
    bin_close = np.minimum((bin_open + pd.Timedelta(minutes=tf_min)).values, day_close.values)
    bin_close = pd.DatetimeIndex(bin_close, tz="UTC")

    d["bin_open"] = bin_open
    d["bin_close"] = bin_close
    d["day_key"] = sf["day_key"].to_numpy()
    d["day_open"] = day_open
    d["src_start"] = d.index
    d["src_end"] = d.index + step               # when this source bar is complete

    g = d.groupby("bin_open", sort=True)
    out = _agg(g)
    meta = g.agg(bar_close=("bin_close", "first"), day_key=("day_key", "first"),
                 day_open=("day_open", "first"), covered_from=("src_start", "min"),
                 covered_to=("src_end", "max"))
    out = out.join(meta)

    # Completeness at both ends of the series: a bin is published only if the
    # source starts at (or within `tol` of) the bin open and reaches its close.
    # Interior bins with holes are kept - that is a data gap, not look-ahead.
    tol = _tolerance(step)
    lead_ok = out["covered_from"] <= out.index + tol
    trail_ok = out["covered_to"] >= out["bar_close"] - tol
    out = _trim_ends(out, lead_ok & trail_ok)
    out = out.drop(columns=["covered_from", "covered_to"])
    out.index.name = "timestamp"
    return out


def _tolerance(step: pd.Timedelta) -> pd.Timedelta:
    """Feeds built from 1m/5m bars may miss a minute at a session edge
    (HistData stops at :58 on Friday). Coarser sources get no slack."""
    return 3 * step if step <= pd.Timedelta("5min") else pd.Timedelta(0)


def _trim_ends(out: pd.DataFrame, ok: pd.Series) -> pd.DataFrame:
    """Drop incomplete bars at the head and tail; keep interior ones."""
    if out.empty:
        return out
    okv = ok.to_numpy()
    first = int(np.argmax(okv)) if okv.any() else len(okv)
    last = len(okv) - int(np.argmax(okv[::-1])) if okv.any() else 0
    return out.iloc[first:last]


def resample_daily(df: pd.DataFrame, spec: SessionSpec,
                   step: pd.Timedelta | None = None) -> pd.DataFrame:
    """One bar per trading day, stamped at the day's open. Adds `day_key`, `bar_close`."""
    step = step or infer_step(df)
    sf = session_frame(df.index, spec)
    keep = sf["in_session"].to_numpy()
    d = df[keep].copy()
    sf = sf[keep]
    d["day_open"] = pd.DatetimeIndex(sf["day_open"])
    d["day_close"] = pd.DatetimeIndex(sf["day_close"])
    d["day_key"] = sf["day_key"].to_numpy()
    d["early_close"] = sf["early_close"].to_numpy()
    d["src_start"] = d.index
    d["src_end"] = d.index + step

    g = d.groupby("day_open", sort=True)
    out = _agg(g)
    meta = g.agg(bar_close=("day_close", "first"), day_key=("day_key", "first"),
                 early_close=("early_close", "first"), covered_from=("src_start", "min"),
                 covered_to=("src_end", "max"), src_bars=("close", "size"))
    out = out.join(meta)

    tol = _tolerance(step)
    complete = ((out["covered_from"] <= out.index + tol)
                & (out["covered_to"] >= out["bar_close"] - tol))
    # Leading/trailing partial days are dropped (a forming today bar, or a
    # series that starts mid-session). Interior partial days - a holiday
    # half-session on a continuous market - are kept and flagged.
    out = _trim_ends(out, complete)
    out["partial"] = ~complete.loc[out.index]
    out = out.drop(columns=["covered_from", "covered_to"])
    out.index.name = "timestamp"
    return out


def aggregate_period(daily: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    Daily bars -> weekly (1w) / monthly (1M) / quarterly (3M) / yearly (1Y)
    bars, stamped at the first daily bar's open. Adds `period_key`,
    `bar_close` (last daily close time) and `days` (how many sessions).
    Only complete periods are returned: the trailing period is dropped if the
    daily series ends before that period's last possible session.
    """
    if period == "1d":
        out = daily.copy()
        out["period_key"] = out["day_key"]
        return out
    if daily.empty:
        return daily.iloc[0:0].assign(period_key=pd.NaT, days=0)

    pk = period_key(pd.DatetimeIndex(daily["day_key"]), period)
    d = daily.copy()
    d["period_key"] = pk
    d["ts"] = d.index
    g = d.groupby("period_key", sort=True)
    out = _agg(g)
    meta = g.agg(timestamp=("ts", "first"), bar_close=("bar_close", "last"),
                 days=("close", "size"), first_day=("day_key", "first"),
                 last_day=("day_key", "last"))
    out = out.join(meta)
    out["period_key"] = out.index
    out = out.set_index("timestamp")
    out.index.name = "timestamp"

    # A period is complete when its first observed session sits within a few
    # days of the period start and its last within a few days of the period
    # end. The tolerance absorbs weekends and holidays; a period the series
    # enters or leaves midway is dropped so no ATR is ever built on a stub.
    nxt = {"1w": pd.DateOffset(weeks=1), "1M": pd.DateOffset(months=1),
           "3M": pd.DateOffset(months=3), "1Y": pd.DateOffset(years=1)}[period]
    tol = pd.Timedelta(days=4)
    start = pd.DatetimeIndex(out["period_key"])
    end = pd.DatetimeIndex([s + nxt for s in start]) - pd.Timedelta(days=1)
    first_ok = pd.DatetimeIndex(out["first_day"]) <= start + tol
    last_ok = pd.DatetimeIndex(out["last_day"]) >= end - tol
    out["complete"] = first_ok & last_ok
    out = out[out["complete"]].drop(columns=["first_day", "last_day", "complete"])
    return out


def resample_all(df: pd.DataFrame, timeframes: list[str], spec: SessionSpec,
                 periods: tuple[str, ...] = ("1w", "1M", "3M", "1Y")) -> dict[str, pd.DataFrame]:
    """Every chart timeframe plus the level-period bars, from one base series."""
    step = infer_step(df)
    out: dict[str, pd.DataFrame] = {}
    daily = resample_daily(df, spec, step)
    for tf in timeframes:
        if timeframe_minutes(tf) >= 1440:
            out["1d"] = daily
        else:
            out[tf] = resample_intraday(df, tf, spec, step)
    out.setdefault("1d", daily)
    for p in periods:
        out[p] = aggregate_period(daily, p)
    return out


def first_bar_of_day(df: pd.DataFrame) -> pd.Series:
    """True on the first bar of each trading day (needs `day_key`)."""
    return df["day_key"] != df["day_key"].shift(1)
