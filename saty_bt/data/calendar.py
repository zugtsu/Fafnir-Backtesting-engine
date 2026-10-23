"""
Sessions, anchors and calendars.

A *session spec* says when an asset trades and where its trading day starts:

    rth_us : NYSE 09:30-16:00 America/New_York, holiday and early-close aware.
             The trading day is the exchange session; the day's "open" is the
             market open, and the daily close is the exchange close.
    fx     : continuous Sun->Fri. The day starts at the anchor (17:00 New York
             by convention; HistData's feed is fixed 17:00 EST = 22:00 UTC).
             The weekend, anchor Friday -> anchor Sunday, is dropped.
    crypto : continuous 24/7. The day starts at the anchor (00:00 UTC default).

Every function takes UTC timestamps and returns UTC timestamps. The only
timezone arithmetic happens inside, on wall-clock time, so DST is handled
once and in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

WEEKEND_START_DOW = 4   # Friday
WEEKEND_END_DOW = 6     # Sunday


@dataclass(frozen=True)
class Anchor:
    kind: str                 # "fixed" | "exchange"
    hour: int = 0
    minute: int = 0
    tz: str = "UTC"

    @property
    def offset(self) -> pd.Timedelta:
        return pd.Timedelta(hours=self.hour, minutes=self.minute)


@dataclass(frozen=True)
class SessionSpec:
    kind: str                 # "rth_us" | "fx" | "crypto"
    anchor: Anchor
    tz: str = "UTC"
    calendar: str | None = None

    @classmethod
    def from_config(cls, block: dict) -> "SessionSpec":
        a = block.get("anchor", {"kind": "fixed"})
        anchor = Anchor(kind=a.get("kind", "fixed"), hour=int(a.get("hour", 0)),
                        minute=int(a.get("minute", 0)), tz=a.get("tz", "UTC"))
        return cls(kind=block["kind"], anchor=anchor,
                   tz=block.get("tz", anchor.tz), calendar=block.get("calendar"))

    @property
    def continuous(self) -> bool:
        return self.kind in ("fx", "crypto")


# ---------------------------------------------------------------------------
# Exchange calendar (NYSE) - holidays and early closes
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def exchange_schedule(calendar: str, start: str, end: str) -> pd.DataFrame:
    """
    One row per trading day: market_open, market_close (UTC), plus
    `early_close` (bool). Uses pandas_market_calendars, which knows the
    holiday list and the 13:00 early closes.
    """
    import pandas_market_calendars as mcal
    cal = mcal.get_calendar(calendar)
    sched = cal.schedule(start_date=start, end_date=end)
    sched = sched[["market_open", "market_close"]].copy()
    sched["market_open"] = sched["market_open"].dt.tz_convert("UTC")
    sched["market_close"] = sched["market_close"].dt.tz_convert("UTC")
    local_close = sched["market_close"].dt.tz_convert(cal.tz)
    sched["early_close"] = (local_close.dt.hour * 60 + local_close.dt.minute) < 16 * 60
    sched.index = pd.DatetimeIndex(sched.index).tz_localize(None).normalize()
    sched.index.name = "session_date"
    return sched


def _schedule_for(index: pd.DatetimeIndex, spec: SessionSpec) -> pd.DataFrame:
    start = (index[0] - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    end = (index[-1] + pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    return exchange_schedule(spec.calendar or "NYSE", start, end)


# ---------------------------------------------------------------------------
# Day assignment
# ---------------------------------------------------------------------------

def _fixed_day_open(index: pd.DatetimeIndex, anchor: Anchor) -> pd.DatetimeIndex:
    """UTC timestamp of the anchor that starts the trading day containing each bar."""
    local = index.tz_convert(anchor.tz)
    shifted = local - anchor.offset
    # normalize() on a tz-aware index works on wall-clock time; anchors are
    # never at 02:00 so DST transitions cannot make this ambiguous.
    day = shifted.normalize()
    day_open_local = day + anchor.offset
    return day_open_local.tz_convert("UTC")


def session_frame(index: pd.DatetimeIndex, spec: SessionSpec) -> pd.DataFrame:
    """
    For every bar: `in_session` (bool), `day_open`, `day_close` (UTC) and
    `day_key` (the calendar date the trading day is named after).

    Bars outside the session (equity pre/post market, FX weekend) get
    in_session=False and NaT day boundaries.
    """
    out = pd.DataFrame(index=index)
    if spec.anchor.kind == "exchange":
        sched = _schedule_for(index, spec)
        # each bar belongs to the session whose open is the latest <= bar ts
        opens = sched["market_open"].values
        pos = np.searchsorted(opens, index.values, side="right") - 1
        pos = np.clip(pos, 0, len(sched) - 1)
        s = sched.iloc[pos]
        day_open = pd.DatetimeIndex(s["market_open"].values, tz="UTC")
        day_close = pd.DatetimeIndex(s["market_close"].values, tz="UTC")
        in_sess = (index >= day_open) & (index < day_close)
        out["in_session"] = in_sess
        out["day_open"] = day_open.where(in_sess, pd.NaT)
        out["day_close"] = day_close.where(in_sess, pd.NaT)
        out["early_close"] = np.where(in_sess, s["early_close"].values, False)
        keys = pd.DatetimeIndex(s.index.values)
        out["day_key"] = keys.where(in_sess, pd.NaT)
        return out

    day_open = _fixed_day_open(index, spec.anchor)
    day_close = day_open + pd.Timedelta(days=1)
    # trading-day name: the local calendar date on which the day CLOSES for
    # FX (Sun 17:00 -> Mon 17:00 is "Monday"), the date it opens for crypto.
    if spec.kind == "fx":
        key_local = (day_open + pd.Timedelta(days=1)).tz_convert(spec.anchor.tz).normalize()
        dow = key_local.dayofweek
        in_sess = ~np.isin(dow, [5, 6])          # Saturday/Sunday-named days are the weekend
    else:
        key_local = day_open.tz_convert(spec.anchor.tz).normalize()
        in_sess = np.ones(len(index), dtype=bool)
    out["in_session"] = in_sess
    out["day_open"] = day_open.where(in_sess, pd.NaT)
    out["day_close"] = day_close.where(in_sess, pd.NaT)
    out["early_close"] = False
    out["day_key"] = pd.DatetimeIndex(key_local.tz_localize(None)).where(in_sess, pd.NaT)
    return out


def session_mask(index: pd.DatetimeIndex, spec: SessionSpec) -> np.ndarray:
    return session_frame(index, spec)["in_session"].to_numpy()


def trading_days(spec: SessionSpec, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Naive calendar dates that are trading days for this session in [start, end]."""
    if spec.anchor.kind == "exchange":
        sched = exchange_schedule(spec.calendar or "NYSE",
                                  start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
        return sched.index
    days = pd.date_range(start.normalize().tz_localize(None), end.normalize().tz_localize(None), freq="D")
    if spec.kind == "fx":
        days = days[days.dayofweek < 5]
    return days


# ---------------------------------------------------------------------------
# Period keys for Multiday / Swing / Position / Long-term level grids
# ---------------------------------------------------------------------------

def period_key(day_key: pd.DatetimeIndex, period: str) -> pd.DatetimeIndex:
    """
    Map a trading-day date to the first calendar date of its level period.
        1d -> itself, 1w -> Monday of its ISO week, 1M -> first of month,
        3M -> first of quarter, 1Y -> Jan 1.
    """
    dk = pd.DatetimeIndex(day_key)
    if period == "1d":
        return dk
    if period == "1w":
        return dk - pd.to_timedelta(dk.dayofweek, unit="D")
    if period == "1M":
        return dk.to_period("M").to_timestamp()
    if period == "3M":
        return dk.to_period("Q").to_timestamp()
    if period == "1Y":
        return dk.to_period("Y").to_timestamp()
    raise ValueError(f"unknown period {period}")


def utc_wall(ts: pd.Timestamp, tz: str) -> pd.Timestamp:
    """Convenience: a UTC timestamp's wall-clock reading in `tz`."""
    return ts.tz_convert(tz)
