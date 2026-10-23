"""
True wall-clock time for the HistData feeds.

HistData says its clock is "EST without daylight saving". It is not: the US
jobs report (08:30 New York) lands on the 13:30 stamp in BOTH summer and
winter, and the 14:00 Fed decision on the 19:00 stamp in both, on USDJPY,
EURUSD and XAUUSD alike (found 2026-10-05). The importer adds a fixed 5 hours,
so every stored stamp is

    New York local wall time + 5 hours

which is true UTC in winter and ONE HOUR LATE during US daylight saving
(about eight months a year). SPX and the crypto feeds are true UTC.

Consequences for anything already in the repo: the "07:00-09:00 UTC" London
box the backtests measured is really 02:00-04:00 New York = 07:00-09:00 London
local almost all year, and the 22:00 daily anchor is the 17:00 New York close
all year (which is the correct FX convention). Tokyo has no daylight saving,
so a Tokyo-time event such as the 09:55 JST fix sits on different stamps in
summer and winter - use true_utc() for anything keyed to a clock outside New
York.

The store itself is left as it is: every golden master and published number
was measured on it, and rewriting it would silently move them all.
"""
from __future__ import annotations

import pandas as pd

HISTDATA_ASSETS = frozenset({"USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDCHF", "XAUUSD"})


def true_utc(index: pd.DatetimeIndex, asset: str | None = None) -> pd.DatetimeIndex:
    """Stored stamps -> real UTC. Assets outside HISTDATA_ASSETS pass through.

    FX is shut from Friday 17:00 to Sunday 17:00 New York, so no bar falls on
    a daylight-saving switch (02:00 Sunday local) and the conversion is exact.
    """
    idx = pd.DatetimeIndex(index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    if asset is not None and asset not in HISTDATA_ASSETS:
        return idx
    wall = (idx.tz_localize(None) - pd.Timedelta(hours=5))
    ny = wall.tz_localize("America/New_York", ambiguous=True, nonexistent="shift_forward")
    return ny.tz_convert("UTC")


def with_true_utc(df: pd.DataFrame, asset: str | None = None) -> pd.DataFrame:
    """Copy of df re-indexed on real UTC (bars stay in order; nothing moves)."""
    out = df.copy()
    out.index = true_utc(df.index, asset)
    out.index.name = df.index.name
    return out
