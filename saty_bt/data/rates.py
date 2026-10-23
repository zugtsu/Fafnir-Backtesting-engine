"""
Overnight FX financing from central-bank rates.

OANDA (like every spot FX broker) credits or debits the interest-rate
differential between a pair's two currencies on every position held through
the 17:00 New York rollover, less its own markup. Until 2026-10-06 the engine
modelled this only as a flat ~0.5%/yr markup charged both ways, which ignores
the differential itself - e.g. ~4-5%/yr earned on a long USDJPY in 2023-2025.

    costs:
      financing: rate_diff           # opt-in; default is the old flat model
      financing_markup_pct: 1.0      # charged against the trader, per year
      financing_weekend: triple_wed  # Wednesday's roll carries the weekend
      financing_day_count: 365

Per rollover held, in USD:
    notional_usd x weight x (direction x (r_base - r_quote) - markup) / 100 / day_count
where r is the policy rate (percent) in force at that rollover - the last
decision ANNOUNCED strictly before it - from policy_rates.csv. A position is
charged for every 17:00 New York instant R with entry < R <= exit.

The table is central-bank policy rates, a proxy for the interbank overnight
benchmarks brokers actually use; differences are a few basis points a year.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .clock import true_utc

RATES_CSV = Path(__file__).with_name("policy_rates.csv")
NY = "America/New_York"
WEIGHTS = {"triple_wed": {2: 3}, "triple_fri": {4: 3}, "flat": {}}


def parse_table(d: pd.DataFrame) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    out = {}
    for ccy, g in d.groupby("ccy"):
        ts = pd.DatetimeIndex([pd.Timestamp(f"{r.announced_date} {r.announced_time_local}")
                               .tz_localize(r.tz).tz_convert("UTC") for r in g.itertuples()])
        order = np.argsort(ts.to_numpy())
        out[str(ccy)] = (ts[order], g.rate_pct.to_numpy(float)[order])
    return out


@lru_cache(maxsize=4)
def load_table(path: str | None = None) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    return parse_table(pd.read_csv(path or RATES_CSV))


def rate_at(ccy: str, when_utc: pd.DatetimeIndex, table: dict | None = None) -> np.ndarray:
    """Policy rate (percent) in force at each instant: last announcement strictly before it."""
    table = table if table is not None else load_table()
    if ccy not in table:
        raise KeyError(f"no policy-rate history for {ccy}")
    idx, vals = table[ccy]
    pos = idx.searchsorted(pd.DatetimeIndex(when_utc), side="left") - 1
    if len(pos) and (pos < 0).any():
        raise ValueError(f"{ccy}: no rate before {pd.DatetimeIndex(when_utc)[pos < 0][0]}")
    return vals[pos]


def rollovers(entry_utc: pd.Timestamp, exit_utc: pd.Timestamp,
              weekend: str = "triple_wed") -> tuple[pd.DatetimeIndex, np.ndarray]:
    """17:00 New York rollovers R with entry < R <= exit, and their day weights."""
    a, b = pd.Timestamp(entry_utc).tz_convert(NY), pd.Timestamp(exit_utc).tz_convert(NY)
    if b <= a:
        return pd.DatetimeIndex([], tz="UTC"), np.array([])
    days = pd.date_range(a.normalize().tz_localize(None) - pd.Timedelta(days=1),
                         b.normalize().tz_localize(None) + pd.Timedelta(days=1), freq="D")
    days = days[days.dayofweek < 5]
    inst = pd.DatetimeIndex([pd.Timestamp(f"{d:%Y-%m-%d} 17:00").tz_localize(NY) for d in days])
    keep = (inst > a) & (inst <= b)
    inst = inst[keep]
    w = WEIGHTS[weekend]
    weights = np.array([w.get(int(d), 1) for d in inst.dayofweek], float)
    return inst.tz_convert("UTC"), weights


def financing_usd(notional_usd: float, base: str, quote: str, direction: int,
                  entry: pd.Timestamp, exit: pd.Timestamp, markup_pct: float,
                  asset: str | None = None, weekend: str = "triple_wed",
                  day_count: int = 365, table: dict | None = None) -> float:
    """Signed financing over the holding period (positive = credited to the trader).
    entry/exit are the asset's STORED stamps; for a HistData asset they are
    converted to real UTC first (clock.true_utc), so the rollover lands on
    17:00 New York. asset=None means the stamps are already real UTC."""
    stamps = pd.DatetimeIndex([entry, exit])
    e, x = true_utc(stamps, asset) if asset is not None else stamps.tz_convert("UTC")
    inst, w = rollovers(e, x, weekend)
    if not len(inst):
        return 0.0
    diff = rate_at(base, inst, table) - rate_at(quote, inst, table)
    per = direction * diff - markup_pct
    return float(notional_usd * np.sum(w * per) / 100.0 / day_count)
