"""
Shared harness for the October 2026 edge hunt (EDGE_PREREG.md).

Calendar and order-flow hypotheses are "be in the market from clock time A to
clock time B" trades, which the bar engine does not express. Every such test
goes through window_trades() so the fill and cost arithmetic is written once,
tested once, and identical across hypotheses.

Conventions (same as the engine):
  * bars must be indexed on REAL UTC - pass HistData frames through
    saty_bt.data.clock.with_true_utc first; bars are stamped at bar OPEN.
  * a trade scheduled for time T fills at the OPEN of the first bar starting
    at or after T. An exit at T therefore realises the move up to T and
    nothing after it. If that bar starts more than `max_lag` after T (a
    holiday, a data gap), the trade is dropped and counted, never shifted.
  * costs: the full venue round-trip is charged on every trade - half the
    spread at each fill plus any per-side fee - and holding cost per hour.
  * pnl_usd is at a fixed notional, so trades are comparable across years;
    `ret` is the net fractional return; r_multiple is filled only when the
    caller supplies a stop distance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

NOTIONAL = 10_000.0


@dataclass(frozen=True)
class Venue:
    name: str
    spread: dict            # price units, full spread, per asset (charged half per fill)
    spread_frac: float = 0.0  # OR full spread as a fraction of price (used when asset not in `spread`)
    fee_side: float = 0.0     # fraction of notional per fill
    hold_bps_hour: float = 0.0

    def round_trip_frac(self, asset: str, price: float, mult: float = 1.0) -> float:
        sp = self.spread.get(asset)
        s = (sp / price) if sp is not None else self.spread_frac
        return mult * (s + 2 * self.fee_side)

    def hold_frac(self, hours: float, mult: float = 1.0) -> float:
        return mult * self.hold_bps_hour * hours / 1e4


# OANDA US retail, spread-only (scripts/fx_hunt.py); financing markup ~0.5%/yr
OANDA = Venue("oanda", spread={"USDJPY": 0.014, "AUDUSD": 0.00014, "EURUSD": 0.00010,
                               "GBPUSD": 0.00014, "USDCAD": 0.00020, "USDCHF": 0.00018,
                               "XAUUSD": 0.35},
              hold_bps_hour=0.14 / 24)
# gTrade perps (memory: venue docs 2026-09-14); gas ignored at $10k notional
GTRADE_BTC = Venue("gtrade_btc", spread={}, spread_frac=0.00005, fee_side=0.00035, hold_bps_hour=0.64)
GTRADE_SPY = Venue("gtrade_spy", spread={}, spread_frac=0.000216, fee_side=0.0005)
GTRADE_FX = Venue("gtrade_fx", spread={}, spread_frac=0.00005, fee_side=0.00012)
# spot BTC on a US exchange, taker at the lowest tier - an ASSUMPTION
SPOT_BTC = Venue("spot_btc", spread={}, spread_frac=0.0002, fee_side=0.0040)
# S&P 500 index CFD / futures-like cost, for reference only (the user has no such account)
INDEX_CHEAP = Venue("index_cheap", spread={}, spread_frac=0.0001)


def _fill_idx(index: pd.DatetimeIndex, when: pd.DatetimeIndex, max_lag: pd.Timedelta) -> np.ndarray:
    # Timedelta arithmetic, never .asi8: pandas 3 defaults to microsecond
    # resolution and parquet often loads as nanoseconds, so raw integers from
    # two indexes need not share a unit.
    pos = index.searchsorted(when, side="left")
    ok = pos < len(index)
    lag = np.full(len(when), np.inf)
    if ok.any():
        lag[ok] = (index[pos[ok]] - when[ok]).total_seconds()
    pos = np.where(ok & (lag <= max_lag.total_seconds()), pos, -1)
    return pos


def window_trades(bars: pd.DataFrame, schedule: pd.DataFrame, asset: str, venue: Venue,
                  *, cost_mult: float = 1.0, max_lag: str = "30min",
                  notional: float = NOTIONAL) -> pd.DataFrame:
    """
    schedule: entry_time, exit_time (UTC), direction (+1/-1), optional extra
    columns (carried through, e.g. `cfg`, `label`). Returns one row per filled
    trade: entry_time, exit_time (the actual fill stamps), asset, direction,
    entry_price, exit_price, gross, ret, pnl_usd, hours, plus the extras.
    `.attrs['dropped']` counts schedule rows that could not be filled.
    """
    if bars.index.tz is None:
        raise ValueError("bars must be tz-aware real UTC (use clock.with_true_utc)")
    s = schedule.reset_index(drop=True).copy()
    if not len(s):
        out = pd.DataFrame(columns=["entry_time", "exit_time", "asset", "direction", "ret", "pnl_usd"])
        out.attrs["dropped"] = 0
        return out
    et = pd.DatetimeIndex(pd.to_datetime(s["entry_time"], utc=True))
    xt = pd.DatetimeIndex(pd.to_datetime(s["exit_time"], utc=True))
    if (xt <= et).any():
        raise ValueError("exit_time must be after entry_time")
    lag = pd.Timedelta(max_lag)
    ie, ix = _fill_idx(bars.index, et, lag), _fill_idx(bars.index, xt, lag)
    o = pd.to_numeric(bars["open"], errors="coerce").to_numpy(float)
    keep = (ie >= 0) & (ix > ie)
    s, ie, ix = s[keep].reset_index(drop=True), ie[keep], ix[keep]
    pe, px = o[ie], o[ix]
    keep2 = np.isfinite(pe) & np.isfinite(px) & (pe > 0)
    s, ie, ix, pe, px = s[keep2].reset_index(drop=True), ie[keep2], ix[keep2], pe[keep2], px[keep2]
    d = s["direction"].to_numpy(float)
    t_in, t_out = bars.index[ie], bars.index[ix]
    hours = np.asarray((t_out - t_in).total_seconds(), float) / 3600.0
    gross = d * (px - pe) / pe
    rt = np.array([venue.round_trip_frac(asset, p, cost_mult) for p in pe])
    ret = gross - rt - venue.hold_frac(1.0, cost_mult) * hours
    out = s.drop(columns=["entry_time", "exit_time"]).assign(
        entry_time=t_in, exit_time=t_out, asset=asset, entry_price=pe, exit_price=px,
        gross=gross, ret=ret, pnl_usd=ret * notional, hours=hours)
    out.attrs["dropped"] = int(len(schedule) - len(out))
    return out


def pf(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    neg = x[x < 0].sum()
    return float(x[x > 0].sum() / -neg) if neg < 0 else np.nan


def tstat(x) -> float:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    return float(x.mean() / x.std() * np.sqrt(len(x))) if len(x) > 2 and x.std() > 0 else np.nan


def summary(t: pd.DataFrame, split, col: str = "pnl_usd") -> dict:
    """Headline numbers for one cell, both regimes, in the unit `col`."""
    t = t.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    sp = pd.Timestamp(split)
    sp = sp.tz_localize("UTC") if sp.tz is None else sp
    a, b = t[t.exit_time < sp], t[t.exit_time >= sp]
    return {"trades": len(t), "pf": pf(t[col]), "t": tstat(t[col]),
            "mean_bp": float(t["ret"].mean() * 1e4) if "ret" in t and len(t) else np.nan,
            "gross_bp": float(t["gross"].mean() * 1e4) if "gross" in t and len(t) else np.nan,
            "win": float((t[col] > 0).mean()) if len(t) else np.nan,
            "total": float(t[col].sum()), "n_old": len(a), "n_new": len(b),
            "pf_old": pf(a[col]), "pf_new": pf(b[col])}


def default_split(t: pd.DataFrame) -> pd.Timestamp:
    """2021-01-01 where the market has a 2015-2020 history; otherwise its midpoint."""
    x = pd.to_datetime(t["exit_time"], utc=True)
    if x.min().year < 2020:
        return pd.Timestamp("2021-01-01", tz="UTC")
    return x.min() + (x.max() - x.min()) / 2
