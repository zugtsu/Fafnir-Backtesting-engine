"""
Saty ATR Levels.

Port of Saty Mahajan's ATR Levels (Pine / thinkScript). The published script
anchors on the previous period close and the previous period's Wilder ATR:

    previous_close = security(tf, close[1])
    atr            = security(tf, ta.atr(atr_length)[1])
    call_trigger   = previous_close + trigger_pct * atr      (0.236)
    put_trigger    = previous_close - trigger_pct * atr
    + the fib ladder 0.382, 0.5, 0.618, 0.786, 1.0, 1.236, 1.618, 2.0, 2.618

Here the "period" bars are the level-period frame (daily for Day mode,
weekly for Multiday, monthly for Swing ...). For period k the grid uses
period k-1's close and ATR, both of which are complete before period k
opens - so no chart bar inside period k can see its own period's close.

Range utilisation ("Day Range is N% of ATR" in Saty's info box) is the
running high-low of the current period so far, over the period ATR,
computed only from chart bars up to and including the current one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.calendar import period_key
from .core import atr as _atr

LEVEL_NAMES = {
    0.236: "trigger", 0.382: "gg", 0.5: "mid", 0.618: "target",
    0.786: "l786", 1.0: "full", 1.236: "l1236", 1.618: "l1618",
    2.0: "l2000", 2.618: "l2618",
}


def period_grid(period_bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Level grid per period, indexed by `period_key`, built from the PREVIOUS
    period's close and ATR. Row k is the grid that is live during period k.
    """
    n = int(cfg.get("atr_length", 14))
    smoothing = cfg.get("smoothing", "wilder")
    fracs = [float(f) for f in cfg.get("fractions", list(LEVEL_NAMES))]

    p = period_bars.copy()
    a = _atr(p, n, smoothing)
    out = pd.DataFrame(index=p.index)
    out["period_key"] = p["period_key"].to_numpy() if "period_key" in p else p.index
    out["pdc"] = p["close"].shift(1)
    out["pdh"] = p["high"].shift(1)          # prior period high / low: the S/R map
    out["pdl"] = p["low"].shift(1)           # (same previous bar as pdc, so causal)
    # the daily trend, as of yesterday's close: above or below the 21-period EMA
    # of period closes (the ribbon's pivot line, on the level timeframe)
    d_ema = p["close"].ewm(span=21, adjust=False).mean()
    out["daily_trend"] = np.sign(p["close"] - d_ema).shift(1)
    out["atr"] = a.shift(1)
    for f in fracs:
        name = LEVEL_NAMES.get(f, f"l{int(round(f * 1000)):04d}")
        out[f"{name}_up"] = out["pdc"] + f * out["atr"]
        out[f"{name}_down"] = out["pdc"] - f * out["atr"]
    # Saty's names for the ones the strategy reads
    out["call_trigger"] = out["trigger_up"]
    out["put_trigger"] = out["trigger_down"]
    out["gg_up"], out["gg_down"] = out["gg_up"], out["gg_down"]
    out["mid_up"], out["mid_down"] = out["mid_up"], out["mid_down"]
    out["target_up"], out["target_down"] = out["target_up"], out["target_down"]
    out["full_up"], out["full_down"] = out["full_up"], out["full_down"]
    return out.set_index("period_key")


def atr_levels(chart: pd.DataFrame, period_bars: pd.DataFrame, period: str,
               cfg: dict) -> pd.DataFrame:
    """
    chart       : chart-timeframe bars with `day_key`
    period_bars : level-period bars (1d/1w/1M/3M/1Y frame from the store);
                  may come from a different series (cash index for SPX)
    period      : '1d' | '1w' | '1M' | '3M' | '1Y'
    Returns chart with pdc, atr, the level columns, and range_util added.
    """
    grid = period_grid(period_bars, cfg)
    pk = period_key(pd.DatetimeIndex(chart["day_key"]), period)
    out = chart.copy()
    out["period_key"] = pk
    cols = [c for c in grid.columns]
    joined = grid.reindex(pk)
    for c in cols:
        out[c] = joined[c].to_numpy()

    # range utilisation from completed chart bars inside the period
    g = out.groupby("period_key", sort=False)
    run_hi = g["high"].cummax()
    run_lo = g["low"].cummin()
    out["range_util"] = (run_hi - run_lo) / out["atr"]
    out["levels_valid"] = out["atr"].notna() & out["pdc"].notna()
    return out


def level_columns(cfg: dict) -> list[str]:
    fracs = [float(f) for f in cfg.get("fractions", list(LEVEL_NAMES))]
    names = [LEVEL_NAMES.get(f, f"l{int(round(f * 1000)):04d}") for f in fracs]
    return [f"{n}_{side}" for n in names for side in ("up", "down")]
