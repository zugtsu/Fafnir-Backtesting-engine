"""
Saty Pivot Ribbon.

Default EMA set is the thinkorswim one (8/13/21/48/200); TradingView's
saty_pivot_ribbon.pine uses 8/21/34 for the ribbon and 13/48 for the
conviction arrows. All configurable.

Columns added (chart timeframe):
    ema_<n> for each n
    ribbon_bull  : fast > pivot > slow
    ribbon_bear  : fast < pivot < slow
    ribbon_mixed : neither
    ribbon_state : 'bull' | 'bear' | 'mixed'
    conviction_bull / conviction_bear : EMA13 vs EMA48
    conviction_cross_up / _down       : state-change bar only (the arrows)
    bias_pivot   : close > pivot EMA (21)   bias_long : close > EMA200
    ribbon_spread_atr : (max(fast,pivot,slow) - min) / ATR(14, chart)
    ribbon_flat  : spread < flat_spread_atr (entangled ribbon, avoid)

Time Warp: `time_warp()` computes the same ribbon on a higher timeframe
and carries each HTF bar's state onto the chart bars that close at or
after that HTF bar's close. A chart bar never sees an HTF bar that has not
finished yet.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import atr, ema


def pivot_ribbon(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    fast, pivot, slow = int(cfg["fast"]), int(cfg["pivot"]), int(cfg["slow"])
    cf, cs = int(cfg["conviction_fast"]), int(cfg["conviction_slow"])
    bias_n = int(cfg.get("bias_long", 200))
    lens = sorted(set([int(n) for n in cfg.get("emas", [])] + [fast, pivot, slow, cf, cs, bias_n]))

    out = df.copy()
    for n in lens:
        out[f"ema_{n}"] = ema(df["close"], n)
    f, p, s = out[f"ema_{fast}"], out[f"ema_{pivot}"], out[f"ema_{slow}"]
    out["ribbon_bull"] = (f > p) & (p > s)
    out["ribbon_bear"] = (f < p) & (p < s)
    out["ribbon_mixed"] = ~(out["ribbon_bull"] | out["ribbon_bear"])
    out["ribbon_state"] = np.select([out["ribbon_bull"], out["ribbon_bear"]], ["bull", "bear"], "mixed")

    cb = out[f"ema_{cf}"] > out[f"ema_{cs}"]
    out["conviction_bull"] = cb
    out["conviction_bear"] = out[f"ema_{cf}"] < out[f"ema_{cs}"]
    out["conviction_cross_up"] = cb & ~cb.shift(1, fill_value=False)
    out["conviction_cross_down"] = ~cb & cb.shift(1, fill_value=True)
    out["conviction_spread"] = out[f"ema_{cf}"] - out[f"ema_{cs}"]
    out["conviction_widening"] = out["conviction_spread"].abs() > out["conviction_spread"].abs().shift(1)

    out["bias_pivot"] = df["close"] > p
    out["bias_long"] = df["close"] > out[f"ema_{bias_n}"]

    a = atr(df, int(cfg.get("flat_atr_length", 14)))
    spread = pd.concat([f, p, s], axis=1).max(axis=1) - pd.concat([f, p, s], axis=1).min(axis=1)
    out["ribbon_spread_atr"] = spread / a
    out["ribbon_flat"] = out["ribbon_spread_atr"] < float(cfg.get("flat_spread_atr", 0.15))
    # warm-up: the slowest EMA needs ~3n bars to be meaningful; flag it
    out["ribbon_warm"] = np.arange(len(out)) >= 3 * max(lens)
    return out


def time_warp(chart: pd.DataFrame, htf: pd.DataFrame, cfg: dict,
              prefix: str = "htf_") -> pd.DataFrame:
    """
    Ribbon on `htf` bars, carried onto `chart` without look-ahead.

    chart must have `bar_close`; htf must have `bar_close`. A chart bar
    closing at time T takes the last HTF bar whose close <= T.
    """
    rib = pivot_ribbon(htf[["open", "high", "low", "close", "volume"]], cfg)
    keep = ["ribbon_state", "ribbon_bull", "ribbon_bear", "conviction_bull", "bias_long", "ribbon_flat"]
    right = pd.DataFrame({c: rib[c].to_numpy() for c in keep})
    right["htf_close_time"] = pd.DatetimeIndex(htf["bar_close"]).tz_convert("UTC")
    right["htf_bar_time"] = htf.index
    right = right.sort_values("htf_close_time")

    left = pd.DataFrame({"chart_close_time": pd.DatetimeIndex(chart["bar_close"]).tz_convert("UTC"),
                         "_i": np.arange(len(chart))}).sort_values("chart_close_time")
    merged = pd.merge_asof(left, right, left_on="chart_close_time", right_on="htf_close_time",
                           direction="backward", allow_exact_matches=True)
    merged = merged.sort_values("_i")
    out = chart.copy()
    for c in keep:
        out[f"{prefix}{c}"] = merged[c].to_numpy()
    out[f"{prefix}bar_time"] = merged["htf_bar_time"].to_numpy()
    for c in ("ribbon_bull", "ribbon_bear", "conviction_bull", "bias_long", "ribbon_flat"):
        out[f"{prefix}{c}"] = out[f"{prefix}{c}"].astype("boolean").fillna(False).astype(bool)
    out[f"{prefix}ribbon_state"] = out[f"{prefix}ribbon_state"].fillna("mixed")
    return out
