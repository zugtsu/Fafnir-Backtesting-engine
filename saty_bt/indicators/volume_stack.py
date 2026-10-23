"""
Saty Volume Stack.

    buy_pct  = (close - low) / (high - low)     (50/50 when high == low)
    sell_pct = (high - close) / (high - low)
    buy_vol  = volume * buy_pct,  sell_vol = volume * sell_pct
    stack_winner : 'buy' | 'sell' | 'tie'   (buy_vol vs sell_vol - which is
                   the same comparison as buy_pct vs sell_pct, so the winner
                   is defined even when the feed carries no volume)
    stack_ratio  : winner's share of the bar
    rel_vol      : volume / SMA(volume, 20)   (NaN without volume)
    stack_streak : consecutive bars won by the same side (signed: +buy, -sell)
    vol_available: the volume column carries real numbers on this bar
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import sma


def volume_stack(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    n = int(cfg.get("rel_vol_length", 20))
    out = df.copy()
    rng = df["high"] - df["low"]
    flat = rng <= 0
    buy_pct = ((df["close"] - df["low"]) / rng).where(~flat, 0.5)
    sell_pct = ((df["high"] - df["close"]) / rng).where(~flat, 0.5)
    out["buy_pct"], out["sell_pct"] = buy_pct, sell_pct
    vol = df["volume"]
    out["buy_vol"] = vol * buy_pct
    out["sell_vol"] = vol * sell_pct
    winner = np.select([buy_pct > sell_pct, sell_pct > buy_pct], ["buy", "sell"], "tie")
    out["stack_winner"] = winner
    out["stack_ratio"] = np.maximum(buy_pct, sell_pct)
    out["rel_vol"] = vol / sma(vol, n)
    out["vol_available"] = vol.notna()
    out["vol_conviction"] = out["rel_vol"] >= float(cfg.get("conviction_mult", 1.5))

    sign = pd.Series(np.select([winner == "buy", winner == "sell"], [1, -1], 0), index=df.index)
    grp = (sign != sign.shift(1)).cumsum()
    streak = sign.groupby(grp).cumcount() + 1
    out["stack_streak"] = (streak * sign).astype(int)
    return out
