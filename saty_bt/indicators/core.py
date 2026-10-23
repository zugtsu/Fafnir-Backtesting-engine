"""
Primitives shared by the four Saty indicators. Every function is a pure
Series/DataFrame transform computed from completed bars only: the value on
row t uses rows <= t and nothing later.

EMA and Wilder RMA follow TradingView semantics so plots can be spot-checked:
    ta.ema: seeded with the first value, alpha = 2/(n+1)
    ta.rma: seeded with the SMA of the first n values, alpha = 1/n
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rma(s: pd.Series, n: int) -> pd.Series:
    """Wilder's moving average with TradingView's SMA seed."""
    vals = s.to_numpy(dtype="float64")
    out = np.full(len(vals), np.nan)
    if len(vals) < n:
        return pd.Series(out, index=s.index)
    # first index where n consecutive non-NaN values exist
    ok = ~np.isnan(vals)
    start = None
    run = 0
    for i, flag in enumerate(ok):
        run = run + 1 if flag else 0
        if run == n:
            start = i
            break
    if start is None:
        return pd.Series(out, index=s.index)
    out[start] = vals[start - n + 1:start + 1].mean()
    alpha = 1.0 / n
    prev = out[start]
    for i in range(start + 1, len(vals)):
        v = vals[i]
        if np.isnan(v):
            out[i] = prev
            continue
        prev = alpha * v + (1 - alpha) * prev
        out[i] = prev
    return pd.Series(out, index=s.index)


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - pc).abs(),
                    (df["low"] - pc).abs()], axis=1).max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, n: int = 14, smoothing: str = "wilder") -> pd.Series:
    tr = true_range(df)
    if smoothing == "wilder":
        return rma(tr, n)
    if smoothing == "sma":
        return sma(tr, n)
    if smoothing == "ema":
        return ema(tr, n)
    raise ValueError(smoothing)


def stdev(s: pd.Series, n: int) -> pd.Series:
    """Population stdev, as TradingView's ta.stdev."""
    return s.rolling(n, min_periods=n).std(ddof=0)


def cross_above(s: pd.Series, level: float | pd.Series) -> pd.Series:
    prev = s.shift(1)
    lv_prev = level.shift(1) if isinstance(level, pd.Series) else level
    return (s > level) & (prev <= lv_prev)


def cross_below(s: pd.Series, level: float | pd.Series) -> pd.Series:
    prev = s.shift(1)
    lv_prev = level.shift(1) if isinstance(level, pd.Series) else level
    return (s < level) & (prev >= lv_prev)


def swing_lows(df: pd.DataFrame, n: int = 2) -> pd.Series:
    """
    Most recent CONFIRMED swing low (n bars either side), known only n bars
    after the pivot bar. Forward-filled so row t carries the last pivot that
    was confirmed at or before t.
    """
    low = df["low"]
    is_piv = pd.Series(True, index=df.index)
    for k in range(1, n + 1):
        is_piv &= (low < low.shift(k)) & (low < low.shift(-k))
    piv = low.where(is_piv).shift(n)          # confirmed n bars later
    return piv.ffill()


def swing_highs(df: pd.DataFrame, n: int = 2) -> pd.Series:
    high = df["high"]
    is_piv = pd.Series(True, index=df.index)
    for k in range(1, n + 1):
        is_piv &= (high > high.shift(k)) & (high > high.shift(-k))
    piv = high.where(is_piv).shift(n)
    return piv.ffill()
