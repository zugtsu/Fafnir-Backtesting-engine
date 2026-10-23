"""
ADX (Wilder) and the Ehlers Fisher Transform, chart-timeframe, completed bars.

ADX, as TradingView's built-in (DI length, ADX smoothing):
    +DM = high - high[1] if that exceeds low[1] - low and is > 0 else 0
    -DM = low[1] - low  if that exceeds high - high[1] and is > 0 else 0
    TR  = max(high - low, |high - close[1]|, |low - close[1]|)
    smoothed with Wilder's RMA over `di_length`
    +DI = 100 * RMA(+DM) / RMA(TR), -DI likewise
    DX  = 100 * |+DI - -DI| / (+DI + -DI)
    ADX = RMA(DX, adx_length)

Fisher (HPotter's Pine, length 10 on hl2):
    v1 = 0.33 * 2 * ((hl2 - min) / (max - min) - 0.5) + 0.67 * v1[1]
    v2 = clamp(v1, -0.999, 0.999)
    fish = 0.5 * ln((1 + v2) / (1 - v2)) + 0.5 * fish[1]
    trigger = fish[1]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import rma


def adx(df: pd.DataFrame, di_length: int = 14, adx_length: int = 14) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    up = h - h.shift(1)
    dn = l.shift(1) - l
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr_w = rma(tr, di_length)
    pdi = 100 * rma(pd.Series(plus_dm, index=df.index), di_length) / atr_w
    mdi = 100 * rma(pd.Series(minus_dm, index=df.index), di_length) / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["plus_di"], out["minus_di"] = pdi, mdi
    out["adx"] = rma(dx, adx_length)
    out["adx_rising"] = out["adx"] > out["adx"].shift(1)
    return out


def fisher(df: pd.DataFrame, length: int = 10) -> pd.DataFrame:
    hl2 = (df["high"] + df["low"]) / 2.0
    mx = hl2.rolling(length, min_periods=length).max()
    mn = hl2.rolling(length, min_periods=length).min()
    rng = (mx - mn).replace(0, np.nan)
    x = (hl2 - mn) / rng - 0.5
    n = len(df)
    v1 = np.zeros(n)
    fish = np.zeros(n)
    xv = x.to_numpy()
    for i in range(n):
        xi = xv[i]
        prev_v1 = v1[i - 1] if i > 0 else 0.0
        prev_f = fish[i - 1] if i > 0 else 0.0
        if np.isnan(xi):
            v1[i] = prev_v1
            fish[i] = prev_f
            continue
        v = 0.33 * 2 * xi + 0.67 * prev_v1
        v = min(max(v, -0.999), 0.999)
        v1[i] = v
        fish[i] = 0.5 * np.log((1 + v) / (1 - v)) + 0.5 * prev_f
    out = pd.DataFrame(index=df.index)
    out["fisher"] = fish
    out["fisher_trigger"] = out["fisher"].shift(1)
    out["fisher_warm"] = x.notna() & (np.arange(n) >= 3 * length)
    return out


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD as TradingView's built-in: EMA(fast) - EMA(slow), signal = EMA of that."""
    c = df["close"]
    line = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    out = pd.DataFrame(index=df.index)
    out["macd"] = line
    out["macd_signal"] = sig
    out["macd_hist"] = line - sig
    out["macd_warm"] = np.arange(len(df)) >= slow + signal
    return out


def rsi(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Wilder's RSI, as TradingView's built-in."""
    d = df["close"].diff()
    au = d.clip(lower=0.0).ewm(alpha=1.0 / length, adjust=False).mean()
    ad = (-d).clip(lower=0.0).ewm(alpha=1.0 / length, adjust=False).mean()
    rs = au / ad.replace(0.0, np.nan)
    out = pd.DataFrame(index=df.index)
    out["rsi"] = (100 - 100 / (1 + rs)).fillna(100.0)
    out["rsi_warm"] = np.arange(len(df)) >= 3 * length
    return out
