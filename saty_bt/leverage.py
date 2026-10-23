"""
Leveraged and inverse funds the way they are really built (Kaufman, Trading
Systems and Methods 5th ed., Ch 2, p.56-57).

A 3x fund does not give you 3x the market. It resets to 3x EVERY DAY, so each
day's return is 3x that day's move, and those daily returns compound:

    XC_0 = 100
    XC_i = XC_(i-1) x ((C_i / C_(i-1) - 1) x L + 1)          (long, p.56)

Compounding L-times-bigger swings costs more than L-times the growth, because
the variance of the daily return grows with L^2: per period the log growth is
roughly  L x mu - L^2 x sigma^2 / 2  against the market's  mu - sigma^2 / 2.
The gap, (L^2 - L) x sigma^2 / 2 per period, is the volatility decay - see
volatility_decay(). On a path that goes nowhere (+10%, -9.09%) the market is
back where it started and the 2x fund is down 1.8%.

Inverse funds (p.57) invert the middle term - the price ratio:

    XC_i = XC_(i-1) x ((C_(i-1) / C_i - 1) x L + 1)          (inverse, "kaufman")

which makes a 1x inverse fund exactly the reciprocal of the price,
XC_i = 100 x C_0 / C_i (no decay at 1x). Real inverse ETFs (SH, SQQQ, ...)
instead target -L x the day's return,

    XC_i = XC_(i-1) x (1 - (C_i / C_(i-1) - 1) x L)          (inverse, "etf")

which decays even at 1x. The two differ by second-order terms
(C_(i-1)/C_i - 1 = -r + r^2 - ...); for a quiet index it is noise, for a
10%-a-day market it is not. Both are available; "kaufman" is the default
because it is the book's formula.

static_leverage() is the comparison: a position sized ONCE at L x the account
and never rebalanced - its value is the account plus L x the market's move,
so it delivers exactly L x the buy-and-hold return (until it is wiped out).

Both are floored at zero and zero is absorbing: a fund whose day loses 100%
(a -1/L move for a long fund) is gone, and a margin account whose equity
reaches zero is liquidated. No fees, financing or tracking error are charged
here - a real 3x fund also pays ~1%/yr in fees plus interest on the borrowed
2x, and a static margin position pays interest on what it borrowed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

INVERSE_STYLES = ("kaufman", "etf")


def _close(close) -> pd.Series:
    c = pd.to_numeric(pd.Series(close), errors="coerce").astype(float)
    if c.isna().any():
        raise ValueError("close has NaN values - drop or fill them first (a gap is not a 0% day)")
    if (c <= 0).any():
        raise ValueError("close must be positive")
    return c


def daily_growth(close, leverage: float, short: bool = False, inverse: str = "kaufman") -> pd.Series:
    """Per-period growth factor of a daily-reset fund, floored at 0 (first value 1)."""
    if leverage <= 0:
        raise ValueError("leverage must be > 0; use short=True for an inverse fund")
    if inverse not in INVERSE_STYLES:
        raise ValueError(f"inverse must be one of {INVERSE_STYLES}")
    c = _close(close)
    ratio = c / c.shift(1)
    if short:
        move = (1.0 / ratio - 1.0) if inverse == "kaufman" else -(ratio - 1.0)
    else:
        move = ratio - 1.0
    g = (move * float(leverage) + 1.0).clip(lower=0.0)
    if len(g):
        g.iloc[0] = 1.0
    return g


def daily_reset_index(close, leverage: float, short: bool = False, start: float = 100.0,
                      inverse: str = "kaufman") -> pd.Series:
    """
    Kaufman's leveraged / inverse fund index (p.56-57): starts at `start` (100)
    and resets to `leverage` x every period. short=True builds the inverse
    fund - by default the book's inverted price ratio (C_(i-1)/C_i), or the
    ETF convention -L x the period's return with inverse="etf". Floored at 0;
    once at 0 it stays there.
    """
    g = daily_growth(close, leverage, short, inverse)
    out = float(start) * np.cumprod(g.to_numpy(float))
    return pd.Series(out, index=g.index, name=f"{'-' if short else ''}{leverage:g}x_daily")


def static_leverage(close, leverage: float, start: float = 100.0, short: bool = False) -> pd.Series:
    """
    A position of `leverage` x the account bought ONCE at the first close and
    never resized: value = start x (1 + L x (C_i / C_0 - 1)) (short: minus the
    move). This is L x the buy-and-hold return. When the value reaches 0 the
    account is wiped out and stays at 0 (a broker liquidates it).
    """
    if leverage <= 0:
        raise ValueError("leverage must be > 0; use short=True for a short position")
    c = _close(close)
    move = c / c.iloc[0] - 1.0
    v = 1.0 + float(leverage) * (-move if short else move)
    v = v.to_numpy(float, copy=True)
    dead = np.flatnonzero(v <= 0)
    if len(dead):
        v[dead[0]:] = 0.0
    return pd.Series(float(start) * v, index=c.index, name=f"{'-' if short else ''}{leverage:g}x_static")


def volatility_decay(returns, leverage: float, periods_per_year: float) -> float:
    """
    Predicted yearly cost of resetting leverage every period, in LOG growth:
    (L^2 - L) x var(r) / 2 x periods_per_year. It is the gap between the
    daily-reset fund's log growth and L x the market's (to second order); at
    L = 1 it is zero. Fat tails make the realised gap differ from this
    estimate, so report both.
    """
    r = np.asarray(pd.to_numeric(pd.Series(returns), errors="coerce").dropna(), dtype=float)
    if len(r) < 2:
        return np.nan
    L = float(leverage)
    return (L * L - L) * float(r.var(ddof=1)) / 2.0 * periods_per_year


def max_drawdown(values) -> float:
    """Worst peak-to-trough fall of a value path (e.g. -0.55 = -55%)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return np.nan
    peak = np.maximum.accumulate(v)
    with np.errstate(invalid="ignore", divide="ignore"):
        dd = np.where(peak > 0, v / peak - 1.0, 0.0)
    return float(dd.min())


def cagr(start_value: float, end_value: float, years: float) -> float:
    """Compounded annual growth; -1 for an account that reached zero."""
    if years <= 0 or not math.isfinite(end_value):
        return np.nan
    if end_value <= 0:
        return -1.0
    return (end_value / start_value) ** (1.0 / years) - 1.0
