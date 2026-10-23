"""
Two textbook daily-bar systems, for the record and as a yardstick.

donchian : trend following. Long on a close above the highest high of the
           last `entry_n` days, exit on a close below the lowest low of the
           last `exit_n` days (or an ATR trailing stop); shorts mirror.
           Positions are held across days/months (flat_at_period_end: false).
           (Turtles: 20/10 and 55/20.)
rsi2     : Connors' short-term mean reversion. Long when close > SMA200 and
           RSI(2) < `rsi_buy`; exit at the next open once close > SMA(`exit_sma`).
           Shorts mirror below the SMA200 when `allow_shorts`.

Both expect the 1d timeframe. Stops are static ATR multiples (stop_atr_k)
from the entry level so the engine can size them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import finish_direction


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1.0 / n, adjust=False).mean()
    rs = au / ad.replace(0.0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100.0)


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    system = scfg.get("system", "donchian")
    k_stop = float(scfg.get("stop_atr_k", 2.0))
    close, high, low, atr = f["close"], f["high"], f["low"], f["atr"]
    sig = pd.DataFrame(index=f.index)
    if system == "donchian":
        n_in, n_out = int(scfg.get("entry_n", 20)), int(scfg.get("exit_n", 10))
        hh = high.rolling(n_in).max().shift(1)
        ll = low.rolling(n_in).min().shift(1)
        long_setup = (close > hh) & (close.shift(1) <= hh.shift(1))
        short_setup = (close < ll) & (close.shift(1) >= ll.shift(1))
        if not scfg.get("allow_shorts", True):
            short_setup = pd.Series(False, index=f.index)
        sig["long_exit"] = close < low.rolling(n_out).min().shift(1)
        sig["short_exit"] = close > high.rolling(n_out).max().shift(1)
        lvl_up, lvl_dn = hh, ll
    elif system == "rsi2":
        n_sma = int(scfg.get("trend_sma", 200))
        buy = float(scfg.get("rsi_buy", 10.0))
        ex = int(scfg.get("exit_sma", 5))
        rsi = _rsi(close, int(scfg.get("rsi_n", 2)))
        if n_sma > 0:
            sma = close.rolling(n_sma).mean()
            above, below = close > sma, close < sma
        else:                                   # no trend filter
            above = below = pd.Series(True, index=f.index)
        long_setup = above & (rsi < buy)
        short_setup = below & (rsi > 100 - buy) if scfg.get("allow_shorts", False) else pd.Series(False, index=f.index)
        sig["long_exit"] = close > close.rolling(ex).mean()
        sig["short_exit"] = close < close.rolling(ex).mean()
        lvl_up = lvl_dn = close
    else:
        raise ValueError(f"classics system {system}")

    long_setup &= f["levels_valid"] if "levels_valid" in f else True
    short_setup &= f["levels_valid"] if "levels_valid" in f else True
    sig["entry_level_up"], sig["entry_level_down"] = lvl_up, lvl_dn
    finish_direction(sig, f, 1, long_setup.fillna(False), scfg, acfg, cfg, gates_enabled)
    finish_direction(sig, f, -1, short_setup.fillna(False), scfg, acfg, cfg, gates_enabled)
    sig["stop_mode"] = "atr"
    sig["long_stop"], sig["short_stop"] = close - k_stop * atr, close + k_stop * atr
    far = 99.0 * atr
    sig["long_target"], sig["short_target"] = close + far, close - far
    sig["long_runner"], sig["short_runner"] = sig["long_target"], sig["short_target"]
    sig["long_trigger"], sig["short_trigger"] = sig["long_stop"], sig["short_stop"]
    return sig
