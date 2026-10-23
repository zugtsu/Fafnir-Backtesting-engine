"""
Strategy V - the "Vomy" (vomiting dolphin), Saty's ribbon reversal.

From his video: a clean stacked trend (8 > 13 > 21 > 48 for longs) runs
into resistance, breaks the 8/13, tests the 48, then RETESTS the 13 from
below. That retest is the early entry (risk: a close back above the 13).
The move is confirmed by a close below the 48; the confirmed entry is the
first close below the 48. Targets are the ATR levels below (previous
close, put trigger, ...). The inverse Vomy mirrors for longs.

mode:
    retest13 : short on the bar that pokes the 13 from below and closes
               under it, after the 48 has been tested (early entry)
    break48  : short on the first close below the 48 (confirmed entry)
trend_bars : how many consecutive stacked bars count as "clear trend"
lookback   : the trend run and the 48 test must be within this many bars
stop:
    ema_close : confirmed close back through the 13 (retest13) or the
                21 (break48); dynamic
    swing     : the last confirmed swing high/low, intrabar, static
Target/runner: the next two ATR-ladder levels in the trade direction
(half off at the first, stop to breakeven, runner to the second).
"""

from __future__ import annotations

import pandas as pd

from .base import finish_direction, next_levels


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    mode = scfg.get("mode", "retest13")
    tb = int(scfg.get("trend_bars", 12))
    look = int(scfg.get("lookback", 20))
    stop_mode = scfg.get("stop", "ema_close")
    e8, e13, e21, e48 = f["ema_8"], f["ema_13"], f["ema_21"], f["ema_48"]
    hi, lo, close = f["high"], f["low"], f["close"]

    stacked_bull = (e8 > e13) & (e13 > e21) & (e21 > e48) & f["ribbon_warm"]
    stacked_bear = (e8 < e13) & (e13 < e21) & (e21 < e48) & f["ribbon_warm"]
    run_bull = stacked_bull.astype(float).rolling(tb, min_periods=tb).min() > 0     # tb stacked bars in a row
    run_bear = stacked_bear.astype(float).rolling(tb, min_periods=tb).min() > 0
    trend_bull = run_bull.shift(1).astype(float).rolling(look, min_periods=1).max() > 0   # such a run ended within `look`
    trend_bear = run_bear.shift(1).astype(float).rolling(look, min_periods=1).max() > 0

    # the 48 has been tested (bull case: low reached the 48) within the window, after the trend
    tested48_bull = (lo <= e48).shift(1).astype(float).rolling(look, min_periods=1).max() > 0
    tested48_bear = (hi >= e48).shift(1).astype(float).rolling(look, min_periods=1).max() > 0

    if mode == "retest13":
        # bull trend gone sick: bar pokes up into the 13 and closes back under it; 48 not broken yet
        short_setup = trend_bull & tested48_bull & (hi >= e13) & (close < e13) & (close.shift(1) < e13) & (close > e48)
        long_setup = trend_bear & tested48_bear & (lo <= e13) & (close > e13) & (close.shift(1) > e13) & (close < e48)
        stop_ema = e13
    elif mode == "break48":
        short_setup = trend_bull & (close < e48) & (close.shift(1) >= e48)
        long_setup = trend_bear & (close > e48) & (close.shift(1) <= e48)
        stop_ema = e21
    else:
        raise ValueError(f"vomy mode {mode}")

    sig = pd.DataFrame(index=f.index)
    finish_direction(sig, f, 1, long_setup, scfg, acfg, cfg, gates_enabled)
    finish_direction(sig, f, -1, short_setup, scfg, acfg, cfg, gates_enabled)

    t_l, r_l = next_levels(f, 1)
    t_s, r_s = next_levels(f, -1)
    sig["long_target"], sig["short_target"] = t_l, t_s
    sig["long_runner"], sig["short_runner"] = r_l, r_s
    if stop_mode == "ema_close":
        sig["stop_mode"] = "close_through"
        sig["long_stop"], sig["short_stop"] = stop_ema, stop_ema
    elif stop_mode == "swing":
        sig["stop_mode"] = "swing"
        sig["long_stop"], sig["short_stop"] = f["swing_low"], f["swing_high"]
    else:
        raise ValueError(f"vomy stop {stop_mode}")
    sig["long_trigger"], sig["short_trigger"] = sig["long_stop"], sig["short_stop"]
    return sig
