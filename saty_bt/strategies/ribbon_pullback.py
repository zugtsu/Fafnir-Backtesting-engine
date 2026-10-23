"""
Strategy B - Ribbon Pullback (trend continuation).

Regime (long): ribbon_bull and conviction_bull (EMA13 > EMA48) and close
above EMA200, and the Time Warp ribbon agrees. Mirror for shorts.
Setup: within the last `pullback_lookback` bars price touched or closed
below the pullback EMA (13 by default, 21 optional) while every close
stayed above EMA48.
Entry: the reclaim - a confirmed close back above the pullback EMA, with a
buy-winning Volume Stack bar (the volume gate).
Target: the next ATR level beyond the close (PDC/trigger/38.2/61.8/100 ...),
runner to the one after; half off at the target, stop to breakeven.
Trail: confirmed close through EMA `trail_ema` (13 or 21) exits the runner.
Stop: 'close_through' EMA48 (confirmed close) or 'swing' (intrabar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import finish_direction, next_levels


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    pb = int(scfg.get("pullback_ema", 13))
    trail = int(scfg.get("trail_ema", 13))
    look = int(scfg.get("pullback_lookback", 5))
    e_pb, e48, e200 = f[f"ema_{pb}"], f["ema_48"], f["ema_200"]
    e_tr = f[f"ema_{trail}"]
    htf_l = f["htf_ribbon_bull"] if "htf_ribbon_bull" in f else True
    htf_s = f["htf_ribbon_bear"] if "htf_ribbon_bear" in f else True
    use_htf = bool(scfg.get("require_htf", True))

    regime_l = f["ribbon_bull"] & f["conviction_bull"] & f["bias_long"] & f["ribbon_warm"]
    regime_s = f["ribbon_bear"] & f["conviction_bear"] & ~f["bias_long"] & f["ribbon_warm"]
    if use_htf:
        regime_l &= htf_l
        regime_s &= htf_s

    touched_l = (f["low"] <= e_pb).astype(float)
    touched_s = (f["high"] >= e_pb).astype(float)
    pulled_l = touched_l.shift(1).rolling(look, min_periods=1).max() > 0
    pulled_s = touched_s.shift(1).rolling(look, min_periods=1).max() > 0
    held_l = (f["close"] > e48).astype(float).rolling(look, min_periods=1).min() > 0
    held_s = (f["close"] < e48).astype(float).rolling(look, min_periods=1).min() > 0
    reclaim_l = (f["close"] > e_pb) & (f["close"].shift(1) <= e_pb)
    reclaim_s = (f["close"] < e_pb) & (f["close"].shift(1) >= e_pb)

    sig = pd.DataFrame(index=f.index)
    finish_direction(sig, f, 1, regime_l & pulled_l & held_l & reclaim_l, scfg, acfg, cfg, gates_enabled)
    finish_direction(sig, f, -1, regime_s & pulled_s & held_s & reclaim_s, scfg, acfg, cfg, gates_enabled)

    t_l, r_l = next_levels(f, 1)
    t_s, r_s = next_levels(f, -1)
    sig["long_target"], sig["short_target"] = t_l, t_s
    sig["long_runner"], sig["short_runner"] = r_l, r_s
    sig["long_trail"], sig["short_trail"] = e_tr, e_tr
    stop_mode = scfg.get("stop", "close_through")
    sig["stop_mode"] = stop_mode
    if stop_mode == "close_through":
        sig["long_stop"], sig["short_stop"] = e48, e48
    elif stop_mode == "swing":
        sig["long_stop"], sig["short_stop"] = f["swing_low"], f["swing_high"]
    else:
        raise ValueError(f"ribbon_pullback: unknown stop {stop_mode}")
    sig["long_trigger"], sig["short_trigger"] = sig["long_stop"], sig["short_stop"]
    return sig
