"""
Strategy D - Compression Breakout.

Setup: `compression` true for at least K bars (default 5).
Entry: the first bar where compression ends, closing beyond the ATR
trigger in the direction the ribbon / oscillator sign points, with the
Volume Stack winner agreeing (volume gate).
Target / stop as Strategy A: 61.8 target with a partial, runner to 100,
stop on a confirmed close back through the trigger.
"""

from __future__ import annotations

import pandas as pd

from .base import finish_direction


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    k = int(scfg.get("min_compression_bars", 5))
    ended = f["compression"].shift(1, fill_value=False) & ~f["compression"]
    long_enough = f["compression_bars"].shift(1, fill_value=0) >= k
    how = scfg.get("direction_from", "ribbon_or_osc")
    if how == "ribbon":
        dir_l, dir_s = f["ribbon_bull"], f["ribbon_bear"]
    elif how == "osc":
        dir_l, dir_s = f["osc"] > 0, f["osc"] < 0
    else:
        dir_l = f["ribbon_bull"] | (f["osc"] > 0)
        dir_s = f["ribbon_bear"] | (f["osc"] < 0)
    tf_ = float(scfg.get("trigger_frac", 0.236))
    tgf, rnf = float(scfg.get("target_frac", 0.618)), float(scfg.get("runner_frac", 1.0))
    trig_up, trig_dn = f["pdc"] + tf_ * f["atr"], f["pdc"] - tf_ * f["atr"]

    sig = pd.DataFrame(index=f.index)
    sig["entry_level_up"], sig["entry_level_down"] = trig_up, trig_dn     # for entry_fill: limit
    finish_direction(sig, f, 1, ended & long_enough & dir_l & (f["close"] > trig_up), scfg, acfg, cfg, gates_enabled)
    finish_direction(sig, f, -1, ended & long_enough & dir_s & (f["close"] < trig_dn), scfg, acfg, cfg, gates_enabled)
    sig["long_target"], sig["short_target"] = f["pdc"] + tgf * f["atr"], f["pdc"] - tgf * f["atr"]
    sig["long_runner"], sig["short_runner"] = f["pdc"] + rnf * f["atr"], f["pdc"] - rnf * f["atr"]
    sig["long_trigger"], sig["short_trigger"] = trig_up, trig_dn
    stop_mode = scfg.get("stop", "trigger")
    sig["stop_mode"] = stop_mode
    if stop_mode == "trigger":
        sig["long_stop"], sig["short_stop"] = trig_up, trig_dn
    elif stop_mode == "swing":
        sig["long_stop"], sig["short_stop"] = f["swing_low"], f["swing_high"]
    elif stop_mode == "pdc":
        sig["long_stop"], sig["short_stop"] = f["pdc"], f["pdc"]
    else:
        raise ValueError(f"compression_breakout: unknown stop {stop_mode}")
    return sig
