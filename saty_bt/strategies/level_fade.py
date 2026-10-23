"""
Strategy L - ATR level fade ("range exhaustion").

Saty's reading of the outer levels: once price has used a full ATR (the
1.0 level, or the 0.786 / 1.236 around it) the day's range is spent more
often than not, and the move tends to give some back. This strategy
fades that level back toward the inner ladder.

Short setup (long mirrors below the PDC):
    mode "rejection" : the bar's high reaches the fade level and the bar
                       CLOSES back under it; entry at the next open
    mode "touch"     : the bar's high reaches the level, no close test
    mode "close"     : the bar closes beyond the level (buy the blow-off
                       and fade it; the weakest form, kept for comparison)
    fresh            : the previous bar had not already reached the level

Stop   : pdc + stop_frac x ATR (static, intrabar)            e.g. 1.236 or 1.5
Target : pdc + target_frac x ATR (limit)                     e.g. 0.618
Limit fills use entry_level_up/down = the fade level itself, so a limit
order rests at the level and fills only if price comes back to it.
"""

from __future__ import annotations

import pandas as pd

from .base import finish_direction


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    k = float(scfg.get("fade_frac", 1.0))
    s = float(scfg.get("stop_frac", 1.236))
    t = float(scfg.get("target_frac", 0.618))
    mode = scfg.get("mode", "rejection")
    pdc, atr = f["pdc"], f["atr"]
    up, dn = pdc + k * atr, pdc - k * atr
    hi, lo, close = f["high"], f["low"], f["close"]

    if mode == "rejection":
        short_setup = (hi >= up) & (close < up)
        long_setup = (lo <= dn) & (close > dn)
    elif mode == "touch":
        short_setup, long_setup = hi >= up, lo <= dn
    elif mode == "close":
        short_setup, long_setup = close > up, close < dn
    else:
        raise ValueError(f"level_fade mode {mode}")
    if scfg.get("fresh", True):
        short_setup &= ~(hi.shift(1) >= up)
        long_setup &= ~(lo.shift(1) <= dn)

    sig = pd.DataFrame(index=f.index)
    sig["entry_level_up"], sig["entry_level_down"] = dn, up      # limit fill for a LONG rests at the lower level
    finish_direction(sig, f, 1, long_setup, scfg, acfg, cfg, gates_enabled)
    finish_direction(sig, f, -1, short_setup, scfg, acfg, cfg, gates_enabled)
    sig["stop_mode"] = "atr"
    sig["long_stop"], sig["short_stop"] = pdc - s * atr, pdc + s * atr
    sig["long_target"], sig["short_target"] = pdc - t * atr, pdc + t * atr
    sig["long_runner"], sig["short_runner"] = sig["long_target"], sig["short_target"]
    sig["long_trigger"], sig["short_trigger"] = sig["long_stop"], sig["short_stop"]
    return sig
