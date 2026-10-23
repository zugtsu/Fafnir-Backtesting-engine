"""
Strategy A - Golden Gate.

Setup (long): a confirmed close through the entry level. Variant "gate"
enters when the bar closes above the gate level (0.382 ATR by default,
the gate opens); variant "trigger" enters on the close above the trigger
(0.236) and treats the gate as the first scale. Shorts mirror.

"Fresh" means the previous bar closed at or below the level, so a bar
already sitting above it does not re-fire. On the first bar of a period
the previous close is the prior period's last close; if the bar OPENS
beyond the level it is a gap entry and needs allow_gap_entries.

All fractions are parameters so the sweep can move them:
    trigger_frac (0.236) gate_frac (0.382) target_frac (0.618) runner_frac (1.0)

Stops:
    trigger       : close back through the trigger (confirmed close, exit
                    next open) - Saty's published rule
    trigger_touch : trigger level touched intrabar
    swing         : last confirmed swing low/high (intrabar)
    pdc           : previous period close (intrabar)
    atr           : entry level minus stop_atr_frac x ATR (intrabar)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import evaluate_gates

STOP_MODES = ("trigger", "trigger_touch", "swing", "pdc", "atr",
              "next_level",                       # nearest ATR-ladder level below the entry level (intrabar)
              "fixed",                            # stop_abs price units below the entry level (e.g. 20 pips), intrabar
              "bar_low", "prev_bar_low", "prev2_bar_low",   # low/high of the signal bar, the bar before it, or two bars before, intrabar
              "bar_open", "prev_bar_open",                  # OPEN price of the signal bar / the bar before it, intrabar
              "ema21_touch", "ema34_touch",       # ribbon stop: prior bar's EMA, intrabar, trails
              "ema21_close", "ema34_close")       # ribbon stop: confirmed close through the EMA


def _fresh_cross(close: pd.Series, level: pd.Series, direction: int) -> pd.Series:
    prev = close.shift(1)
    if direction > 0:
        return (close > level) & (prev <= level)
    return (close < level) & (prev >= level)


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    variant = scfg.get("entry", "gate")
    tf_, gf = float(scfg.get("trigger_frac", 0.236)), float(scfg.get("gate_frac", 0.382))
    tgf, rnf = float(scfg.get("target_frac", scfg.get("target", 0.618))), float(scfg.get("runner_frac", scfg.get("runner", 1.0)))
    pdc, atr = f["pdc"], f["atr"]
    trig_up, trig_dn = pdc + tf_ * atr, pdc - tf_ * atr
    gate_up, gate_dn = pdc + gf * atr, pdc - gf * atr
    if variant == "gate":
        lvl_up, lvl_dn = gate_up, gate_dn
        tgt_up, tgt_dn = pdc + tgf * atr, pdc - tgf * atr
        run_up, run_dn = pdc + rnf * atr, pdc - rnf * atr
    elif scfg.get("scale_at_gate", True):   # trigger entry, first scale at the gate (brief's variant B)
        lvl_up, lvl_dn = trig_up, trig_dn
        tgt_up, tgt_dn = gate_up, gate_dn
        run_up, run_dn = pdc + tgf * atr, pdc - tgf * atr
    else:                                   # trigger entry, Saty's ladder: target 61.8, runner 1 ATR / 161.8
        lvl_up, lvl_dn = trig_up, trig_dn
        tgt_up, tgt_dn = pdc + tgf * atr, pdc - tgf * atr
        run_up, run_dn = pdc + rnf * atr, pdc - rnf * atr

    first_bar = f["day_key"] != f["day_key"].shift(1)
    allow_gap = bool(acfg.get("allow_gap_entries", False))

    sig = pd.DataFrame(index=f.index)
    sig["entry_level_up"], sig["entry_level_down"] = lvl_up, lvl_dn
    if scfg.get("entry_at", "level") == "ribbon":
        # "look for an entry at the ribbon": with entry_fill "limit" the order
        # rests at the fast ribbon EMA as of the signal bar and fills only if
        # price pulls back to it within limit_valid_bars. A fill below the
        # stop is refused by the engine (stop_dist must be positive).
        from ..indicators.core import ema as _ema
        e = _ema(f["close"], int(scfg.get("ribbon_entry_ema", 8)))
        sig["entry_level_up"], sig["entry_level_down"] = e, e
    for d, name, lvl in ((1, "long", lvl_up), (-1, "short", lvl_dn)):
        setup = _fresh_cross(f["close"], lvl, d) & f["levels_valid"]
        gap = first_bar & ((f["open"] > lvl) if d > 0 else (f["open"] < lvl))
        sig[f"{name}_gap"] = gap & setup
        if not allow_gap:
            setup &= ~gap
        g = evaluate_gates(f, d, scfg, acfg, cfg, gates_enabled)
        sig[f"{name}_setup"] = setup
        entry = setup & g["gates_ok"]
        block = g["block_reason"].copy()
        if scfg.get("require_conviction", False):      # 13/48 conviction EMAs must agree
            conv = f["conviction_bull"] if d > 0 else f["conviction_bear"]
            block = block.where(~(entry & ~conv), "conviction")
            entry &= conv
        sig[f"{name}_entry"] = entry
        sig[f"{name}_block"] = block
        for c in g.columns:
            if c.startswith("gate_") or c.startswith("why_"):
                sig[f"{name}_{c}"] = g[c]

    # optional close-based take profit: exit at the next open once a bar CLOSES
    # beyond the named level (gate = 38.2, target = 61.8), in the trade's favour
    exit_lvl = scfg.get("exit_on_close_through")          # None | "gate" | "target" | a fraction of ATR from PDC
    if exit_lvl:
        if exit_lvl == "gate":
            up, dn = gate_up, gate_dn
        elif exit_lvl == "target":
            up, dn = pdc + tgf * atr, pdc - tgf * atr
        else:
            k = float(exit_lvl)
            up, dn = pdc + k * atr, pdc - k * atr
        sig["long_exit"] = f["close"] >= up
        sig["short_exit"] = f["close"] <= dn

    stop_mode = scfg.get("stop", "trigger")
    if stop_mode not in STOP_MODES:
        raise ValueError(f"unknown stop mode {stop_mode}")
    sig["stop_mode"] = stop_mode
    sig["long_target"], sig["short_target"] = tgt_up, tgt_dn
    sig["long_runner"], sig["short_runner"] = run_up, run_dn
    sig["long_trigger"], sig["short_trigger"] = trig_up, trig_dn
    if stop_mode in ("trigger", "trigger_touch"):
        sig["long_stop"], sig["short_stop"] = trig_up, trig_dn
    elif stop_mode == "pdc":
        sig["long_stop"], sig["short_stop"] = pdc, pdc
    elif stop_mode == "swing":
        sig["long_stop"], sig["short_stop"] = f["swing_low"], f["swing_high"]
    elif stop_mode == "atr":
        k = float(scfg.get("stop_atr_frac", 0.5))
        sig["long_stop"], sig["short_stop"] = lvl_up - k * atr, lvl_dn + k * atr
    elif stop_mode in ("bar_low", "prev_bar_low", "prev2_bar_low"):
        k = {"bar_low": 0, "prev_bar_low": 1, "prev2_bar_low": 2}[stop_mode]
        pad = float(scfg.get("stop_pad_atr", 0.0))
        sig["long_stop"] = f["low"].shift(k) - pad * atr
        sig["short_stop"] = f["high"].shift(k) + pad * atr
        sig["stop_mode"] = "atr"                                     # intrabar, static at entry
    elif stop_mode in ("bar_open", "prev_bar_open"):
        k = {"bar_open": 0, "prev_bar_open": 1}[stop_mode]
        sig["long_stop"] = f["open"].shift(k)
        sig["short_stop"] = f["open"].shift(k)
        sig["stop_mode"] = "atr"                                     # intrabar, static at entry
    elif stop_mode == "fixed":
        k = float(scfg.get("stop_abs", 0))
        sig["long_stop"], sig["short_stop"] = lvl_up - k, lvl_dn + k
        sig["stop_mode"] = "atr"                                     # intrabar, static at entry
    elif stop_mode == "next_level":
        # the ladder level immediately below the entry level, at least min_gap away
        from .base import LADDER
        min_gap = float(scfg.get("min_stop_atr_gap", 0.1))
        entry_frac = gf if variant == "gate" else tf_
        below = [x for x in LADDER if x <= entry_frac - min_gap]
        k = max(below) if below else -1.0
        sig["long_stop"], sig["short_stop"] = pdc + k * atr, pdc - k * atr
        sig["stop_mode"] = "atr"                                     # intrabar, static at entry
    elif stop_mode.startswith("ema"):
        n = int(stop_mode[3:5])
        e = f[f"ema_{n}"]
        if stop_mode.endswith("_touch"):
            sig["long_stop"], sig["short_stop"] = e.shift(1), e.shift(1)
            sig["stop_mode"] = "dynamic_touch"
        else:
            sig["long_stop"], sig["short_stop"] = e, e
            sig["stop_mode"] = "close_through"
    return sig
