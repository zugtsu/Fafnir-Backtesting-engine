"""
Strategy E - Full stack confluence (the "everything on" control).

Strategy A's gate entry with every gate required, plus:
    oscillator between -23.6 and +61.8 and rising (require_rising)
    conviction agreeing (EMA13 > EMA48 for longs)
    volume conviction where volume exists (rel-vol >= 1.5, hard gate)
Expect fewer, higher-quality trades; compare against A-D.
"""

from __future__ import annotations

import copy

import pandas as pd

from . import golden_gate


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    s = copy.deepcopy(scfg)
    s.setdefault("gates", {})
    for g in ("ribbon", "time_warp", "time_of_day", "volume", "oscillator", "range_util"):
        s["gates"].setdefault(g, True)
    s.setdefault("oscillator", {})
    s["oscillator"]["require_rising"] = True
    sig = golden_gate.build_signals(f, s, acfg, cfg, gates_enabled)
    if scfg.get("require_conviction", True):
        sig["long_entry"] &= f["conviction_bull"]
        sig["short_entry"] &= f["conviction_bear"]
        sig.loc[sig["long_setup"] & ~f["conviction_bull"] & (sig["long_block"] == ""), "long_block"] = "conviction"
        sig.loc[sig["short_setup"] & ~f["conviction_bear"] & (sig["short_block"] == ""), "short_block"] = "conviction"
    return sig
