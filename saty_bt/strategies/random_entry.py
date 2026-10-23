"""
Random-entry baseline (validation, brief section 7).

Entries fire at random bars, seeded, at roughly the same frequency as the
strategy being tested; exits, sizing and costs are identical. If a
strategy's edge is real its expectancy must beat this by a clear margin;
if the baseline earns the same, the money is coming from the exit
structure (or the market drift), not the entry rule.

exit_structure:
    saty    : Strategy A geometry - target +0.24 ATR from entry, runner +0.62,
              intrabar stop 0.24 ATR from entry, 50% partial, flat at close
    spx_opt : the SPX walk-forward structure - direction is the side of PDC
              price is on, stop at PDC, target 0.786 ATR from PDC, runner
              1.236, 50% partial, flat at close
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .base import finish_direction


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    seed = int(cfg.get("seed", 0)) + int(hashlib.sha256(acfg["name"].encode()).hexdigest()[:6], 16)
    rng = np.random.default_rng(seed)
    per_day = float(scfg.get("entries_per_day", 0.25))
    bars_per_day = f.groupby("day_key").size().median()
    p = min(1.0, per_day / max(bars_per_day, 1))
    fire = rng.random(len(f)) < p
    side = rng.random(len(f)) < 0.5
    structure = scfg.get("exit_structure", "saty")
    pdc, atr, close = f["pdc"], f["atr"], f["close"]

    sig = pd.DataFrame(index=f.index)
    if structure == "spx_opt":
        above = close > pdc
        finish_direction(sig, f, 1, pd.Series(fire, index=f.index) & above, scfg, acfg, cfg, {})
        finish_direction(sig, f, -1, pd.Series(fire, index=f.index) & ~above, scfg, acfg, cfg, {})
        sig["stop_mode"] = "pdc"
        sig["long_stop"], sig["short_stop"] = pdc, pdc
        sig["long_target"], sig["short_target"] = pdc + 0.786 * atr, pdc - 0.786 * atr
        sig["long_runner"], sig["short_runner"] = pdc + 1.236 * atr, pdc - 1.236 * atr
    else:
        finish_direction(sig, f, 1, pd.Series(fire & side, index=f.index), scfg, acfg, cfg, {})
        finish_direction(sig, f, -1, pd.Series(fire & ~side, index=f.index), scfg, acfg, cfg, {})
        k_t, k_r, k_s = (float(scfg.get(k, v)) for k, v in (("target_atr", 0.24), ("runner_atr", 0.62), ("stop_atr", 0.24)))
        sig["stop_mode"] = "atr"
        sig["long_stop"], sig["short_stop"] = close - k_s * atr, close + k_s * atr
        sig["long_target"], sig["short_target"] = close + k_t * atr, close - k_t * atr
        sig["long_runner"], sig["short_runner"] = close + k_r * atr, close - k_r * atr
    sig["long_trigger"], sig["short_trigger"] = sig["long_stop"], sig["short_stop"]
    return sig
