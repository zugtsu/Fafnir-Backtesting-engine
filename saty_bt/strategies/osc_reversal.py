"""
Strategy C - Phase Oscillator Reversal (mean reversion).

Entry long: the oscillator leaves accumulation (crosses above -61.8) or
leaves the extended-down zone (crosses back above -100), on a bar with a
buy-winning Volume Stack and rel-vol >= `rel_vol_min` (1.2, advisory
where the feed has no volume). Shorts mirror from distribution / extended up.
Filter (toggle `ribbon_filter`): reject longs while the ribbon is in
strong bear conviction (EMA13 < EMA48 with a widening spread), because
Saty markets this as a mean-reversion tool and the filter may starve it.
Target: PDC or the nearest ATR level in the trade direction (whichever is
nearer, PDC included in the ladder); runner to the next.
Stop: recent swing low/high (intrabar). Exit signal: the oscillator drops
back below -100 (longs) / above +100 (shorts) - the re-entry rule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import finish_direction, next_levels


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    ext = float(cfg["indicators"]["phase_oscillator"].get("zones", {}).get("extreme", 100.0))
    setup_l = f["leaving_accumulation"] | f["leaving_extreme_down"]
    setup_s = f["leaving_distribution"] | f["leaving_extreme_up"]
    if scfg.get("ribbon_filter", True):
        strong_bear = f["conviction_bear"] & f["conviction_widening"]
        strong_bull = f["conviction_bull"] & f["conviction_widening"]
        setup_l &= ~strong_bear
        setup_s &= ~strong_bull

    sig = pd.DataFrame(index=f.index)
    finish_direction(sig, f, 1, setup_l, scfg, acfg, cfg, gates_enabled)
    finish_direction(sig, f, -1, setup_s, scfg, acfg, cfg, gates_enabled)

    t_l, r_l = next_levels(f, 1)
    t_s, r_s = next_levels(f, -1)
    sig["long_target"], sig["short_target"] = t_l, t_s
    sig["long_runner"], sig["short_runner"] = r_l, r_s
    sig["stop_mode"] = "swing"
    sig["long_stop"], sig["short_stop"] = f["swing_low"], f["swing_high"]
    sig["long_trigger"], sig["short_trigger"] = sig["long_stop"], sig["short_stop"]
    sig["long_exit"] = f["osc"] < -ext
    sig["short_exit"] = f["osc"] > ext
    return sig
