"""
Saty Phase Oscillator - ported from the thinkScript:

    pivot       = EMA(close, 21)
    raw         = ((close - pivot) / (3.0 * ATR(14))) * 100
    oscillator  = EMA(raw, 3)           # rounded to 2 dp in the original
    bband_up/dn = pivot +/- 2 * stdev(close, 21)
    kelt_up/dn  = pivot +/- 2 * ATR(14)
    expansion   = kelt_up < bband_up and kelt_dn > bband_dn
    compression = not expansion

The ATR is the CHART timeframe ATR(14), not the level-period ATR.

Zones (Wyckoff-style): > +100 extended up, +61.8..+100 distribution,
+23.6..+61.8 mark-up, -23.6..+23.6 launch/neutral, -61.8..-23.6 mark-down,
-100..-61.8 accumulation, < -100 extended down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .core import atr, cross_above, cross_below, ema, stdev


def phase_oscillator(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pe, an = int(cfg.get("pivot_ema", 21)), int(cfg.get("atr_length", 14))
    mult, sm = float(cfg.get("atr_mult", 3.0)), int(cfg.get("smooth", 3))
    bbl, bbs = int(cfg.get("bb_length", 21)), float(cfg.get("bb_std", 2.0))
    km = float(cfg.get("keltner_mult", 2.0))
    z = cfg.get("zones", {})
    ext, dist, neu = float(z.get("extreme", 100.0)), float(z.get("distribution", 61.8)), float(z.get("neutral", 23.6))

    out = df.copy()
    pivot = ema(df["close"], pe)
    a = atr(df, an)
    raw = (df["close"] - pivot) / (mult * a) * 100.0
    osc = ema(raw, sm)
    if cfg.get("round_2dp", True):
        osc = osc.round(2)
    out["po_pivot"] = pivot
    out["po_atr"] = a
    out["osc"] = osc
    out["osc_prev"] = osc.shift(1)
    out["osc_rising"] = osc > osc.shift(1)

    sd = stdev(df["close"], bbl)
    bb_up, bb_dn = pivot + bbs * sd, pivot - bbs * sd
    mode = cfg.get("compression_mode", "pine")
    if mode == "pine":
        # Saty's Pine source (2022-2026): only the side of the pivot price is
        # on counts. compression <= 0 when that Bollinger band sits inside
        # pivot +/- 2 ATR; the tracker releases only once the band is
        # expanding AND beyond 1.854 ATR (a hysteresis band 1.854 to 2.0).
        above = df["close"] >= pivot
        exp_mult = float(cfg.get("expansion_mult", 1.854))
        comp = np.where(above, bb_up - (pivot + km * a), (pivot - km * a) - bb_dn)
        in_exp = np.where(above, bb_up - (pivot + exp_mult * a), (pivot - exp_mult * a) - bb_dn)
        comp = pd.Series(comp, index=df.index)
        expanding = comp.shift(1) <= comp
        tracker = (comp <= 0) & ~(expanding & (pd.Series(in_exp, index=df.index) > 0))
        out["compression"] = tracker.fillna(False) & sd.notna()
        out["expansion"] = ~out["compression"] & sd.notna()
        out["compression_value"] = comp
    else:
        # symmetric variant (the brief's pseudo-code): both bands inside the channel
        k_up, k_dn = pivot + km * a, pivot - km * a
        expansion = (k_up < bb_up) & (k_dn > bb_dn)
        out["expansion"] = expansion
        out["compression"] = ~expansion & sd.notna()
    comp_prev = out["compression"].shift(1, fill_value=False)
    released = comp_prev & ~out["compression"]
    out["compression_exit_bull"] = released & out["osc_rising"]
    out["compression_exit_bear"] = released & ~out["osc_rising"]
    # consecutive compression bars
    grp = (out["compression"] != comp_prev).cumsum()
    out["compression_bars"] = out["compression"].groupby(grp).cumsum().where(out["compression"], 0).astype(int)

    zone = np.select(
        [osc > ext, osc > dist, osc > neu, osc >= -neu, osc >= -dist, osc >= -ext],
        ["extended_up", "distribution", "markup", "neutral", "markdown", "accumulation"],
        "extended_down")
    out["osc_zone"] = np.where(osc.isna(), "na", zone)

    # exactly as the Pine source: previous bar at or beyond the line, this bar inside it
    prev = osc.shift(1)
    out["leaving_accumulation"] = (prev <= -dist) & (osc > -dist)
    out["leaving_distribution"] = (prev >= dist) & (osc < dist)
    out["leaving_extreme_up"] = (prev >= ext) & (osc < ext)
    out["leaving_extreme_down"] = (prev <= -ext) & (osc > -ext)
    out["zero_cross_up"] = cross_above(osc, 0.0)
    out["zero_cross_down"] = cross_below(osc, 0.0)
    yl = pd.Series(False, index=df.index)
    for lv in (dist, -dist, ext, -ext):
        yl |= cross_above(osc, lv) | cross_below(osc, lv)
    out["yellow_light"] = yl
    return out
