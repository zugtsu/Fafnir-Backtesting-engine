"""
How many of the user's indicators agree, bar by bar.

The user's chart (2026-10-06): Stochastic SMI, Ichimoku, Saty Pivot Ribbon,
Ehlers MADH and Chaikin Money Flow, with these settings as defaults. Each one
gives a bullish / bearish / neither reading:

    smi       SMI above / below its signal line          K 10, D 10, EMA 5
    ichimoku  close above the cloud AND Tenkan > Kijun    9 / 26 / 52
              (mirror for bearish)
    ribbon    fast > pivot > slow  (stacked)              8 / 21 / 48
    madh      above / below zero                          short 10, cycle 30
    cmf       above / below zero - ONLY where the feed    20
              has volume; on FX it is skipped, not
              counted as a disagreement

Shared by the `confluence` gate (base.py) and the `confluence` system
(indicator_lab.py) so the two can never drift apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {"smi": (10, 10, 10, 5), "ichimoku": (9, 26, 52), "ribbon": (8, 21, 48),
            "madh": (10, 30), "cmf": 20}


def readings(f: pd.DataFrame, cfg: dict | None = None) -> dict[str, pd.Series]:
    """name -> +1 bullish / -1 bearish / 0 neither (NaN where not computable)."""
    from .indicator_lab import _cmf, _ema, _madh, _smi
    cfg = cfg or {}
    p = {**DEFAULTS, **cfg.get("params", {})}
    use = cfg.get("use", ["smi", "ichimoku", "ribbon", "madh", "cmf"])
    h, l, c = f["high"], f["low"], f["close"]
    out = {}
    if "smi" in use:
        k, r, s, sg = p["smi"]
        smi, sig = _smi(f, k, r, s, sg)
        out["smi"] = np.sign(smi - sig)
    if "ichimoku" in use:
        t, kj, sb = p["ichimoku"]
        conv = (h.rolling(t).max() + l.rolling(t).min()) / 2
        base = (h.rolling(kj).max() + l.rolling(kj).min()) / 2
        sa = ((conv + base) / 2).shift(kj)
        sbl = ((h.rolling(sb).max() + l.rolling(sb).min()) / 2).shift(kj)
        top, bot = np.maximum(sa, sbl), np.minimum(sa, sbl)
        bull = (c > top) & (conv > base)
        bear = (c < bot) & (conv < base)
        r_ = pd.Series(np.where(bull, 1.0, np.where(bear, -1.0, 0.0)), index=f.index)
        out["ichimoku"] = r_.where(top.notna())
    if "ribbon" in use:
        a, b, s_ = p["ribbon"]
        e1, e2, e3 = _ema(c, a), _ema(c, b), _ema(c, s_)
        out["ribbon"] = pd.Series(np.where((e1 > e2) & (e2 > e3), 1.0,
                                           np.where((e1 < e2) & (e2 < e3), -1.0, 0.0)), index=f.index)
    if "madh" in use:
        out["madh"] = np.sign(_madh(c, *p["madh"]))
    if "cmf" in use and "volume" in f:
        v = pd.to_numeric(f["volume"], errors="coerce")
        if (v > 0).mean() >= 0.9:                 # a real volume feed, not a partial one
            out["cmf"] = np.sign(_cmf(f, int(p["cmf"])))
    return out


def agreement(f: pd.DataFrame, cfg: dict | None = None) -> tuple[pd.Series, pd.Series, int]:
    """-> (bullish count, bearish count, number of indicators in play)"""
    r = readings(f, cfg)
    m = pd.DataFrame(r)
    ready = m.notna().all(axis=1)
    bull = (m > 0).sum(axis=1).where(ready)
    bear = (m < 0).sum(axis=1).where(ready)
    return bull, bear, m.shape[1]


def need_for(o: dict, n: int) -> int:
    """min_agree: absent = all n; negative = all but that many (-1 -> n-1)."""
    ma = o.get("min_agree")
    if ma is None:
        return n
    ma = int(ma)
    return n + ma if ma < 0 else ma
