"""
Ehlers Voss Predictive Filter (John Ehlers, "A Peek Into The Future",
Technical Analysis of Stocks & Commodities, August 2019).

Two lines:
    voss_filt   a wide band-pass filter of price, centred on `period` bars -
                a zero-mean cycle oscillator in price units
    voss        Henning Voss's predictor applied to that filter, which LEADS
                it by roughly `predict` bars on a cycle near `period`

Literal port of Ehlers' EasyLanguage (degrees -> radians):

    Order = 3*Predict
    F1 = cos(2pi/Period);  G1 = cos(Bandwidth*2pi/Period)
    S1 = 1/G1 - sqrt(1/G1^2 - 1)
    Filt = .5*(1-S1)*(Close-Close[2]) + F1*(1+S1)*Filt[1] - S1*Filt[2]
    (Filt = 0 on the first 5 bars)
    SumC = sum over count=0..Order-1 of ((count+1)/Order) * Voss[Order-count]
    Voss = ((3+Order)/2)*Filt - SumC

Reading used by the trading rules ("dots green, lines crossed above zero"):
    voss_bull   voss > filt (the predictor is above the filter)
    voss_up     voss_bull and both lines above zero
Shorts mirror. Both are causal: every value at bar t uses bars <= t.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def voss_filter(close: pd.Series, period: int = 20, predict: int = 3,
                bandwidth: float = 0.25) -> tuple[pd.Series, pd.Series]:
    c = pd.to_numeric(close, errors="coerce").to_numpy(float)
    n = len(c)
    order = 3 * predict
    f1 = math.cos(2 * math.pi / period)
    g1 = math.cos(bandwidth * 2 * math.pi / period)
    s1 = 1 / g1 - math.sqrt(1 / (g1 * g1) - 1)
    filt = np.zeros(n)
    voss = np.zeros(n)
    for i in range(n):
        if i <= 5 or not (np.isfinite(c[i]) and np.isfinite(c[i - 2])):
            filt[i] = 0.0
        else:
            filt[i] = (0.5 * (1 - s1) * (c[i] - c[i - 2])
                       + f1 * (1 + s1) * filt[i - 1] - s1 * filt[i - 2])
        sumc = 0.0
        for count in range(order):
            j = i - (order - count)
            if j >= 0:
                sumc += ((count + 1) / order) * voss[j]
        voss[i] = ((3 + order) / 2) * filt[i] - sumc
    idx = close.index
    return pd.Series(voss, index=idx), pd.Series(filt, index=idx)


def voss(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or {}
    out = df.copy()
    v, flt = voss_filter(df["close"], int(cfg.get("period", 20)), int(cfg.get("predict", 3)),
                         float(cfg.get("bandwidth", 0.25)))
    out["voss"], out["voss_filt"] = v, flt
    out["voss_bull"] = v > flt
    out["voss_bear"] = v < flt
    out["voss_up"] = out["voss_bull"] & (v > 0) & (flt > 0)
    out["voss_down"] = out["voss_bear"] & (v < 0) & (flt < 0)
    out["voss_cross_up"] = out["voss_bull"] & ~out["voss_bull"].shift(1, fill_value=False)
    out["voss_cross_down"] = out["voss_bear"] & ~out["voss_bear"].shift(1, fill_value=False)
    return out
