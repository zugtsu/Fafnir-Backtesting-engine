"""
Heikin Ashi bars.

    ha_close = (open + high + low + close) / 4
    ha_open  = (previous ha_open + previous ha_close) / 2
    ha_high  = max(high, ha_open, ha_close)
    ha_low   = min(low,  ha_open, ha_close)

THESE ARE NOT PRICES YOU CAN TRADE AT. ha_close is an average of the whole
bar, so it sits inside the bar's range and lags the real close; a backtest
that fills at ha_close is buying at a price that has already been and gone,
and it will print an equity curve that cannot be had. Signals may be computed
on these columns. Fills, stops and targets must use the real ones.

ha_open is recursive - each bar's open is half the previous ha_open plus half
the previous ha_close - so it is an infinite-impulse filter seeded at the
first bar. That recursion only ever looks backwards, so the columns are
causal, but it does mean the value at bar t depends on where the series
started.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def heikin_ashi(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    out = df.copy()
    o = pd.to_numeric(df["open"], errors="coerce").to_numpy(float)
    h = pd.to_numeric(df["high"], errors="coerce").to_numpy(float)
    l = pd.to_numeric(df["low"], errors="coerce").to_numpy(float)
    c = pd.to_numeric(df["close"], errors="coerce").to_numpy(float)

    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty(len(df), dtype=float)
    if len(df):
        ha_o[0] = (o[0] + c[0]) / 2.0
        for i in range(1, len(df)):
            prev_o = ha_o[i - 1]
            prev_c = ha_c[i - 1]
            if not np.isfinite(prev_o) or not np.isfinite(prev_c):
                ha_o[i] = (o[i] + c[i]) / 2.0        # reseed across a data hole
            else:
                ha_o[i] = (prev_o + prev_c) / 2.0

    out["ha_open"] = ha_o
    out["ha_close"] = ha_c
    out["ha_high"] = np.nanmax(np.vstack([h, ha_o, ha_c]), axis=0)
    out["ha_low"] = np.nanmin(np.vstack([l, ha_o, ha_c]), axis=0)
    # a run of same-coloured bars is the whole reason people use these
    up = out["ha_close"] > out["ha_open"]
    sign = pd.Series(np.where(up, 1, -1), index=df.index)
    grp = (sign != sign.shift(1)).cumsum()
    out["ha_streak"] = ((sign.groupby(grp).cumcount() + 1) * sign).astype(int)
    return out
