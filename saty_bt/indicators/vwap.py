"""
VWAP, session-anchored and rolling, with standard-deviation bands.

THE VOLUME PROBLEM. VWAP is only volume-weighted if there is volume. Spot FX
has none - there is no central exchange, so HistData ships a volume column of
zeros and any broker's "volume" is that one broker's tick count. Rather than
divide by zero or quietly produce a line that pretends to be something it is
not, this module weights by volume when the feed carries it and by 1.0 when it
does not, and records which happened in `vwap_weighted`.

With equal weights the session line is the running mean of typical price since
the session open. That is still a real, causal, tradeable anchor - it is what
an FX trader is looking at when they say "VWAP" - but it is NOT volume
weighted, and anything built on it must say so.

Columns added:
    vwap_sess       session-anchored VWAP (anchor = day_key, the trading day)
    vwap_sess_sd    weighted SD of typical price around it, same anchor
    vwap_sess_z     (close - vwap_sess) / vwap_sess_sd
    vwap_roll       n-bar rolling VWAP
    vwap_roll_sd    weighted SD around the rolling line
    vwap_roll_z     (close - vwap_roll) / vwap_roll_sd
    vwap_bars       bars elapsed in the session, 1 on the first bar
    vwap_warm       enough bars in the session for the SD to mean anything
    vwap_weighted   True if real volume was used, False if equal-weighted

Every value on row t uses bars completing at or before row t's close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _weights(df: pd.DataFrame) -> tuple[pd.Series, bool]:
    """Volume where the feed carries it, 1.0 where it does not."""
    vol = pd.to_numeric(df.get("volume"), errors="coerce") if "volume" in df else None
    if vol is None:
        return pd.Series(1.0, index=df.index), False
    usable = vol.notna() & (vol > 0)
    # a feed that is mostly empty is not a volume feed; do not half-use it
    if usable.mean() < 0.5:
        return pd.Series(1.0, index=df.index), False
    return vol.where(usable, 0.0).astype(float), True


def _sd_from_moments(sw: pd.Series, swx: pd.Series, swx2: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Weighted mean and SD from running sums of w, w*x and w*x^2."""
    sw = sw.replace(0.0, np.nan)
    mean = swx / sw
    var = (swx2 / sw) - mean ** 2
    return mean, np.sqrt(var.clip(lower=0.0))


def vwap(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or {}
    n = int(cfg.get("roll_length", 50))
    warm = int(cfg.get("warmup_bars", 5))
    out = df.copy()

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    w, weighted = _weights(df)
    wx, wx2 = w * tp, w * tp * tp

    if "day_key" in df:
        anchor = pd.Series(df["day_key"].to_numpy(), index=df.index)
    else:                                        # daily bars: each bar is its own session
        anchor = pd.Series(df.index, index=df.index)
    g = anchor.groupby(anchor.to_numpy())

    sw = w.groupby(anchor.to_numpy()).cumsum()
    swx = wx.groupby(anchor.to_numpy()).cumsum()
    swx2 = wx2.groupby(anchor.to_numpy()).cumsum()
    mean, sd = _sd_from_moments(sw, swx, swx2)
    out["vwap_sess"], out["vwap_sess_sd"] = mean, sd
    out["vwap_sess_z"] = (df["close"] - mean) / sd.replace(0.0, np.nan)

    bars = g.cumcount() + 1
    out["vwap_bars"] = bars.astype(int)
    out["vwap_warm"] = bars >= warm

    rw = w.rolling(n).sum()
    rmean, rsd = _sd_from_moments(rw, wx.rolling(n).sum(), wx2.rolling(n).sum())
    out["vwap_roll"], out["vwap_roll_sd"] = rmean, rsd
    out["vwap_roll_z"] = (df["close"] - rmean) / rsd.replace(0.0, np.nan)

    out["vwap_weighted"] = weighted
    return out
