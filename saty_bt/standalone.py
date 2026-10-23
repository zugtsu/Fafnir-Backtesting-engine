"""
Strategies that run outside the engine, as importable code.

Both of these were originally run as inline scripts in a chat session and
existed nowhere on disk - including Storm Chaser, the strategy recommended for
real money, whose published numbers (287 trades, PF 1.84) are printed in the
header of pine/btc_storm_chaser.pine. A result that only exists in a chat
transcript cannot be re-run, regression-tested or audited, so they live here
now, lifted verbatim. Do not "tidy" the arithmetic: the golden-master tests in
tests/test_golden.py pin their exact trade lists, and any change that moves a
fill is a change to the strategy.

  rubber_band   AUDUSD 4h z-score fade (a third party's idea): enter at +/-2 SD
                from the 20-bar mean, exit back at the mean, stop 2 SD further
                out, 30-bar (5-day) time cap. Percent returns net of 2.9 bps.
  storm_chaser  BTC 1h 50-bar breakout taken only while ATR > 1.3x its 250-bar
                average and volume > 1.25x its 50-bar average; 3 ATR stop, no
                target, 500-bar cap. Percent returns net of 24 bps (perps).
                Runs on the long Bitstamp file, 2013-2026, NOT the engine's
                BTCUSD store (which starts 2021).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RB_COST = 0.00029
STORM_COST = 0.0024


def rubber_band(f: pd.DataFrame, z_in: float = 2.0, z_tp: float = 0.0, z_sl: float = 2.0,
                max_bars: int = 30, cost: float = RB_COST) -> pd.DataFrame:
    """-> trades with columns t (exit time), pct, why"""
    c = pd.to_numeric(f["close"], errors="coerce")
    o = pd.to_numeric(f["open"], errors="coerce").to_numpy(float)
    z = ((c - c.rolling(20).mean()) / c.rolling(20).std().replace(0, np.nan)).to_numpy(float)
    cl = c.to_numpy(float)
    pos, out = None, []
    for i in range(42, len(f) - 1):
        if pos is not None:
            d, e, i0 = pos
            zz = z[i]
            if not np.isfinite(zz):
                continue
            sl = zz <= -(z_in + z_sl) if d > 0 else zz >= (z_in + z_sl)
            tp = zz >= -z_tp if d > 0 else zz <= z_tp
            if sl or tp or i - i0 >= max_bars:
                out.append({"t": f.index[i], "pct": ((cl[i] - e) * d / e) - cost,
                            "why": "STOP" if sl else ("target" if tp else "TIME")})
                pos = None
            continue
        if not np.isfinite(z[i]):
            continue
        d = 1 if z[i] <= -z_in else -1 if z[i] >= z_in else 0
        if d:
            pos = (d, o[i + 1], i + 1)
    return pd.DataFrame(out)


def load_bitstamp_long(root: str | Path) -> pd.DataFrame:
    d = pd.read_csv(Path(root) / "cache" / "bitstamp_btcusd_3600_long.csv",
                    parse_dates=["ts"]).set_index("ts").sort_index()
    return d[~d.index.duplicated()]


def storm_chaser(d: pd.DataFrame, n: int = 50, vol_min: float = 1.3, stop_atr: float = 3.0,
                 rv_min: float = 1.25, max_bars: int = 500, cost: float = STORM_COST) -> pd.DataFrame:
    """-> trades with columns t (exit time), pct"""
    h, l, c, o, v = (pd.to_numeric(d[k], errors="coerce") for k in ("high", "low", "close", "open", "volume"))
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    hh = h.rolling(n).max().shift(1).to_numpy(float)
    ll = l.rolling(n).min().shift(1).to_numpy(float)
    ratio = (atr / atr.rolling(n * 5).mean()).to_numpy(float)
    av = atr.to_numpy(float)
    rv = (v / v.rolling(50).mean()).to_numpy(float)
    H, L, C, O = h.to_numpy(float), l.to_numpy(float), c.to_numpy(float), o.to_numpy(float)
    pos, out = None, []
    for i in range(n * 6 + 2, len(d) - 1):
        if pos is not None:
            dd_, e, st, i0 = pos
            hit = L[i] <= st if dd_ > 0 else H[i] >= st
            if hit or i - i0 >= max_bars:
                px = st if hit else C[i]
                out.append({"t": d.index[i], "pct": ((px - e) * dd_ / e) - cost})
                pos = None
            continue
        if not np.isfinite(ratio[i]) or ratio[i] < vol_min or not np.isfinite(hh[i]):
            continue
        if rv_min and (not np.isfinite(rv[i]) or rv[i] < rv_min):
            continue
        dr = 1 if C[i] > hh[i] and C[i - 1] <= hh[i] else (-1 if C[i] < ll[i] and C[i - 1] >= ll[i] else 0)
        if not dr:
            continue
        e = O[i + 1]
        pos = (dr, e, e - dr * stop_atr * av[i], i + 1)
    return pd.DataFrame(out)
