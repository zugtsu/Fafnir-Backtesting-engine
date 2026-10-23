"""
Kaufman's noise measures and adaptive trend signal (EDGE_PREREG_KAUFMAN.md).

Every function here is a pure transform of CLOSED bars: the value on row t
uses rows <= t only, and is known at row t's close. Nothing is shifted
forward; acting on a value is the simulator's job (next bar's open).

  efficiency_ratio  ER_t = |C_t - C_(t-n)| / sum_(i=t-n+1..t) |C_i - C_(i-1)|.
                    1 = a straight line, 0 = pure noise; 0 when the
                    denominator is 0 (a flat stretch). NaN for t < n.

  kama              KAMA(n, fast, slow): SC_t = (ER_t x (2/(fast+1) -
                    2/(slow+1)) + 2/(slow+1))^2, KAMA_t = KAMA_(t-1) +
                    SC_t x (C_t - KAMA_(t-1)), seeded with C at the first bar
                    where ER exists (row n).

  kama_state        Kaufman's filtered KAMA turn, +1 / -1 (0 before the first
                    signal). Rules are spelled out in its docstring.

  breakout_signals  Close vs the PRIOR n-bar highest / lowest close.

  er_pass           ER_t above the q-quantile of ER over the trailing window
                    ending at t (t included).

  breakout_position The pre-registered H6 baseline as a desired-position
                    series for saty_bt.sim.simulate, with an optional entry
                    gate (the ER filter).

Sums and standard deviations are taken window by window (numpy sliding
windows), not with running add/remove updates, so a value depends only on its
own window and an independent re-implementation can match it to the last bit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def _num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")


def efficiency_ratio(close: pd.Series, n: int = 10) -> pd.Series:
    """ER over n bars, as defined above. NaN where any input in the window is NaN."""
    if n < 1:
        raise ValueError("n must be >= 1")
    c = _num(close)
    out = np.full(len(c), np.nan)
    if len(c) > n:
        step = np.abs(np.diff(c))                          # step[i] = |C_(i+1) - C_i|
        noise = sliding_window_view(step, n).sum(axis=1)   # row w -> t = w + n
        signal = np.abs(c[n:] - c[:-n])
        with np.errstate(invalid="ignore", divide="ignore"):
            er = np.where(noise > 0, signal / noise, 0.0)
        er = np.minimum(er, 1.0)                           # rounding guard only
        er[~(np.isfinite(noise) & np.isfinite(signal))] = np.nan
        out[n:] = er
    return pd.Series(out, index=close.index, name=f"er{n}")


def kama(close: pd.Series, n: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman's adaptive moving average. A bar with a NaN close (or NaN ER)
    after the seed carries the previous KAMA forward."""
    c = _num(close)
    er = efficiency_ratio(close, n).to_numpy()
    fsc, ssc = 2.0 / (fast + 1), 2.0 / (slow + 1)
    sc = (er * (fsc - ssc) + ssc) ** 2
    out = np.full(len(c), np.nan)
    ok = np.isfinite(er) & np.isfinite(c)
    if ok.any():
        first = int(np.argmax(ok))
        k = c[first]
        out[first] = k
        for t in range(first + 1, len(c)):
            if ok[t]:
                k = k + sc[t] * (c[t] - k)
            out[t] = k
    return pd.Series(out, index=close.index, name=f"kama{n}")


def kama_filter(kama_s: pd.Series, k: float, sd_n: int = 20, ddof: int = 0) -> pd.Series:
    """f_t = k x SD of the last sd_n one-bar KAMA changes (the window ends at
    t and includes KAMA_t - KAMA_(t-1)). Population SD (ddof=0) by default, as
    in Kaufman's EasyLanguage StdDev. NaN until sd_n changes exist."""
    K = _num(kama_s)
    d = np.full(len(K), np.nan)
    if len(K) > 1:
        d[1:] = K[1:] - K[:-1]
    sd = np.full(len(K), np.nan)
    if len(K) >= sd_n:
        sd[sd_n - 1:] = sliding_window_view(d, sd_n).std(axis=1, ddof=ddof)
    return pd.Series(k * sd, index=kama_s.index, name="kama_filter")


def kama_state(kama_s: pd.Series, k: float, sd_n: int = 20, ddof: int = 0) -> pd.Series:
    """
    Kaufman's KAMA signal: +1 long / -1 short, 0 before the first signal.
    Known at bar close (act at the next open).

    - f_t = kama_filter(kama, k, sd_n). Nothing happens while f_t is NaN; the
      same warm-up applies for every k, k = 0 included.
    - Tracking starts at the first bar with a filter: lowest = highest =
      KAMA there. On each later bar both are updated with KAMA_t first
      (they include the current bar).
    - lowest = lowest KAMA since the last sell (reset to KAMA at a sell bar);
      highest = highest KAMA since the last buy (reset at a buy bar). Before
      the first signal both run from the start of tracking.
    - Turn LONG (if not long) when KAMA_t - lowest >= f_t AND KAMA_t > lowest.
      Turn SHORT (if not short) when highest - KAMA_t >= f_t AND
      KAMA_t < highest. The strict part is what makes k = 0 mean "any turn
      counts" rather than "always triggered".
    - Only possible before the first signal, and only when f falls between
      bars: if both fire on one bar, the sign of KAMA_t - KAMA_(t-1) picks
      (no signal if it is 0).
    """
    K = _num(kama_s)
    f = kama_filter(kama_s, k, sd_n, ddof).to_numpy()
    out = np.zeros(len(K), dtype=np.int64)
    state, started = 0, False
    low = high = np.nan
    for t in range(len(K)):
        kt, ft = K[t], f[t]
        if not (np.isfinite(kt) and np.isfinite(ft)):
            out[t] = state
            continue
        if not started:
            low = high = kt
            started = True
            out[t] = state
            continue
        low = min(low, kt)
        high = max(high, kt)
        up = (kt - low >= ft) and (kt > low)
        dn = (high - kt >= ft) and (kt < high)
        if state == 0 and up and dn:
            step = kt - K[t - 1]
            up, dn = step > 0, step < 0
        if up and state != 1:
            state, high = 1, kt
        elif dn and state != -1:
            state, low = -1, kt
        out[t] = state
    return pd.Series(out, index=kama_s.index, name="kama_state")


def breakout_signals(close: pd.Series, entry_n: int = 20, exit_n: int = 10) -> pd.DataFrame:
    """
    Channel breakout on PRIOR closes (the channel at t excludes C_t):
      long_entry   C_t > max(C_(t-entry_n) .. C_(t-1))
      short_entry  C_t < min(same window)
      long_exit    C_t < min(C_(t-exit_n) .. C_(t-1))
      short_exit   C_t > max(same window)
    All False until the window is full. Channel levels are returned too.
    """
    c = pd.to_numeric(close, errors="coerce")
    hi_e = c.rolling(entry_n, min_periods=entry_n).max().shift(1)
    lo_e = c.rolling(entry_n, min_periods=entry_n).min().shift(1)
    hi_x = c.rolling(exit_n, min_periods=exit_n).max().shift(1)
    lo_x = c.rolling(exit_n, min_periods=exit_n).min().shift(1)
    return pd.DataFrame({
        "long_entry": (c > hi_e).fillna(False).astype(bool),
        "short_entry": (c < lo_e).fillna(False).astype(bool),
        "long_exit": (c < lo_x).fillna(False).astype(bool),
        "short_exit": (c > hi_x).fillna(False).astype(bool),
        "entry_hi": hi_e, "entry_lo": lo_e, "exit_hi": hi_x, "exit_lo": lo_x,
    }, index=close.index)


def er_pass(er: pd.Series, window: int = 250, q: float = 0.5) -> pd.Series:
    """True when ER_t > the q-quantile (linear interpolation) of ER over the
    `window` bars ending at t, t included. False until `window` non-NaN ER
    values are in the window. Median: q = 0.5; 67th percentile: q = 0.67."""
    e = pd.to_numeric(er, errors="coerce")
    thr = e.rolling(window, min_periods=window).quantile(q)
    return ((e > thr) & thr.notna()).astype(bool).rename("er_pass")


def breakout_position(sig: pd.DataFrame, long_only: bool = False,
                      allow: pd.Series | None = None) -> pd.Series:
    """
    The H6 baseline as a desired position (+1 / 0 / -1) known at each close,
    for saty_bt.sim.simulate. `sig` is breakout_signals(...). Per bar, from
    the previous state:
      long:  short_entry (and allowed, and not long_only) -> -1;
             else long_exit -> 0; else stay long
      short: long_entry (and allowed) -> +1; else short_exit -> 0
      flat:  long_entry (and allowed) -> +1;
             else short_entry (and allowed, not long_only) -> -1
    Entries are level conditions (close beyond the prior channel), so a bar
    that is still beyond the channel while flat is a signal bar. `allow` gates
    ENTRIES only (the ER filter, at the signal bar); exits are never gated.
    """
    le, se = sig["long_entry"].to_numpy(bool), sig["short_entry"].to_numpy(bool)
    lx, sx = sig["long_exit"].to_numpy(bool), sig["short_exit"].to_numpy(bool)
    al = (np.ones(len(sig), bool) if allow is None
          else allow.reindex(sig.index).fillna(False).to_numpy(bool))
    out = np.zeros(len(sig), dtype=np.int64)
    pos = 0
    for t in range(len(sig)):
        if pos == 1:
            if se[t] and al[t] and not long_only:
                pos = -1
            elif lx[t]:
                pos = 0
        elif pos == -1:
            if le[t] and al[t]:
                pos = 1
            elif sx[t]:
                pos = 0
        else:
            if le[t] and al[t]:
                pos = 1
            elif se[t] and al[t] and not long_only:
                pos = -1
        out[t] = pos
    return pd.Series(out, index=sig.index, name="breakout_pos")
