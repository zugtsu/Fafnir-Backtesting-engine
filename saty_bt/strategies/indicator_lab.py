"""
Strategy I - the indicator lab.

Seventeen textbook indicators, each wired up the way its documentation says to
trade it, all sharing one exit geometry so the comparison is about the signal
and nothing else: enter at the next open after the signal bar closes, stop a
fixed ATR multiple away, exit when the indicator flips (or at target_r).

Nothing here reads the Saty stack. These are the indicators a retail trader
actually has on the chart, computed from OHLC alone.

system:
    ma_cross       fast MA crosses slow MA              (trend)
    macd_cross     MACD line crosses its signal line    (trend/momentum)
    macd_zero      MACD line crosses zero               (trend)
    rsi_revert     RSI below lo buys, above hi sells    (mean reversion)
    rsi_trend      RSI crosses 50                       (momentum)
    stoch          %K crosses %D inside the OB/OS zone  (mean reversion)
    bb_revert      close outside the band, fade it      (mean reversion)
    bb_break       close outside the band, follow it    (breakout)
    keltner_break  same on an ATR channel               (breakout)
    cci            CCI crosses +/-100                   (momentum)
    williams_r     %R leaves the OB/OS zone             (mean reversion)
    psar           parabolic SAR flips                  (trend)
    supertrend     ATR band flips                       (trend)
    ichimoku       Tenkan/Kijun cross, cloud as filter  (trend)
    adx_di         +DI crosses -DI with ADX above min   (trend)
    aroon          Aroon up/down cross                  (trend)
    roc            rate of change crosses zero          (momentum)
    smi_cross      Blau SMI crosses its signal line     (momentum)
    smi_revert     SMI leaves the +/-lo extreme         (mean reversion)

Shared knobs: n_fast, n_slow, n, lo, hi, k_band, stop_atr_k, ma_kind,
exit_on_flip, target_r (engine).

A second indicator can stand behind the trigger as a filter, configured with
filt_kind (none | sma_trend | ema_trend | macd | supertrend | di | rsi50 |
cloud | ema_stack | adx_min | adx_max | atr_min | atr_max | stoch_* | smi_side | smi_zero |
smi_not_extended | smi_extended) plus filt_n / filt_v / filt_k / filt_n_fast /
filt_n_slow / filt_r / filt_s.

A momentum flip can also be used as an EARLY EXIT rather than an entry screen,
via exsig_kind (smi_cross | smi_side | smi_zero | stoch_cross | macd_cross |
rsi50) plus exsig_n / exsig_r / exsig_s / exsig_sig; see _exit_mask. The prefix is load-bearing: sweep bundles flatten
into the strategy config, so a bare `n` would retune the trigger instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..indicators.core import atr as _atr, ema as _ema
from .base import finish_direction


def _ma(s: pd.Series, n: int, kind: str) -> pd.Series:
    return _ema(s, n) if kind == "ema" else s.rolling(n).mean()


def _rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    au = d.clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    ad = (-d).clip(lower=0.0).ewm(alpha=1.0 / n, adjust=False).mean()
    return (100 - 100 / (1 + au / ad.replace(0.0, np.nan))).fillna(50.0)


def _stoch(f: pd.DataFrame, n: int, d_n: int, smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    """%K over `n` bars, slowed by `smooth`, with %D the `d_n`-bar average of it."""
    hh, ll = f["high"].rolling(n).max(), f["low"].rolling(n).min()
    k = 100 * (f["close"] - ll) / (hh - ll).replace(0.0, np.nan)
    k = k.rolling(max(1, smooth)).mean()         # slow %K
    return k, k.rolling(d_n).mean()


def _smi(f: pd.DataFrame, n: int = 10, r: int = 3, s: int = 3,
         sig_n: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Blau's Stochastic Momentum Index, and its signal line.

    Where the ordinary stochastic measures the close against the BOTTOM of the
    range, the SMI measures it against the range's MIDPOINT and double-smooths
    both halves of the ratio, so it runs -100..+100 with zero meaning "closed
    in the middle of the range".
    """
    hh, ll = f["high"].rolling(n).max(), f["low"].rolling(n).min()
    d = f["close"] - (hh + ll) / 2
    num = _ema(_ema(d, r), s)
    den = _ema(_ema(hh - ll, r), s) / 2
    smi = 100 * num / den.replace(0.0, np.nan)
    return smi, _ema(smi, sig_n)


def _hann_fir(c: np.ndarray, n: int) -> np.ndarray:
    """Ehlers' Hann-windowed FIR: weight on lag j is 1 - cos(2pi(j+1)/(n+1))."""
    w = 1 - np.cos(2 * np.pi * np.arange(1, n + 1) / (n + 1))
    out = np.convolve(c, w, mode="full")[:len(c)] / w.sum()   # out[t] = sum_j w_j c[t-j]
    out[:n - 1] = np.nan
    return out


def _madh(close: pd.Series, short: int = 8, cycle: int = 27) -> pd.Series:
    """
    Ehlers' MADH - Moving Average Difference, Hann-windowed (TASC, Nov 2021).
    Two Hann-weighted averages, short and short + cycle/2, and their gap as a
    percent of the long one. Crossing zero = the short average crossing the
    long one, with Hann smoothing so it lags and whipsaws less than an SMA.
    """
    c = pd.to_numeric(close, errors="coerce").to_numpy(float)
    f1 = _hann_fir(c, int(short))
    f2 = _hann_fir(c, int(short + cycle / 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = 100 * (f1 - f2) / f2
    return pd.Series(m, index=close.index)


def _madh_literal(close: pd.Series, short: int = 8, cycle: int = 27) -> pd.Series:
    """Bar-by-bar port of Ehlers' EasyLanguage, kept only to verify _madh."""
    c = pd.to_numeric(close, errors="coerce").to_numpy(float)
    long_ = int(short + cycle / 2)
    out = np.full(len(c), np.nan)
    for t in range(long_ - 1, len(c)):
        vals = []
        for n in (int(short), long_):
            filt = coef = 0.0
            for count in range(1, n + 1):
                k = 1 - np.cos(2 * np.pi * count / (n + 1))
                filt += k * c[t - (count - 1)]
                coef += k
            vals.append(filt / coef)
        out[t] = 100 * (vals[0] - vals[1]) / vals[1]
    return pd.Series(out, index=close.index)


def _cmf(f: pd.DataFrame, n: int = 20) -> pd.Series:
    """Chaikin Money Flow: where the close sits in the bar's range, weighted by
    volume, summed over n bars. NaN wherever there is no volume - and spot FX
    has none, so this is a crypto/equity indicator only."""
    h, l, c = f["high"], f["low"], f["close"]
    v = pd.to_numeric(f["volume"], errors="coerce").where(lambda x: x > 0)
    rng = (h - l).replace(0.0, np.nan)
    mfv = ((c - l) - (h - c)) / rng * v
    return mfv.rolling(n).sum() / v.rolling(n).sum()


def _exit_mask(f: pd.DataFrame, scfg: dict) -> tuple[pd.Series, pd.Series]:
    """
    A momentum flip used to LEAVE a trade early, rather than to screen entries.

    Keys are `exsig_`-prefixed for the same reason the filter keys are: sweep
    bundles flatten into the strategy config, and a bare `n` would retune the
    signal it is supposed to be judging.
    -> (exit_long, exit_short); all-False when no rule is set
    """
    kind = scfg.get("exsig_kind", "none")
    no = pd.Series(False, index=f.index)
    if kind in ("none", None):
        return no, no
    n = int(scfg.get("exsig_n", 10))
    close = f["close"]
    if kind.startswith("smi"):
        smi, sg = _smi(f, n, int(scfg.get("exsig_r", 3)), int(scfg.get("exsig_s", 3)),
                       int(scfg.get("exsig_sig", 3)))
        if kind == "smi_cross":               # the flip itself, once
            return _cross_up(sg, smi), _cross_up(smi, sg)
        if kind == "smi_side":                # any bar on the wrong side of the signal line
            return smi < sg, smi > sg
        if kind == "smi_zero":                # lost the midpoint of the range
            return _cross_up(pd.Series(0.0, index=f.index), smi), _cross_up(smi, pd.Series(0.0, index=f.index))
        raise ValueError(f"exsig {kind}")
    if kind == "stoch_cross":
        k_, d_ = _stoch(f, n, int(scfg.get("exsig_sig", 3)))
        return _cross_up(d_, k_), _cross_up(k_, d_)
    if kind == "macd_cross":
        ml = _ema(close, int(scfg.get("exsig_r", 12))) - _ema(close, int(scfg.get("exsig_s", 26)))
        sg = _ema(ml, int(scfg.get("exsig_sig", 9)))
        return _cross_up(sg, ml), _cross_up(ml, sg)
    if kind == "rsi50":
        r = _rsi(close, n)
        return _cross_up(pd.Series(50.0, index=f.index), r), _cross_up(r, pd.Series(50.0, index=f.index))
    raise ValueError(f"exsig {kind}")


def _cci(f: pd.DataFrame, n: int) -> pd.Series:
    tp = (f["high"] + f["low"] + f["close"]) / 3
    sma = tp.rolling(n).mean()
    md = (tp - sma).abs().rolling(n).mean()
    return (tp - sma) / (0.015 * md.replace(0.0, np.nan))


def _psar(f: pd.DataFrame, step: float, cap: float) -> pd.Series:
    """Wilder parabolic SAR: the stop-and-reverse level on each bar."""
    h, l = f["high"].to_numpy(float), f["low"].to_numpy(float)
    n = len(f)
    sar = np.full(n, np.nan)
    if n < 3:
        return pd.Series(sar, index=f.index)
    bull, af = True, step
    ep, cur = h[0], l[0]
    for i in range(1, n):
        cur = cur + af * (ep - cur)
        if bull:
            cur = min(cur, l[i - 1], l[max(0, i - 2)])
            if l[i] < cur:                       # flip to bear
                bull, cur, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, cap)
        else:
            cur = max(cur, h[i - 1], h[max(0, i - 2)])
            if h[i] > cur:                       # flip to bull
                bull, cur, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, cap)
        sar[i] = cur
    return pd.Series(sar, index=f.index)


def _supertrend(f: pd.DataFrame, n: int, k: float) -> pd.Series:
    """-> +1 while the trend is up, -1 while down."""
    a = _atr(f, n)
    hl2 = (f["high"] + f["low"]) / 2
    up, dn = (hl2 - k * a).to_numpy(float), (hl2 + k * a).to_numpy(float)
    c = f["close"].to_numpy(float)
    m = len(f)
    dirn = np.ones(m)
    fu, fl = up.copy(), dn.copy()
    for i in range(1, m):
        fu[i] = max(up[i], fu[i - 1]) if c[i - 1] > fu[i - 1] else up[i]
        fl[i] = min(dn[i], fl[i - 1]) if c[i - 1] < fl[i - 1] else dn[i]
        if c[i] > fl[i - 1]:
            dirn[i] = 1
        elif c[i] < fu[i - 1]:
            dirn[i] = -1
        else:
            dirn[i] = dirn[i - 1]
    return pd.Series(dirn, index=f.index)


def _cross_up(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _filter_mask(f: pd.DataFrame, scfg: dict) -> tuple[pd.Series, pd.Series]:
    """
    A SECOND indicator standing behind the trigger, as a pair of per-direction
    masks. Directional filters (a trend, a MACD sign) let longs through on
    their bull side and shorts on their bear side; regime filters (ADX, band
    width) gate both directions the same way.

    Every key is `filt_`-prefixed on purpose: sweep bundles are flattened into
    the strategy config, so an unprefixed `n` here would silently overwrite the
    TRIGGER's own n and the filter would look like it did something when all it
    did was retune the signal.
    -> (ok_long, ok_short)
    """
    kind = scfg.get("filt_kind", "none")
    idx = f.index
    yes = pd.Series(True, index=idx)
    if kind in ("none", None):
        return yes, yes
    close, a = f["close"], f["chart_atr"]
    n = int(scfg.get("filt_n", 200))
    v = float(scfg.get("filt_v", 20.0))
    fc = {"n_fast": int(scfg.get("filt_n_fast", 12)), "n_slow": int(scfg.get("filt_n_slow", 26)),
          "k": float(scfg.get("filt_k", 3.0))}
    if kind == "sma_trend":
        m = close.rolling(n).mean()
        return close > m, close < m
    if kind == "ema_trend":
        m = _ema(close, n)
        return close > m, close < m
    if kind == "macd":
        ml = _ema(close, fc["n_fast"]) - _ema(close, fc["n_slow"])
        return ml > 0, ml < 0
    if kind == "supertrend":
        st = _supertrend(f, n, fc["k"])
        return st > 0, st < 0
    if kind == "ema_stack":                       # the classic ribbon: 20 > 50 > 100 > 200
        e1, e2 = _ema(close, int(scfg.get("filt_n1", 20))), _ema(close, int(scfg.get("filt_n2", 50)))
        e3, e4 = _ema(close, int(scfg.get("filt_n3", 100))), _ema(close, int(scfg.get("filt_n4", 200)))
        up = (close > e1) & (e1 > e2) & (e2 > e3) & (e3 > e4)
        dn = (close < e1) & (e1 < e2) & (e2 < e3) & (e3 < e4)
        return up, dn
    if kind == "di":
        return f["plus_di"] > f["minus_di"], f["minus_di"] > f["plus_di"]
    if kind == "rsi50":
        r = _rsi(close, n)
        return r > 50, r < 50
    if kind == "cloud":
        nf, ns = fc["n_fast"], fc["n_slow"]
        conv = (f["high"].rolling(nf).max() + f["low"].rolling(nf).min()) / 2
        base = (f["high"].rolling(ns).max() + f["low"].rolling(ns).min()) / 2
        sa = ((conv + base) / 2).shift(ns)
        sb = ((f["high"].rolling(52).max() + f["low"].rolling(52).min()) / 2).shift(ns)
        return close > sa.combine(sb, max), close < sa.combine(sb, min)
    if kind in ("stoch_side", "stoch_50", "stoch_not_extended", "stoch_extended"):
        k_, d_ = _stoch(f, n, int(scfg.get("filt_n_signal", 3)))
        lo_, hi_ = float(scfg.get("filt_lo", 20.0)), float(scfg.get("filt_hi", 80.0))
        if kind == "stoch_side":                  # %K over %D = momentum agrees with the trade
            return k_ > d_, k_ < d_
        if kind == "stoch_50":                    # simply which half of the range we sit in
            return k_ > 50, k_ < 50
        if kind == "stoch_not_extended":          # refuse a break that is already stretched
            return k_ < hi_, k_ > lo_
        return k_ >= hi_, k_ <= lo_               # stoch_extended: only the stretched ones
    if kind in ("smi_side", "smi_zero", "smi_not_extended", "smi_extended"):
        smi, sg = _smi(f, n, int(scfg.get("filt_r", 3)), int(scfg.get("filt_s", 3)),
                       int(scfg.get("filt_n_signal", 3)))
        lim = float(scfg.get("filt_v", 40.0))
        if kind == "smi_side":                    # SMI above its signal line = momentum agrees
            return smi > sg, smi < sg
        if kind == "smi_zero":                    # closing in the upper half of the range
            return smi > 0, smi < 0
        if kind == "smi_not_extended":            # refuse an already-stretched break
            return smi < lim, smi > -lim
        return smi >= lim, smi <= -lim            # smi_extended
    if kind in ("vwap_side", "vwap_roll_side"):    # which side of the anchor we sit on
        line = f["vwap_sess"] if kind == "vwap_side" else f["vwap_roll"]
        return close > line, close < line
    if kind in ("vwap_not_extended", "vwap_extended"):
        z = (f["vwap_sess_z"] if scfg.get("filt_anchor", "sess") == "sess"
             else f["vwap_roll_z"]).abs()
        ok = (z < v) if kind == "vwap_not_extended" else (z >= v)
        return ok, ok
    if kind == "rel_vol_min":                     # the volume gate; NaN on a feed without volume
        ok = f["rel_vol"] >= v
        return ok, ok
    if kind == "adx_min":                         # only trade when a trend is running
        ok = f["adx"] > v
        return ok, ok
    if kind == "adx_max":                         # only trade when the market is ranging
        ok = f["adx"] < v
        return ok, ok
    if kind == "atr_min":                         # only trade when the bar is moving
        ok = a > a.rolling(n).mean() * fc["k"]
        return ok, ok
    if kind == "atr_max":                         # only trade when it is quiet
        ok = a < a.rolling(n).mean() * fc["k"]
        return ok, ok
    raise ValueError(f"indicator_lab filter {kind}")


def _const(v: float, idx) -> pd.Series:
    return pd.Series(v, index=idx, dtype=float)


def _system_signals(f: pd.DataFrame, fs: pd.DataFrame, scfg: dict
                    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    One system's raw signals before the filter and the gates:
    (long_setup, short_setup, exit_long, exit_short). `fs` is the signal-source
    frame (Heikin Ashi when use_ha); `f` is the real one, for its extra columns.
    """
    sysname = scfg.get("system", "ma_cross")
    nf, ns = int(scfg.get("n_fast", 12)), int(scfg.get("n_slow", 26))
    n = int(scfg.get("n", 14))
    lo, hi = float(scfg.get("lo", 30.0)), float(scfg.get("hi", 70.0))
    kb = float(scfg.get("k_band", 2.0))
    kind = scfg.get("ma_kind", "ema")
    close, high, low = fs["close"], fs["high"], fs["low"]
    a = f["chart_atr"]
    idx = f.index
    state: pd.Series | None = None              # +1/-1 regime systems use this

    if sysname == "ma_cross":
        state = np.sign(_ma(close, nf, kind) - _ma(close, ns, kind))
    elif sysname in ("macd_cross", "macd_zero"):
        ml = _ema(close, nf) - _ema(close, ns)
        ref = _ema(ml, int(scfg.get("n_signal", 9))) if sysname == "macd_cross" else ml * 0
        state = np.sign(ml - ref)
    elif sysname == "rsi_revert":
        r = _rsi(close, n)
        long_setup, short_setup = _cross_up(_const(lo, idx), r), _cross_up(r, _const(hi, idx))
        exit_l, exit_s = r > 50, r < 50
    elif sysname == "rsi_trend":
        state = np.sign(_rsi(close, n) - 50)
    elif sysname == "stoch":
        k, d = _stoch(fs, n, int(scfg.get("n_signal", 3)), int(scfg.get("smooth", 3)))
        long_setup = _cross_up(k, d) & (d < lo)
        short_setup = _cross_up(d, k) & (d > hi)
        exit_l, exit_s = k > hi, k < lo
    elif sysname in ("bb_revert", "bb_break"):
        mid = close.rolling(n).mean()
        sd = close.rolling(n).std()
        ub, lb = mid + kb * sd, mid - kb * sd
        if sysname == "bb_revert":
            long_setup, short_setup = _cross_up(lb, close), _cross_up(close, ub)
            exit_l, exit_s = close > mid, close < mid
        else:
            long_setup, short_setup = _cross_up(close, ub), _cross_up(lb, close)
            exit_l, exit_s = close < mid, close > mid
    elif sysname == "keltner_break":
        mid = _ema(close, n)
        ub, lb = mid + kb * a, mid - kb * a
        long_setup, short_setup = _cross_up(close, ub), _cross_up(lb, close)
        exit_l, exit_s = close < mid, close > mid
    elif sysname == "cci":
        c = _cci(fs, n)
        long_setup = _cross_up(c, _const(100.0, idx))
        short_setup = _cross_up(_const(-100.0, idx), c)
        exit_l, exit_s = c < 0, c > 0
    elif sysname == "williams_r":
        hh, ll = high.rolling(n).max(), low.rolling(n).min()
        wr = -100 * (hh - close) / (hh - ll).replace(0.0, np.nan)
        long_setup = _cross_up(wr, _const(-80.0, idx))
        short_setup = _cross_up(_const(-20.0, idx), wr)
        exit_l, exit_s = wr > -20, wr < -80
    elif sysname == "psar":
        state = np.sign(close - _psar(fs, float(scfg.get("psar_step", 0.02)), float(scfg.get("psar_cap", 0.2))))
    elif sysname == "supertrend":
        state = _supertrend(fs, n, kb)
    elif sysname == "ichimoku":
        conv = (high.rolling(nf).max() + low.rolling(nf).min()) / 2
        base = (high.rolling(ns).max() + low.rolling(ns).min()) / 2
        sa = ((conv + base) / 2).shift(ns)
        nb = int(scfg.get("n_spanb", 52))              # Senkou Span B length (was fixed at 52)
        sb = ((high.rolling(nb).max() + low.rolling(nb).min()) / 2).shift(ns)
        long_setup = _cross_up(conv, base) & (close > sa.combine(sb, max))
        short_setup = _cross_up(base, conv) & (close < sa.combine(sb, min))
        exit_l, exit_s = conv < base, conv > base
    elif sysname == "adx_di":
        amin = float(scfg.get("adx_min", 20))
        long_setup = _cross_up(f["plus_di"], f["minus_di"]) & (f["adx"] > amin)
        short_setup = _cross_up(f["minus_di"], f["plus_di"]) & (f["adx"] > amin)
        exit_l, exit_s = f["plus_di"] < f["minus_di"], f["minus_di"] < f["plus_di"]
    elif sysname == "aroon":
        up = 100 * (n - high.rolling(n + 1).apply(lambda x: len(x) - 1 - int(np.argmax(x)), raw=True)) / n
        dn = 100 * (n - low.rolling(n + 1).apply(lambda x: len(x) - 1 - int(np.argmin(x)), raw=True)) / n
        state = np.sign(up - dn)
    elif sysname == "smi_cross":
        smi, sg = _smi(fs, n, int(scfg.get("n_fast", 3)), int(scfg.get("n_slow", 3)),
                       int(scfg.get("n_signal", 3)))
        state = np.sign(smi - sg)
    elif sysname == "smi_revert":
        smi, _sg = _smi(fs, n, int(scfg.get("n_fast", 3)), int(scfg.get("n_slow", 3)))
        long_setup = _cross_up(_const(-abs(lo), idx), smi)
        short_setup = _cross_up(smi, _const(abs(hi), idx))
        exit_l, exit_s = smi > 0, smi < 0
    elif sysname == "donchian":
        # the turtle: break the n-bar extreme, leave on the opposite exit_n extreme.
        # shift(1) so the channel is the bars BEFORE this one, never including it.
        xn = int(scfg.get("exit_n", max(2, n // 2)))
        hh, ll = high.rolling(n).max().shift(1), low.rolling(n).min().shift(1)
        xh, xl = high.rolling(xn).max().shift(1), low.rolling(xn).min().shift(1)
        long_setup = (close > hh) & (close.shift(1) <= hh)
        short_setup = (close < ll) & (close.shift(1) >= ll)
        exit_l, exit_s = close < xl, close > xh
    elif sysname in ("vwap_revert", "vwap_break", "vwap_roll_revert", "vwap_roll_break"):
        roll = sysname.startswith("vwap_roll")
        line = f["vwap_roll"] if roll else f["vwap_sess"]
        sd = f["vwap_roll_sd"] if roll else f["vwap_sess_sd"]
        ub, lb = line + kb * sd, line - kb * sd
        warm = pd.Series(True, index=idx) if roll else f["vwap_warm"]
        if sysname.endswith("revert"):
            long_setup, short_setup = _cross_up(lb, close), _cross_up(close, ub)
            exit_l, exit_s = close > line, close < line
        else:
            long_setup, short_setup = _cross_up(close, ub), _cross_up(lb, close)
            exit_l, exit_s = close < line, close > line
        long_setup, short_setup = long_setup & warm, short_setup & warm
    elif sysname == "vwap_cross":
        state = np.sign(close - f["vwap_roll"])
    elif sysname == "confluence":                 # the user's indicators agreeing (strategies/confluence.py)
        from .confluence import agreement, need_for
        o = scfg.get("confluence", {})
        bull, bear, n_in = agreement(fs, o)
        need = need_for(o, n_in)
        state = pd.Series(np.where(bull >= need, 1.0, np.where(bear >= need, -1.0, 0.0)), index=idx)
        state = state.where(bull.notna())
    elif sysname == "madh":                       # Ehlers MADH zero cross; n_fast=short, n_slow=cycle
        state = np.sign(_madh(close, nf, ns))
    elif sysname == "cmf":                        # Chaikin Money Flow crossing cmf_level (volume only)
        state = np.sign(_cmf(fs, n) - float(scfg.get("cmf_level", 0.0)))
    elif sysname == "ribbon_stack":               # the pivot ribbon stacking: fast > pivot > slow
        e1, e2, e3 = _ema(close, nf), _ema(close, int(scfg.get("n_mid", 21))), _ema(close, ns)
        state = pd.Series(np.where((e1 > e2) & (e2 > e3), 1.0,
                                   np.where((e1 < e2) & (e2 < e3), -1.0, 0.0)), index=idx)
    elif sysname == "roc":
        state = np.sign(close - close.shift(n))
    elif sysname == "custom":
        # A user indicator (gui/custom_indicators/): custom_fn(frame, params) returns
        # +1 (want long), -1 (want short) or 0 (flat) per bar. Unlike the regime
        # systems above, 0 means flat, so it exits instead of holding the last side.
        out = scfg["custom_fn"](fs.copy(), dict(scfg.get("custom_params", {})))
        raw = out.reindex(idx) if isinstance(out, pd.Series) else pd.Series(np.asarray(out, dtype=float), index=idx)
        want = pd.Series(np.sign(raw.astype(float)), index=idx).fillna(0.0)
        prev = want.shift(1).fillna(0.0)
        long_setup, short_setup = (want > 0) & (prev <= 0), (want < 0) & (prev >= 0)
        exit_l, exit_s = want <= 0, want >= 0
    else:
        raise ValueError(f"indicator_lab system {sysname}")

    if state is not None:                        # regime system: trade every flip
        st = pd.Series(np.asarray(state, dtype=float), index=idx).replace(0.0, np.nan).ffill()
        long_setup = (st > 0) & (st.shift(1) <= 0)
        short_setup = (st < 0) & (st.shift(1) >= 0)
        exit_l, exit_s = st < 0, st > 0
    return long_setup, short_setup, exit_l, exit_s


def _hold(long_setup: pd.Series, short_setup: pd.Series, exit_l: pd.Series, exit_s: pd.Series) -> pd.Series:
    """
    A system's events turned into the side it wants on every bar: +1, -1 or 0.
    A setup takes that side; that side's exit (or the opposite setup) leaves it.
    Regime systems come out as their state; event systems (an RSI cross, a
    band touch) hold from the signal until their own exit.
    """
    ls, ss = long_setup.fillna(False).to_numpy(bool), short_setup.fillna(False).to_numpy(bool)
    el, es = exit_l.fillna(False).to_numpy(bool), exit_s.fillna(False).to_numpy(bool)
    out = np.zeros(len(ls))
    held = 0.0
    for i in range(len(ls)):
        if (held > 0 and el[i]) or (held < 0 and es[i]):
            held = 0.0
        if ls[i]:
            held = 1.0
        elif ss[i]:
            held = -1.0
        out[i] = held
    return pd.Series(out, index=long_setup.index)


def _combo_signals(f: pd.DataFrame, fs: pd.DataFrame, scfg: dict
                   ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Up to five systems voting together. `combo` is a list of legs, each a dict
    with `system` plus its own settings (or custom_fn / custom_params), laid
    over the lab defaults. combo_mode:
        all       hold a side only while every leg holds it
        at_least  hold a side while combo_need legs hold it, and more than hold the other
        trigger   the first leg's own entries and exits, taken only when every
                  other leg already holds that side
    """
    legs = scfg.get("combo", [])
    if not 1 <= len(legs) <= 5:
        raise ValueError(f"indicator_lab combo needs 1 to 5 legs, got {len(legs)}")
    base = {k: v for k, v in scfg.items() if k not in ("system", "combo", "combo_mode", "combo_need")}
    raw = [_system_signals(f, fs, {**base, **leg}) for leg in legs]
    held = pd.concat([_hold(*r) for r in raw], axis=1)
    mode = scfg.get("combo_mode", "all")
    if mode == "trigger":
        rest = held.iloc[:, 1:]
        ls, ss, el, es = raw[0]
        return ls.fillna(False) & (rest > 0).all(axis=1), ss.fillna(False) & (rest < 0).all(axis=1), el, es
    if mode == "all":
        want = np.where((held > 0).all(axis=1), 1.0, np.where((held < 0).all(axis=1), -1.0, 0.0))
    elif mode == "at_least":
        need = int(scfg.get("combo_need", len(legs)))
        up, dn = (held > 0).sum(axis=1), (held < 0).sum(axis=1)
        want = np.where((up >= need) & (up > dn), 1.0, np.where((dn >= need) & (dn > up), -1.0, 0.0))
    else:
        raise ValueError(f"indicator_lab combo_mode {mode}")
    want = pd.Series(want, index=f.index)
    prev = want.shift(1).fillna(0.0)
    # 0 = the legs no longer agree, which exits like a flip would
    return (want > 0) & (prev <= 0), (want < 0) & (prev >= 0), want <= 0, want >= 0


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    # px is the REAL close and is the only thing fills, stops and targets may
    # use. The signal source below is what use_ha swaps for Heikin Ashi -
    # synthetic prices that are fine to reason about and impossible to trade at.
    px = f["close"]
    if scfg.get("use_ha", False):
        # helpers take a FRAME, so they need the swap too - otherwise a system
        # built on one of them silently ignores use_ha and reports a delta of
        # exactly zero.
        fs = f.assign(open=f["ha_open"], high=f["ha_high"], low=f["ha_low"], close=f["ha_close"])
    else:
        fs = f
    a = f["chart_atr"]
    idx = f.index
    if scfg.get("system") == "combo":
        long_setup, short_setup, exit_l, exit_s = _combo_signals(f, fs, scfg)
    else:
        long_setup, short_setup, exit_l, exit_s = _system_signals(f, fs, scfg)

    ok_l, ok_s = _filter_mask(f, scfg)
    long_setup = long_setup & ok_l.fillna(False)
    short_setup = short_setup & ok_s.fillna(False)

    sig = pd.DataFrame(index=idx)
    sig["entry_level_up"], sig["entry_level_down"] = px, px
    finish_direction(sig, f, 1, long_setup.fillna(False), scfg, acfg, cfg, gates_enabled)
    finish_direction(sig, f, -1, short_setup.fillna(False), scfg, acfg, cfg, gates_enabled)
    if scfg.get("exit_on_flip", True):
        sig["long_exit"], sig["short_exit"] = exit_l.fillna(False), exit_s.fillna(False)
    k_stop = float(scfg.get("stop_atr_k", 2.0))
    sig["stop_mode"] = "atr"
    sig["long_stop"], sig["short_stop"] = px - k_stop * a, px + k_stop * a
    far = 99.0 * a
    sig["long_target"], sig["short_target"] = px + far, px - far
    sig["long_runner"], sig["short_runner"] = sig["long_target"], sig["short_target"]
    sig["long_trigger"], sig["short_trigger"] = sig["long_stop"], sig["short_stop"]
    return sig
