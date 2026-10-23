"""
Gate framework.

A gate is a function (features, direction, scfg, acfg, cfg) -> (ok, reason)
where `ok` is a boolean Series over the feature frame and `reason` is a
string Series (why the gate failed on that bar, '' when it passed). A
strategy is a setup rule plus a list of gate names; every gate can be
switched off in config and the trade log carries every gate's verdict at
entry, so the ablation table in Phase 4 is a config sweep, not new code.

direction: +1 long, -1 short.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd

GateFn = Callable[[pd.DataFrame, int, dict, dict, dict], tuple[pd.Series, pd.Series]]
GATES: dict[str, GateFn] = {}


def gate(name: str):
    def deco(fn: GateFn) -> GateFn:
        GATES[name] = fn
        return fn
    return deco


def _reason(ok: pd.Series, text: str) -> pd.Series:
    return pd.Series(np.where(ok, "", text), index=ok.index)


# ---------------------------------------------------------------------------
# the gates Saty's doctrine uses
# ---------------------------------------------------------------------------

@gate("ribbon")
def g_ribbon(f, d, scfg, acfg, cfg):
    ok = (f["ribbon_bull"] if d > 0 else f["ribbon_bear"]) & f["ribbon_warm"]
    if scfg.get("reject_flat_ribbon", True):
        ok &= ~f["ribbon_flat"]
    return ok, _reason(ok, "ribbon_disagrees")


@gate("time_warp")
def g_time_warp(f, d, scfg, acfg, cfg):
    col = "htf_ribbon_bull" if d > 0 else "htf_ribbon_bear"
    if col not in f:
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    ok = f[col].astype(bool)
    return ok, _reason(ok, "htf_ribbon_disagrees")


@gate("time_of_day")
def g_time_of_day(f, d, scfg, acfg, cfg):
    tod = scfg.get("time_of_day", {})
    kind = acfg["session"]["kind"]
    if kind == "rth_us":
        hours = float(tod.get("rth_first_hours", 2))
        offset = (f.index - pd.DatetimeIndex(f["day_open"])).total_seconds() / 3600.0
        ok = pd.Series(offset < hours, index=f.index)
        return ok, _reason(ok, "late_session")
    win = tod.get("fx_window_utc") if kind == "fx" else tod.get("crypto_window_utc")
    if not win:
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    h0 = int(win[0][:2]) * 60 + int(win[0][3:5])
    h1 = int(win[1][:2]) * 60 + int(win[1][3:5])
    m = f.index.hour * 60 + f.index.minute
    ok = pd.Series((m >= h0) & (m < h1), index=f.index)
    return ok, _reason(ok, "outside_window")


@gate("volume")
def g_volume(f, d, scfg, acfg, cfg):
    want = "buy" if d > 0 else "sell"
    winner_ok = f["stack_winner"] == want
    relmin = float(scfg.get("rel_vol_min", cfg["indicators"]["volume_stack"].get("conviction_mult", 1.5)))
    rel_ok = f["rel_vol"] >= relmin
    mode = acfg.get("volume_gate", "hard")
    avail = f["vol_available"] & f["rel_vol"].notna()
    if mode == "off":
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    if mode == "advisory":
        # winner must agree (pure geometry, always available); rel-vol is logged only
        ok = winner_ok
        reason = np.where(ok, "", "stack_disagrees")
        return ok, pd.Series(reason, index=f.index)
    # hard: winner + rel-vol. Bars without a volume reading (no feed, proxy
    # not covering the bar) either pass on the stack winner alone or block,
    # per `volume_unavailable` - the trade log records which it was.
    unavailable_passes = acfg.get("volume_unavailable", "pass") == "pass"
    ok = winner_ok & np.where(avail, rel_ok, unavailable_passes)
    # unavailable_passes is a plain Python bool: `~True` is -2, not False, so it
    # must be negated with `not` (the `~` form only worked by bit-level luck)
    reason = np.select([~winner_ok, ~avail & (not unavailable_passes), avail & ~rel_ok],
                       ["stack_disagrees", "no_volume", "rel_vol_low"], "")
    return ok, pd.Series(reason, index=f.index)


@gate("oscillator")
def g_oscillator(f, d, scfg, acfg, cfg):
    o = scfg.get("oscillator", {})
    hi = float(o.get("reject_long_above", 61.8))
    lo = float(o.get("reject_short_below", -61.8))
    osc = f["osc"]
    if d > 0:
        ok = osc <= hi
        if o.get("require_rising"):
            ok &= (osc >= -23.6) & f["osc_rising"]
        reason = "osc_extended"
    else:
        ok = osc >= lo
        if o.get("require_rising"):
            ok &= (osc <= 23.6) & ~f["osc_rising"]
        reason = "osc_extended"
    ok &= osc.notna()
    return ok, _reason(ok, reason)


@gate("range_util")
def g_range_util(f, d, scfg, acfg, cfg):
    mx = float(scfg.get("range_util_max", 0.8))
    # utilisation BEFORE this bar's move: the bar that crosses the level
    # inflates the range itself, so look at the prior bar's reading
    prev = f["range_util"].shift(1).where(f["day_key"] == f["day_key"].shift(1), 0.0)
    ok = prev <= mx
    return ok, _reason(ok, "range_spent")


_GATE_PARAM_KEYS = ("reject_flat_ribbon", "time_of_day", "rel_vol_min", "oscillator", "range_util_max", "adx", "fisher", "macd", "rsi")
_CACHE: dict = {}


def _gate_values(f: pd.DataFrame, direction: int, scfg: dict, acfg: dict, cfg: dict) -> dict:
    """
    Raw per-gate verdicts, cached per feature frame. The verdicts depend on
    the features and the gate parameters, not on which gates are switched
    on, so a sweep over gate toggles or exit rules reuses them.
    """
    key = (id(f), len(f), str(f.index[-1]), direction, acfg["name"], acfg.get("volume_gate"),
           acfg.get("volume_unavailable"), json.dumps({k: scfg.get(k) for k in _GATE_PARAM_KEYS}, sort_keys=True, default=str))
    hit = _CACHE.get(key)
    if hit is not None:
        return hit
    vals = {}
    for name, fn in GATES.items():
        ok, reason = fn(f, direction, scfg, acfg, cfg)
        vals[name] = (ok.fillna(False).to_numpy(dtype=bool), reason.to_numpy(dtype=object))
    if len(_CACHE) > 64:
        _CACHE.clear()
    _CACHE[key] = vals
    return vals


LADDER = [-1.618, -1.236, -1.0, -0.786, -0.618, -0.5, -0.382, -0.236, 0.0,
          0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.236, 1.618]


def next_levels(f: pd.DataFrame, direction: int, min_gap_atr: float = 0.1) -> tuple[pd.Series, pd.Series]:
    """
    The nearest ATR-ladder level (PDC included) beyond the close in the
    trade direction, and the one after it. Used as target and runner by
    strategies whose target is "the next level".
    """
    pdc, atr, close = f["pdc"].to_numpy(), f["atr"].to_numpy(), f["close"].to_numpy()
    grid_ = pdc[:, None] + np.array(LADDER)[None, :] * atr[:, None]
    if direction > 0:
        ok = grid_ > (close + min_gap_atr * atr)[:, None]
        cand = np.where(ok, grid_, np.inf)
        first = cand.min(axis=1)
        cand2 = np.where(cand > first[:, None], cand, np.inf)
        second = cand2.min(axis=1)
    else:
        ok = grid_ < (close - min_gap_atr * atr)[:, None]
        cand = np.where(ok, grid_, -np.inf)
        first = cand.max(axis=1)
        cand2 = np.where(cand < first[:, None], cand, -np.inf)
        second = cand2.max(axis=1)
    first = np.where(np.isfinite(first), first, np.nan)
    second = np.where(np.isfinite(second), second, np.nan)
    return pd.Series(first, index=f.index), pd.Series(second, index=f.index)


def finish_direction(sig: pd.DataFrame, f: pd.DataFrame, direction: int, setup: pd.Series,
                     scfg: dict, acfg: dict, cfg: dict, gates_enabled: dict | None = None) -> None:
    """Fill the standard per-direction signal columns from a setup series."""
    name = "long" if direction > 0 else "short"
    setup = setup.fillna(False).astype(bool) & f["levels_valid"]
    g = evaluate_gates(f, direction, scfg, acfg, cfg, gates_enabled)
    sig[f"{name}_setup"] = setup
    sig[f"{name}_entry"] = setup & g["gates_ok"]
    sig[f"{name}_block"] = g["block_reason"]
    sig[f"{name}_gap"] = False
    for c in g.columns:
        if c.startswith("gate_") or c.startswith("why_"):
            sig[f"{name}_{c}"] = g[c]


@gate("rsi")
def g_rsi(f, d, scfg, acfg, cfg):
    if "rsi" not in f:
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    o = scfg.get("rsi", {})
    mode = o.get("mode", "not_extended")
    mx = float(o.get("max", 70.0))
    r = f["rsi"]
    if mode == "above_50":                 # momentum on the trade's side
        ok = (r > 50) if d > 0 else (r < 50)
    elif mode == "not_extended":           # don't buy an already-stretched move
        ok = (r < mx) if d > 0 else (r > 100 - mx)
    elif mode == "extended":               # only buy a stretched move (the opposite)
        ok = (r >= mx) if d > 0 else (r <= 100 - mx)
    elif mode == "rising":
        ok = (r > r.shift(1)) if d > 0 else (r < r.shift(1))
    else:
        raise ValueError(f"rsi mode {mode}")
    ok &= f["rsi_warm"]
    return ok, _reason(ok, f"rsi_{mode}")


@gate("macd")
def g_macd(f, d, scfg, acfg, cfg):
    if "macd" not in f:
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    o = scfg.get("macd", {})
    mode = o.get("mode", "above_signal")
    m, sg, h = f["macd"], f["macd_signal"], f["macd_hist"]
    if mode == "above_zero":               # MACD on the trade's side of zero
        ok = (m > 0) if d > 0 else (m < 0)
    elif mode == "above_signal":           # MACD above its signal line
        ok = (m > sg) if d > 0 else (m < sg)
    elif mode == "hist_rising":            # histogram expanding in the trade's direction
        ok = (h > h.shift(1)) if d > 0 else (h < h.shift(1))
    elif mode == "zero_and_signal":
        ok = ((m > 0) & (m > sg)) if d > 0 else ((m < 0) & (m < sg))
    else:
        raise ValueError(f"macd mode {mode}")
    ok &= f["macd_warm"]
    return ok, _reason(ok, f"macd_{mode}")


@gate("adx")
def g_adx(f, d, scfg, acfg, cfg):
    if "adx" not in f:                     # frame built without the trend indicators
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    o = scfg.get("adx", {})
    mn = float(o.get("min", 20.0))
    ok = f["adx"] > mn
    if o.get("rising", False):
        ok &= f["adx_rising"]
    if o.get("di_direction", False):
        ok &= (f["plus_di"] > f["minus_di"]) if d > 0 else (f["minus_di"] > f["plus_di"])
    ok &= f["adx"].notna()
    return ok, _reason(ok, "adx_weak")


@gate("fisher")
def g_fisher(f, d, scfg, acfg, cfg):
    if "fisher" not in f:
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    o = scfg.get("fisher", {})
    mode = o.get("mode", "turn_up")
    fi, tr = f["fisher"], f["fisher_trigger"]
    mx = float(o.get("max_abs", 1.5))
    if mode == "turn_up":                 # Fisher above its trigger for longs, below for shorts
        ok = (fi > tr) if d > 0 else (fi < tr)
    elif mode == "above_zero":            # Fisher on the trade's side of zero
        ok = (fi > 0) if d > 0 else (fi < 0)
    elif mode == "not_extended":          # not buying a peak swing
        ok = (fi < mx) if d > 0 else (fi > -mx)
    elif mode == "turn_up_not_extended":
        ok = ((fi > tr) & (fi < mx)) if d > 0 else ((fi < tr) & (fi > -mx))
    elif mode == "zero_not_extended":
        ok = ((fi > 0) & (fi < mx)) if d > 0 else ((fi < 0) & (fi > -mx))
    else:
        raise ValueError(f"fisher mode {mode}")
    ok &= f["fisher_warm"]
    return ok, _reason(ok, f"fisher_{mode}")


@gate("voss")
def g_voss(f, d, scfg, acfg, cfg):
    """Ehlers Voss agrees. up (default): predictor above the filter AND both
    lines above zero - "dots green, crossed above the 0 line". state: predictor
    above the filter only. recent_cross: a fresh cross within `within` bars."""
    if "voss" not in f:
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    o = scfg.get("voss", {})
    mode = o.get("mode", "up")
    if mode == "up":
        ok = f["voss_up"] if d > 0 else f["voss_down"]
    elif mode == "state":
        ok = f["voss_bull"] if d > 0 else f["voss_bear"]
    elif mode == "recent_cross":
        col = f["voss_cross_up"] if d > 0 else f["voss_cross_down"]
        ok = col.astype(int).rolling(int(o.get("within", 3)), min_periods=1).max().astype(bool)
        ok &= f["voss_bull"] if d > 0 else f["voss_bear"]
    else:
        raise ValueError(f"voss mode {mode}")
    ok = ok.fillna(False).astype(bool)
    return ok, _reason(ok, f"voss_{mode}")


@gate("phase")
def g_phase(f, d, scfg, acfg, cfg):
    """Saty Phase compression read. release (default): a compression ended
    with momentum turning our way within `within` bars, and momentum still
    agrees now - magenta followed by green. green: not compressed and momentum
    agreeing. not_compressed: just not magenta."""
    if "compression" not in f:
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    o = scfg.get("phase", {})
    mode = o.get("mode", "release")
    comp = f["compression"].astype(bool)
    agree = f["osc_rising"].astype(bool) if d > 0 else ~f["osc_rising"].astype(bool)
    if mode == "release":
        rel = f["compression_exit_bull"] if d > 0 else f["compression_exit_bear"]
        recent = rel.astype(int).rolling(int(o.get("within", 6)), min_periods=1).max().astype(bool)
        ok = recent & ~comp & agree
    elif mode == "green":
        ok = ~comp & agree
    elif mode == "not_compressed":
        ok = ~comp
    else:
        raise ValueError(f"phase mode {mode}")
    ok = ok.fillna(False).astype(bool)
    return ok, _reason(ok, f"phase_{mode}")


@gate("confluence")
def g_confluence(f, d, scfg, acfg, cfg):
    """The user's indicators agree with the trade (strategies/confluence.py).
    min_agree: how many must agree; default all of those in play."""
    o = scfg.get("confluence", {})
    if not o:                                    # not configured: pass through, skip the work
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    from .confluence import agreement, need_for
    bull, bear, n = agreement(f, o)
    need = need_for(o, n)
    ok = ((bull >= need) if d > 0 else (bear >= need)).fillna(False).astype(bool)
    return ok, _reason(ok, f"confluence_lt_{need}_of_{n}")


@gate("daily_trend")
def g_daily_trend(f, d, scfg, acfg, cfg):
    """Pre-trade rule: trade with the daily trend - yesterday's close above
    (longs) or below (shorts) its 21-period EMA on the level timeframe."""
    if "daily_trend" not in f:
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    ok = (f["daily_trend"] > 0) if d > 0 else (f["daily_trend"] < 0)
    ok = ok.fillna(False).astype(bool)
    return ok, _reason(ok, "against_daily_trend")


@gate("pd_room")
def g_pd_room(f, d, scfg, acfg, cfg):
    """Pre-trade S/R: don't buy straight into yesterday's high. A long is
    blocked when the prior-day high sits between price and the 61.8 target
    (resistance before the target); shorts mirror with the prior-day low."""
    if "pdh" not in f or "target_up" not in f:
        ok = pd.Series(True, index=f.index)
        return ok, _reason(ok, "")
    c = f["close"]
    if d > 0:
        blocked = (f["pdh"] > c) & (f["pdh"] < f["target_up"])
    else:
        blocked = (f["pdl"] < c) & (f["pdl"] > f["target_down"])
    ok = ~blocked.fillna(False)
    return ok, _reason(ok, "pd_level_in_the_way")


def evaluate_gates(f: pd.DataFrame, direction: int, scfg: dict, acfg: dict, cfg: dict,
                   enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    """One bool column per gate plus `gates_ok` and `block_reason`."""
    enabled = enabled if enabled is not None else scfg.get("gates", {})
    vals = _gate_values(f, direction, scfg, acfg, cfg)
    n = len(f)
    all_ok = np.ones(n, dtype=bool)
    block = np.full(n, "", dtype=object)
    cols = {}
    for name, (ok, reason) in vals.items():
        cols[f"gate_{name}"] = ok
        cols[f"why_{name}"] = reason
        if enabled.get(name, False):
            newly = all_ok & ~ok
            block[newly] = name
            all_ok &= ok
    cols["gates_ok"] = all_ok
    cols["block_reason"] = block
    return pd.DataFrame(cols, index=f.index)
