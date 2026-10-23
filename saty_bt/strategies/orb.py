"""
Strategy O - Opening Range Breakout (and its cousins).

The best-documented intraday index trade in the public record: define the
range of the first N bars of the session, go with the break of it, stop on
the other side (or the midpoint), hold to a multiple of the risk or to the
close (Zarattini & Aziz 2023 on QQQ; Larry Williams' volatility breakout
before that). The FX version anchors the range at the London open instead
of the day open.

mode:
    range        : fresh close beyond the opening range high/low (default)
    first_candle : direction = sign of the opening range's own candle; enter
                   at the close of the range (the Zarattini form)
    open_atr     : Williams-style level = day open +/- open_atr_k x ATR(14)
range_start  : bars after the day open where the range starts (0 = the open;
               FX London open on a 22:00 UTC day = 18 bars of 30m)
range_bars   : bars in the range
entry_window : take the break only within this many bars after the range
               (0 = any time in the session)
invert       : fade the break rather than follow it (long the down-break).
               Use stop atr or excursion - the box-edge stops sit on the wrong
               side of a fade entry.
stop         : range_opp | range_mid | range_frac (stop_range_frac of the box
               in from the broken edge) | pdc | atr (stop_atr_k x ATR below entry)
bias         : none | pdc (longs only when the range end is above the previous
               close, shorts below)  |  first_candle (direction of the day's first bar)
Targets come from the engine: target_r (R multiple) or nothing (hold to the
period end). entry_level_up/down carry the range edges so entry_fill: limit
rests at the range edge if wanted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import finish_direction
from .indicator_lab import _exit_mask, _filter_mask


def build_signals(f: pd.DataFrame, scfg: dict, acfg: dict, cfg: dict,
                  gates_enabled: dict[str, bool] | None = None) -> pd.DataFrame:
    mode = scfg.get("mode", "range")
    start = int(scfg.get("range_start", 0))
    n = int(scfg.get("range_bars", 1))
    window = int(scfg.get("entry_window", 0))
    stop_mode = scfg.get("stop", "range_opp")
    bias = scfg.get("bias", "none")
    k_open = float(scfg.get("open_atr_k", 0.5))
    k_stop = float(scfg.get("stop_atr_k", 0.5))

    day = f["day_key"]
    idx_in_day = f.groupby(day).cumcount()
    in_range = (idx_in_day >= start) & (idx_in_day < start + n)
    src = scfg.get("range_source", "hl")          # hl = wicks included | close = closing prices only
    if src == "close":
        rh = f["close"].where(in_range).groupby(day).transform("max")
        rl = f["close"].where(in_range).groupby(day).transform("min")
    elif src == "hl":
        rh = f["high"].where(in_range).groupby(day).transform("max")
        rl = f["low"].where(in_range).groupby(day).transform("min")
    else:
        raise ValueError(f"orb range_source {src}")
    r_open = f["open"].where(idx_in_day == start).groupby(day).transform("max")
    r_close = f["close"].where(idx_in_day == start + n - 1).groupby(day).transform("max")
    day_open_px = f["open"].where(idx_in_day == 0).groupby(day).transform("max")
    delay = int(scfg.get("entry_delay", 0))       # ignore breaks for this many bars after the range
    after = idx_in_day >= start + n + delay
    if window > 0:
        after &= idx_in_day < start + n + window    # window is still counted from the range end
    valid = after & rh.notna() & rl.notna() & f["levels_valid"]

    # The BOX stays on real prices - the stop sits on its edge, so it has to be
    # a price that exists. use_ha only swaps what counts as "closed outside
    # the box": a Heikin Ashi close is the bar's average, so it needs most of
    # the bar beyond the edge, not just the tip.
    px, atr, pdc = f["close"], f["atr"], f["pdc"]
    close = f["ha_close"] if scfg.get("use_ha", False) else px
    if mode == "range":
        long_setup = valid & (close > rh) & (close.shift(1) <= rh)
        short_setup = valid & (close < rl) & (close.shift(1) >= rl)
        lvl_up, lvl_dn = rh, rl
    elif mode == "first_candle":
        at_end = (idx_in_day == start + n - 1) & rh.notna() & f["levels_valid"]
        long_setup = at_end & (r_close > r_open)
        short_setup = at_end & (r_close < r_open)
        lvl_up, lvl_dn = rh, rl
    elif mode == "reentry":
        # the failed-break fade: price closed OUTSIDE the range, then closes back
        # INSIDE it. Trade against the original break.
        back_in_up = (close <= rh) & (close.shift(1) > rh)      # failed upside break -> short
        back_in_dn = (close >= rl) & (close.shift(1) < rl)      # failed downside break -> long
        short_setup = valid & back_in_up
        long_setup = valid & back_in_dn
        lvl_up, lvl_dn = rh, rl
    elif mode == "open_atr":
        lvl_up, lvl_dn = day_open_px + k_open * atr, day_open_px - k_open * atr
        v = (idx_in_day >= 1) & f["levels_valid"] & lvl_up.notna()
        long_setup = v & (close > lvl_up) & (close.shift(1) <= lvl_up)
        short_setup = v & (close < lvl_dn) & (close.shift(1) >= lvl_dn)
    else:
        raise ValueError(f"orb mode {mode}")

    if scfg.get("invert", False):
        # fade the break instead of going with it: the break's own level is now
        # the stop and the far side of the box is where price is heading back to
        long_setup, short_setup = short_setup, long_setup
        lvl_up, lvl_dn = lvl_dn, lvl_up

    if bias == "pdc":
        long_setup &= r_close > pdc if mode != "open_atr" else day_open_px > pdc
        short_setup &= r_close < pdc if mode != "open_atr" else day_open_px < pdc
    elif bias == "first_candle":
        fc_open = f["open"].where(idx_in_day == 0).groupby(day).transform("max")
        fc_close = f["close"].where(idx_in_day == 0).groupby(day).transform("max")
        long_setup &= fc_close > fc_open
        short_setup &= fc_close < fc_open
    elif bias != "none":
        raise ValueError(f"orb bias {bias}")

    # an optional second indicator standing behind the break (filt_* keys, see
    # indicator_lab._filter_mask): momentum has to agree before the break is taken
    ok_l, ok_s = _filter_mask(f, scfg)
    long_setup = long_setup & ok_l.fillna(False)
    short_setup = short_setup & ok_s.fillna(False)

    sig = pd.DataFrame(index=f.index)
    pb = float(scfg.get("entry_pullback_frac", 0.0))
    if pb > 0 and mode == "range":
        # rest the limit a Fibonacci fraction of the box BACK inside it, rather
        # than at the edge: a deeper retracement entry
        w = rh - rl
        sig["entry_level_up"], sig["entry_level_down"] = rh - pb * w, rl + pb * w
    elif scfg.get("confirm_break", False):
        # the YouTube-comment rule: don't take the break itself. Mark the high
        # (low) of the bar that broke out and only enter if a later bar trades
        # through it, i.e. the move follows through. Use with entry_fill: stop.
        sig["entry_level_up"], sig["entry_level_down"] = f["high"], f["low"]
    else:
        sig["entry_level_up"], sig["entry_level_down"] = lvl_up, lvl_dn
    finish_direction(sig, f, 1, long_setup, scfg, acfg, cfg, gates_enabled)
    finish_direction(sig, f, -1, short_setup, scfg, acfg, cfg, gates_enabled)

    if stop_mode == "range_opp":
        ls, ss = rl, rh
    elif stop_mode == "range_mid":
        ls = ss = (rh + rl) / 2
    elif stop_mode == "pdc":
        ls = ss = pdc
    elif stop_mode == "atr":
        ls, ss = lvl_up - k_stop * atr, lvl_dn + k_stop * atr
    elif stop_mode == "range_frac":               # a fraction of the box in from the broken edge:
        w = rh - rl                               # 0 = at the edge, 0.5 = midpoint, 1 = far side
        fr = float(scfg.get("stop_range_frac", 1.0))
        ls, ss = rh - fr * w, rl + fr * w
    elif stop_mode == "excursion":                # beyond the day's extreme so far (past the fake)
        ls = f["low"].groupby(day).cummin()
        ss = f["high"].groupby(day).cummax()
    else:
        raise ValueError(f"orb stop {stop_mode}")
    if mode == "open_atr" and stop_mode in ("range_opp", "range_mid"):
        ls, ss = day_open_px, day_open_px            # the open itself is the natural stop for a Williams break
    fh = scfg.get("flat_hour_utc")
    xl, xs = _exit_mask(f, scfg)                     # optional momentum-flip exit
    if fh is not None:                               # flatten before the 22:00 rollover
        late = pd.Series(f.index.hour >= int(fh), index=f.index)
        xl, xs = xl | late, xs | late
    if xl.any() or xs.any():
        sig["long_exit"], sig["short_exit"] = xl, xs
    sig["stop_mode"] = "atr"                         # static, intrabar
    sig["long_stop"], sig["short_stop"] = ls, ss
    far = 99.0 * atr
    sig["long_target"], sig["short_target"] = px + far, px - far     # engine target_r overrides when set
    sig["long_runner"], sig["short_runner"] = sig["long_target"], sig["short_target"]
    sig["long_trigger"], sig["short_trigger"] = ls, ss
    return sig
