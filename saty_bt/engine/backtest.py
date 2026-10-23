"""
Bar-by-bar event loop. One asset, one timeframe, one strategy per run.

Order of operations on every bar i (a completed bar):
  1. A pending entry from bar i-1's signal fills at open[i] +/- slippage
     and half the spread, unless bar i opened a new period and the
     strategy is flat-at-period-end (no carrying a Day-mode signal across
     the close), or a risk rule blocks it.
  2. Exits on the open position, pessimistic ordering:
        gap open through an intrabar stop  -> fill at the open
        intrabar stop touched              -> fill at the stop
        target touched                     -> partial at the target,
                                              stop to breakeven, runner on
        runner target touched              -> fill at the runner level
        close-through stop ('trigger' mode)-> exit at next bar's open
        last bar of the period             -> flat at this bar's close
     If stop and target are both inside bar i's range the stop wins.
  3. Entry signal on bar i's close -> pending entry for bar i+1.
  4. Risk checks: trades-per-day, cooldown after a stop-out, daily loss
     limit, consecutive-loss limit, drawdown halt.

Every trade row carries every gate's verdict at entry, MAE/MFE in ATR
units, bars held, the highest level reached, and the exit reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.rates import financing_usd
from ..risk import cost_params, fill_price, notional_usd, size_position, slip, to_usd


@dataclass
class Position:
    direction: int
    entry_i: int
    entry_time: pd.Timestamp
    entry_px: float
    units: float
    units_open: float
    notional: float
    risk_usd: float
    stop: float
    stop_mode: str
    trigger: float
    target: float
    runner: float
    pdc: float
    atr: float
    partial_done: bool = False
    realized: float = 0.0
    commission: float = 0.0
    financing: float = 0.0              # signed rollover credit(+)/debit(-), already inside commission
    best_px: float = 0.0
    stop0: float = 0.0
    max_fav: float = 0.0
    max_adv: float = 0.0
    highest_reach: float = 0.0
    pending_exit: str | None = None
    entry_row: dict = field(default_factory=dict)


def _reach(pos: Position, hi: float, lo: float) -> None:
    fav = (hi - pos.entry_px) if pos.direction > 0 else (pos.entry_px - lo)
    adv = (pos.entry_px - lo) if pos.direction > 0 else (hi - pos.entry_px)
    pos.max_fav = max(pos.max_fav, fav)
    pos.max_adv = max(pos.max_adv, adv)
    ext = (hi - pos.pdc) if pos.direction > 0 else (pos.pdc - lo)
    if pos.atr and pos.atr > 0:
        pos.highest_reach = max(pos.highest_reach, ext / pos.atr)


def run_backtest(f: pd.DataFrame, sig: pd.DataFrame, acfg: dict, cfg: dict,
                 scfg: dict, rcfg: dict, risk_over: dict | None = None,
                 starting_equity: float | None = None) -> tuple[pd.DataFrame, pd.Series]:
    risk = {**cfg.get("risk", {}), **(risk_over or {})}
    equity = float(starting_equity if starting_equity is not None else risk.get("starting_equity", 100000))
    start_equity = equity
    cp = cost_params(acfg)
    quote = acfg.get("quote_ccy", "USD")
    # currencies whose rates set the rollover: explicit, else from the pair name
    fin_ccy = (acfg.get("base_ccy") or str(acfg.get("name", ""))[:3],
               acfg.get("quote_ccy") or str(acfg.get("name", ""))[3:6])
    if cp.get("financing") == "rate_diff" and acfg.get("class") != "fx":
        raise ValueError(f"financing: rate_diff is for FX pairs, not {acfg.get('name')}")
    flat_eop = bool(scfg.get("flat_at_period_end", True))
    scale_pct = float(scfg.get("scale_out_pct", 50)) / 100.0
    be_after = bool(risk.get("breakeven_after_partial", True))
    max_per_day = int(risk.get("max_trades_per_asset_per_day", 2))
    cooldown = int(risk.get("cooldown_bars_after_stop", 0))
    daily_loss_pct = float(risk.get("daily_loss_limit_pct", 100))
    consec_limit = int(risk.get("consecutive_loss_limit", 10**9))
    dd_halt = float(risk.get("halt_drawdown_pct", 100))
    # a structural stop sitting a hair from the entry makes R meaningless and
    # the position absurdly large under fixed-fractional sizing: skip those
    min_stop_atr = float(risk.get("min_stop_atr", 0.05))
    min_target_atr = float(risk.get("min_target_atr", 0.02))

    o, h, l, c = (f[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    idx = f.index
    day = f["day_key"].to_numpy()
    n = len(f)
    new_day = np.r_[True, day[1:] != day[:-1]]
    last_of_day = np.r_[day[:-1] != day[1:], True]
    atr_arr = f["atr"].to_numpy(dtype=float)
    pdc_arr = f["pdc"].to_numpy(dtype=float)
    chart_atr = f["chart_atr"].to_numpy(dtype=float)

    long_entry = sig["long_entry"].to_numpy(dtype=bool)
    short_entry = sig["short_entry"].to_numpy(dtype=bool)
    stops = {1: sig["long_stop"].to_numpy(dtype=float), -1: sig["short_stop"].to_numpy(dtype=float)}
    targets = {1: sig["long_target"].to_numpy(dtype=float), -1: sig["short_target"].to_numpy(dtype=float)}
    runners = {1: sig["long_runner"].to_numpy(dtype=float), -1: sig["short_runner"].to_numpy(dtype=float)}
    trig = {1: sig["long_trigger"].to_numpy(dtype=float), -1: sig["short_trigger"].to_numpy(dtype=float)}
    stop_mode = str(sig["stop_mode"].iloc[0])
    # entry fill: market at next open (default) or a limit resting at the
    # signal's entry level for `limit_valid_bars` bars, filled at the level
    # (or the open, if the bar opens beyond it) plus half the spread, no slippage
    entry_fill = str(scfg.get("entry_fill", "market"))
    limit_valid = int(scfg.get("limit_valid_bars", 3))
    # targets defined from the fill rather than the level grid: a multiple of
    # the initial stop distance, or a fraction of the level-ATR from entry
    target_r = float(scfg.get("target_r", 0) or 0)
    target_atr_from_entry = float(scfg.get("target_atr_from_entry", 0) or 0)
    target_abs = float(scfg.get("target_abs", 0) or 0)          # fixed price distance from the fill (e.g. 20 pips)
    # trailing stop: ratchet the stop to the best price reached, less a distance
    # given either as a multiple of the INITIAL risk (trail_r) or of the level
    # ATR (trail_atr). It never loosens. trail_start_r delays it until the trade
    # is that many R in front.
    trail_r = float(scfg.get("trail_r", 0) or 0)
    trail_atr = float(scfg.get("trail_atr", 0) or 0)
    trail_start_r = float(scfg.get("trail_start_r", 0) or 0)
    levels = {d: sig[k].to_numpy(dtype=float) if k in sig else None
              for d, k in ((1, "entry_level_up"), (-1, "entry_level_down"))}
    if entry_fill in ("limit", "stop") and (levels[1] is None or levels[-1] is None):
        raise ValueError(f"entry_fill={entry_fill} needs entry_level_up/down on the signal frame")
    # optional per-bar exit rules (Strategies B/C): a trailing level that
    # exits on a confirmed close through it, and a boolean exit signal
    trails = {d: sig[k].to_numpy(dtype=float) if k in sig else None
              for d, k in ((1, "long_trail"), (-1, "short_trail"))}
    exit_sig = {d: sig[k].to_numpy(dtype=bool) if k in sig else None
                for d, k in ((1, "long_exit"), (-1, "short_exit"))}
    exit_sig_at_close = str(scfg.get("exit_signal_fill", "next_open")) == "close"
    gate_cols = [col for col in sig.columns if "_gate_" in col or "_why_" in col]
    gate_arr = {col: sig[col].to_numpy() for col in gate_cols}
    gap_arr = {1: sig["long_gap"].to_numpy(dtype=bool), -1: sig["short_gap"].to_numpy(dtype=bool)}
    day_open_arr = pd.DatetimeIndex(f["day_open"])
    ctx = {k: f[k].to_numpy() for k in ("ribbon_state", "osc", "osc_zone", "stack_winner", "rel_vol",
                                        "range_util", "compression")}
    ctx["htf_ribbon_state"] = f["htf_ribbon_state"].to_numpy() if "htf_ribbon_state" in f else np.full(n, "na", dtype=object)
    sig_any = long_entry | short_entry

    trades: list[dict] = []
    eq_curve = np.empty(n)
    pos: Position | None = None
    pending: tuple[int, int] | None = None       # (direction, signal_i)
    trades_today = 0
    day_start_equity = equity
    cooldown_until = -1
    consec_losses = 0
    peak = equity
    halted = False
    halted_today = False

    def close_out(i: int, px_ref: float, units: float, reason: str, at_open: bool = False) -> float:
        nonlocal equity
        px = fill_price(px_ref, pos.direction, cp, is_entry=False)
        pnl_q = units * (px - pos.entry_px) * pos.direction
        pnl = to_usd(pnl_q, px, quote)
        comm = notional_usd(units, px, quote) * cp["commission_bps"] / 1e4 + float(cp.get("fee_fixed_usd", 0.0))
        if cp.get("financing") == "rate_diff":                # real rollover: rate differential less markup
            fin = financing_usd(notional_usd(units, pos.entry_px, quote), fin_ccy[0], fin_ccy[1],
                                pos.direction, pos.entry_time, idx[i], cp["financing_markup_pct"],
                                asset=acfg["name"], weekend=cp["financing_weekend"],
                                day_count=cp["financing_day_count"])
            comm -= fin
            pos.financing += fin
        elif cp.get("holding_bps_per_hour", 0.0) > 0:          # funding/borrowing for the hours this slice was open
            hours = (idx[i] - pos.entry_time).total_seconds() / 3600.0
            comm += notional_usd(units, pos.entry_px, quote) * cp["holding_bps_per_hour"] / 1e4 * hours
        pos.realized += pnl - comm
        pos.commission += comm
        equity += pnl - comm
        pos.units_open -= units
        return px

    def record(i: int, exit_px: float, reason: str) -> None:
        nonlocal consec_losses, cooldown_until, trades_today
        p = pos
        r = p.realized / p.risk_usd if p.risk_usd > 0 else np.nan
        row = {
            "asset": acfg["name"], "direction": "long" if p.direction > 0 else "short",
            "entry_time": p.entry_time, "entry_price": p.entry_px,
            "exit_time": idx[i], "exit_price": exit_px, "exit_reason": reason,
            "bars_held": i - p.entry_i + 1, "units": p.units, "notional_usd": p.notional,
            "risk_usd": p.risk_usd, "stop_init": p.stop_mode, "stop_price": p.entry_row.get("stop_price"),
            "target_price": p.target, "runner_price": p.runner,
            "partial_taken": p.partial_done, "pnl_usd": p.realized, "commission_usd": p.commission,
            "financing_usd": p.financing,
            "r_multiple": r, "equity_after": equity,
            "mae_atr": p.max_adv / p.atr if p.atr else np.nan,
            "mfe_atr": p.max_fav / p.atr if p.atr else np.nan,
            "highest_level_reached": p.highest_reach,
            "reached_382": p.highest_reach >= 0.382, "reached_618": p.highest_reach >= 0.618,
            "reached_100": p.highest_reach >= 1.0,
        }
        row.update(p.entry_row)
        trades.append(row)
        if p.realized < 0:
            consec_losses += 1
            if reason in ("stop", "stop_gap", "stop_close_through", "stop_be"):
                cooldown_until = i + cooldown
        else:
            consec_losses = 0

    eq_curve[:] = np.nan
    cur_day = None
    for i in range(n):
        # Fast path: flat, nothing pending, no signal on this bar. Equity is
        # unchanged and no rule can fire, so the bar is skipped; the day
        # bookkeeping below is applied lazily on the next bar that matters.
        if pos is None and pending is None and not sig_any[i]:
            continue
        if day[i] != cur_day:
            cur_day = day[i]
            trades_today = 0
            day_start_equity = equity
            halted_today = False
            consec_losses = 0                       # the breaker is a same-day rule
            if pending is not None and flat_eop and new_day[i]:
                pending = None                      # never carry a Day-mode signal over the close

        # 1. pending entry: market fill at this bar's open, or a resting limit
        if pending is not None and (halted or halted_today):
            pending = None
        if pending is not None and pos is None:
            d, si = pending
            px = None
            if entry_fill == "limit":
                lvl = levels[d][si]
                touched = (l[i] <= lvl) if d > 0 else (h[i] >= lvl)
                if not np.isfinite(lvl) or i > si + limit_valid:
                    pending = None                          # expired unfilled
                elif touched:
                    ref = min(o[i], lvl) if d > 0 else max(o[i], lvl)
                    px = ref + d * cp["spread"] / 2.0
                    pending = None
                # else: keep resting
            elif entry_fill == "stop":
                # a stop order ABOVE the market for a long: only fills if price
                # trades through the level, i.e. the move follows through. Pays
                # slippage as well as the spread, and a gap fills at the open.
                lvl = levels[d][si]
                touched = (h[i] >= lvl) if d > 0 else (l[i] <= lvl)
                if not np.isfinite(lvl) or i > si + limit_valid:
                    pending = None                          # never confirmed
                elif touched:
                    ref = max(o[i], lvl) if d > 0 else min(o[i], lvl)
                    px = fill_price(ref, d, cp, is_entry=True)
                    pending = None
                # else: keep waiting for confirmation
            else:
                px = fill_price(o[i], d, cp, is_entry=True)
                pending = None
            if px is not None and trades_today < max_per_day and i > cooldown_until and equity > 0:
                stop_lvl = stops[d][si]
                stop_dist = (px - stop_lvl) * d
                min_dist = min_stop_atr * atr_arr[si] if np.isfinite(atr_arr[si]) else 0.0
                if np.isfinite(stop_lvl) and stop_dist > max(min_dist, 0.0):
                    units, notional, risk_usd = size_position(rcfg, equity, px, stop_dist, quote)
                    if units > 0:
                        comm = notional * cp["commission_bps"] / 1e4 + float(cp.get("fee_fixed_usd", 0.0))   # fee_fixed_usd: flat per transaction (gas)
                        equity -= comm
                        entry_row = {"signal_time": idx[si], "hour_utc": idx[si].hour,
                                     "session_offset_h": (idx[si] - day_open_arr[si]).total_seconds() / 3600.0,
                                     "ribbon_state": ctx["ribbon_state"][si],
                                     "htf_ribbon_state": ctx["htf_ribbon_state"][si],
                                     "osc": ctx["osc"][si], "osc_zone": ctx["osc_zone"][si],
                                     "stack_winner": ctx["stack_winner"][si], "rel_vol": ctx["rel_vol"][si],
                                     "range_util": ctx["range_util"][si], "compression": bool(ctx["compression"][si]),
                                     "level_atr": atr_arr[si], "pdc": pdc_arr[si], "stop_price": stop_lvl,
                                     "gap_entry": bool(gap_arr[d][si]),
                                     "entry_fill": entry_fill, "bars_to_fill": i - si,
                                     "trades_today_before": trades_today}
                        pref = "long_" if d > 0 else "short_"
                        for col in gate_cols:
                            if col.startswith(pref):
                                entry_row[col.split("_", 1)[1]] = gate_arr[col][si]
                        tgt_px, run_px = targets[d][si], runners[d][si]
                        if target_r > 0:
                            tgt_px = px + d * target_r * stop_dist
                            run_px = tgt_px
                        elif target_abs > 0:
                            tgt_px = px + d * target_abs
                            run_px = tgt_px
                        elif target_atr_from_entry > 0 and np.isfinite(atr_arr[si]):
                            tgt_px = px + d * target_atr_from_entry * atr_arr[si]
                            run_px = tgt_px
                        # a level target the fill has already passed is not a trade
                        min_tgt = min_target_atr * atr_arr[si] if np.isfinite(atr_arr[si]) else 0.0
                        if np.isfinite(tgt_px) and (tgt_px - px) * d <= min_tgt:
                            equity += comm            # refund: no position was opened
                            pending = None
                            continue
                        pos = Position(direction=d, entry_i=i, entry_time=idx[i], entry_px=px, units=units,
                                       units_open=units, notional=notional, risk_usd=risk_usd,
                                       stop=stop_lvl, stop_mode=stop_mode, trigger=trig[d][si],
                                       best_px=px, stop0=stop_lvl,
                                       target=tgt_px, runner=run_px, pdc=pdc_arr[si],
                                       atr=atr_arr[si], commission=comm, entry_row=entry_row)
                        trades_today += 1

        # 2. exits
        if pos is not None:
            d = pos.direction
            exit_px, reason = None, None
            if pos.pending_exit is not None:
                exit_px = close_out(i, o[i], pos.units_open, pos.pending_exit)
                reason = pos.pending_exit
            else:
                _reach(pos, h[i], l[i])
                if trail_r > 0 or trail_atr > 0:
                    pos.best_px = max(pos.best_px, h[i]) if d > 0 else min(pos.best_px, l[i])
                    r0 = abs(pos.entry_px - pos.stop0)
                    moved = (pos.best_px - pos.entry_px) * d
                    if r0 > 0 and moved >= trail_start_r * r0:
                        gap = trail_r * r0 if trail_r > 0 else trail_atr * pos.atr
                        if np.isfinite(gap) and gap > 0:
                            lvl_tr = pos.best_px - d * gap
                            pos.stop = max(lvl_tr, pos.stop) if d > 0 else min(lvl_tr, pos.stop)
                intrabar_stop = pos.stop_mode not in ("trigger", "close_through") or pos.partial_done
                if pos.stop_mode == "dynamic_touch":
                    # trailing level (a ribbon EMA, shifted one bar by the strategy);
                    # after a partial the breakeven stop still applies, whichever is tighter
                    lvl_dyn = stops[d][i]
                    if np.isfinite(lvl_dyn):
                        pos.stop = max(lvl_dyn, pos.stop) if d > 0 else min(lvl_dyn, pos.stop)
                hit_stop = intrabar_stop and ((l[i] <= pos.stop) if d > 0 else (h[i] >= pos.stop))
                gapped = intrabar_stop and ((o[i] <= pos.stop) if d > 0 else (o[i] >= pos.stop))
                hit_tgt = (h[i] >= pos.target) if d > 0 else (l[i] <= pos.target)
                hit_run = (h[i] >= pos.runner) if d > 0 else (l[i] <= pos.runner)
                if hit_stop:
                    ref = o[i] if gapped else pos.stop
                    exit_px = close_out(i, ref, pos.units_open, "stop")
                    reason = "stop_gap" if gapped else ("stop_be" if pos.partial_done else "stop")
                else:
                    if hit_tgt and not pos.partial_done:
                        part = pos.units_open * scale_pct
                        if part > 0 and scale_pct < 1.0:
                            close_out(i, pos.target, part, "partial")
                            pos.partial_done = True
                            if be_after:
                                pos.stop = pos.entry_px
                        else:
                            exit_px = close_out(i, pos.target, pos.units_open, "target")
                            reason = "target"
                    if reason is None and hit_run and pos.partial_done:
                        exit_px = close_out(i, pos.runner, pos.units_open, "runner")
                        reason = "runner"
                    # confirmed-close exits: close-through stop (static trigger or a
                    # per-bar level such as EMA48), trailing level, exit signal
                    why = None
                    if pos.stop_mode in ("trigger", "close_through") and not pos.partial_done:
                        lvl = pos.trigger if pos.stop_mode == "trigger" else stops[d][i]
                        if np.isfinite(lvl) and ((c[i] < lvl) if d > 0 else (c[i] > lvl)):
                            why = "stop_close_through"
                    if why is None and trails[d] is not None and np.isfinite(trails[d][i]) \
                            and ((c[i] < trails[d][i]) if d > 0 else (c[i] > trails[d][i])):
                        why = "trail"
                    if why is None and exit_sig[d] is not None and exit_sig[d][i]:
                        why = "exit_signal"
                    if reason is None and why == "exit_signal" and exit_sig_at_close:
                        exit_px = close_out(i, c[i], pos.units_open, why)     # market on that bar's close
                        reason = why
                    elif reason is None and why is not None:
                        if last_of_day[i] and flat_eop:
                            exit_px = close_out(i, c[i], pos.units_open, why)
                            reason = why
                        else:
                            pos.pending_exit = why
                    if reason is None and pos.pending_exit is None and last_of_day[i] and flat_eop:
                        exit_px = close_out(i, c[i], pos.units_open, "period_end")
                        reason = "period_end"
            if reason is not None:
                record(i, exit_px, reason)
                pos = None

        # 3. new signal on this bar's close
        if pos is None and pending is None and not halted and not halted_today:
            if not (last_of_day[i] and flat_eop):
                if long_entry[i]:
                    pending = (1, i)
                elif short_entry[i]:
                    pending = (-1, i)

        # 4. risk
        peak = max(peak, equity)
        if peak > 0 and (peak - equity) / peak * 100 >= dd_halt:
            halted = True
        if day_start_equity > 0 and (day_start_equity - equity) / day_start_equity * 100 >= daily_loss_pct:
            halted_today = True
        if consec_losses >= consec_limit:
            halted_today = True
        if equity <= 0:
            halted = True
        # mark-to-market equity
        mtm = equity
        if pos is not None:
            unreal = to_usd(pos.units_open * (c[i] - pos.entry_px) * pos.direction, c[i], quote)
            mtm += unreal
        eq_curve[i] = mtm

    trades_df = pd.DataFrame(trades)
    # skipped bars carry the last known equity; before the first event it is the start
    eq = pd.Series(eq_curve, index=idx, name="equity").ffill().fillna(start_equity)
    eq.attrs["starting_equity"] = start_equity
    eq.attrs["halted"] = halted
    return trades_df, eq
