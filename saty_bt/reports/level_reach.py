"""
Level-reach statistics - the core research output.

Independent of any trade: for every level period (day for Day mode, etc.)
and direction, find the first confirmed close through the trigger, the
first confirmed close through the 38.2 gate, and then whether price
subsequently TOUCHED 61.8 and 100 inside the same period. Also whether a
trigger close was followed by a 38.2 touch.

    P(reach 61.8 | closed through 38.2)
    P(reach 100  | reached 61.8)
    P(reach 38.2 | closed through trigger)

Each event row records the state at the 38.2 cross (hour of session,
ribbon state, HTF ribbon, oscillator zone, stack winner, rel-vol,
range utilisation) so the rates can be broken out by any of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _events(f: pd.DataFrame, d: int) -> pd.DataFrame:
    close, hi, lo = f["close"].to_numpy(), f["high"].to_numpy(), f["low"].to_numpy()
    trig = (f["call_trigger"] if d > 0 else f["put_trigger"]).to_numpy()
    gg = (f["gg_up"] if d > 0 else f["gg_down"]).to_numpy()
    tgt = (f["target_up"] if d > 0 else f["target_down"]).to_numpy()
    full = (f["full_up"] if d > 0 else f["full_down"]).to_numpy()
    valid = f["levels_valid"].to_numpy()
    pk = f["period_key"].to_numpy()
    n = len(f)
    rows = []
    i = 0
    while i < n:
        j = i
        while j < n and pk[j] == pk[i]:
            j += 1
        if valid[i]:
            beyond = (lambda x, lvl: x > lvl) if d > 0 else (lambda x, lvl: x < lvl)
            ext = hi if d > 0 else lo
            t_i = next((k for k in range(i, j) if beyond(close[k], trig[k])), None)
            g_i = next((k for k in range(i, j) if beyond(close[k], gg[k])), None)
            row = {"period_key": pk[i], "direction": "long" if d > 0 else "short",
                   "trigger_close": t_i is not None, "gate_close": g_i is not None}
            if t_i is not None:
                row["reach_382_after_trigger"] = bool(any(beyond(ext[k], gg[k]) or ext[k] == gg[k]
                                                          for k in range(t_i, j)))
                row["trigger_hour"] = f.index[t_i].hour
            if g_i is not None:
                after = range(g_i, j)
                r618 = bool(any((ext[k] >= tgt[k]) if d > 0 else (ext[k] <= tgt[k]) for k in after))
                r100 = bool(any((ext[k] >= full[k]) if d > 0 else (ext[k] <= full[k]) for k in after))
                row.update({
                    "reach_618": r618, "reach_100": r100,
                    "gate_time": f.index[g_i], "gate_hour_utc": f.index[g_i].hour,
                    "session_offset_h": (f.index[g_i] - f["day_open"].iloc[g_i]).total_seconds() / 3600.0,
                    "ribbon_state": f["ribbon_state"].iloc[g_i],
                    "htf_ribbon_state": f["htf_ribbon_state"].iloc[g_i] if "htf_ribbon_state" in f else "na",
                    "ribbon_agrees": (f["ribbon_bull"] if d > 0 else f["ribbon_bear"]).iloc[g_i],
                    "htf_agrees": (f["htf_ribbon_bull"] if d > 0 else f["htf_ribbon_bear"]).iloc[g_i]
                                  if "htf_ribbon_bull" in f else np.nan,
                    "osc_zone": f["osc_zone"].iloc[g_i], "osc": f["osc"].iloc[g_i],
                    "stack_winner": f["stack_winner"].iloc[g_i], "rel_vol": f["rel_vol"].iloc[g_i],
                    "stack_agrees": f["stack_winner"].iloc[g_i] == ("buy" if d > 0 else "sell"),
                    "range_util_before": f["range_util"].iloc[g_i - 1] if g_i > i else 0.0,
                    "gap_open": (f["open"].iloc[g_i] > gg[g_i]) if d > 0 else (f["open"].iloc[g_i] < gg[g_i]),
                })
            rows.append(row)
        i = j
    return pd.DataFrame(rows)


def level_reach_events(f: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([_events(f, 1), _events(f, -1)], ignore_index=True)


def base_rates(ev: pd.DataFrame) -> dict:
    out = {"periods": int(ev["period_key"].nunique()) if len(ev) else 0}
    g = ev[ev["gate_close"] == True]  # noqa: E712
    out["gate_closes"] = int(len(g))
    out["p_618_given_382"] = float(g["reach_618"].mean() * 100) if len(g) else np.nan
    r6 = g[g["reach_618"] == True]  # noqa: E712
    out["p_100_given_618"] = float(r6["reach_100"].mean() * 100) if len(r6) else np.nan
    t = ev[ev["trigger_close"] == True]  # noqa: E712
    out["trigger_closes"] = int(len(t))
    out["p_382_given_trigger"] = float(t["reach_382_after_trigger"].mean() * 100) if len(t) else np.nan
    return out


def breakdown(ev: pd.DataFrame, by: str, min_n: int = 10) -> pd.DataFrame:
    g = ev[ev["gate_close"] == True].copy()  # noqa: E712
    if by == "session_hour":
        g["session_hour"] = np.floor(g["session_offset_h"]).astype(int)
    elif by == "rel_vol_bucket":
        g["rel_vol_bucket"] = pd.cut(g["rel_vol"], [0, 0.8, 1.2, 1.5, 2.0, np.inf],
                                     labels=["<0.8", "0.8-1.2", "1.2-1.5", "1.5-2", ">2"])
    elif by == "range_util_bucket":
        g["range_util_bucket"] = pd.cut(g["range_util_before"], [-0.01, 0.4, 0.8, 1.0, np.inf],
                                        labels=["<0.4", "0.4-0.8", "0.8-1.0", ">1.0"])
    if by not in g:
        return pd.DataFrame()
    t = g.groupby(by, observed=True).agg(n=("reach_618", "size"), p_618=("reach_618", "mean"),
                                         p_100=("reach_100", "mean"))
    t["p_618"] *= 100
    t["p_100"] *= 100
    return t[t["n"] >= min_n].round(1)
