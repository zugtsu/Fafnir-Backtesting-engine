"""
Parameter and structure sweeps for a strategy, plus walk-forward validation.

Every configuration is run once over the full history with a sizing that
never binds (fixed $1,000 notional, large equity), so each trade's
R-multiple and dollar P&L are path-independent and the trade list can be
sliced by time. Selection and scoring use R-multiples, which do not
depend on account size.

score(trades) = mean(R) / std(R) * sqrt(n)   - a per-trade t-statistic,
so a config with a good mean on few trades does not beat a config with a
slightly lower mean on many.

Overfit flag: on a one-dimensional slice of the grid (one parameter
varying, others fixed) a value whose score beats BOTH neighbours by more
than `spike_margin` is marked as a spike - an island, not a plateau.
"""

from __future__ import annotations

import copy
import itertools
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import asset_config
from .engine import run_backtest
from .indicators import build_features
from .strategies import STRATEGIES

NEUTRAL_RISK = {"sizing": "fixed_margin", "margin_usd": 200, "leverage": 5}
NEUTRAL_EQUITY = 1_000_000.0          # caps never bind -> $1,000 notional every trade
NEUTRAL_RISK_OVER = {"halt_drawdown_pct": 100, "daily_loss_limit_pct": 100, "consecutive_loss_limit": 10**9}


def score(r: pd.Series, min_n: int = 20) -> float:
    r = r.dropna()
    if len(r) < min_n or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(len(r)))


def grid(space: dict) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(space[k] for k in keys))]


def apply_overrides(scfg: dict, over: dict) -> dict:
    s = copy.deepcopy(scfg)
    for k, v in over.items():
        if k.startswith("gate:"):
            s["gates"][k.split(":", 1)[1]] = bool(v)
        elif k == "target_runner":          # coupled pair (target_frac, runner_frac)
            s["target_frac"], s["runner_frac"] = float(v[0]), float(v[1])
        elif k == "tod_hours":              # RTH first-N-hours window
            s.setdefault("time_of_day", {})["rth_first_hours"] = float(v)
        elif k == "fisher_cap":             # Fisher not-extended cap
            s.setdefault("fisher", {})["max_abs"] = float(v)
            s.setdefault("gates", {})["fisher"] = v is not None and float(v) < 99
        elif isinstance(v, dict):           # a named bundle of settings (e.g. an exit rule)
            for kk, vv in v.items():
                if kk == "name":
                    continue
                s[kk] = vv
        else:
            s[k] = v
    return s


@dataclass
class FeatureCache:
    cfg: dict
    tf: str
    _store: dict | None = None

    def get(self, asset: str, atr_length: int | None = None) -> pd.DataFrame:
        self._store = self._store or {}
        key = (asset, atr_length)
        if key not in self._store:
            c = self.cfg
            if atr_length is not None:
                c = copy.deepcopy(self.cfg)
                c["indicators"]["atr_levels"]["atr_length"] = int(atr_length)
            self._store[key] = build_features(c, asset, self.tf)
        return self._store[key]


def run_config(cfg: dict, cache: FeatureCache, strategy: str, asset: str, over: dict) -> pd.DataFrame:
    acfg = asset_config(cfg, asset)
    scfg = apply_overrides(cfg["strategies"][strategy], {k: v for k, v in over.items() if k != "atr_length"})
    f = cache.get(asset, over.get("atr_length"))
    sig = STRATEGIES[strategy](f, scfg, acfg, cfg)
    trades, _ = run_backtest(f, sig, acfg, cfg, scfg, NEUTRAL_RISK, NEUTRAL_RISK_OVER, NEUTRAL_EQUITY)
    return trades


def run_grid(cfg: dict, strategy: str, assets: list[str], tf: str, space: dict,
             verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """-> (all trades with cfg_id/asset, per (asset, cfg_id) metrics)"""
    cache = FeatureCache(cfg, tf)
    configs = grid(space)
    rows, all_tr = [], []
    t0 = time.time()
    for asset in assets:
        for i, over in enumerate(configs):
            tr = run_config(cfg, cache, strategy, asset, over)
            if len(tr):
                tr = tr.assign(cfg_id=i, asset=asset)
                all_tr.append(tr)
            r = tr["r_multiple"] if len(tr) else pd.Series(dtype=float)
            pnl = tr["pnl_usd"] if len(tr) else pd.Series(dtype=float)
            wins = (pnl > 0).sum() if len(tr) else 0
            gl = pnl[pnl <= 0].sum() if len(tr) else 0.0
            rows.append({"asset": asset, "cfg_id": i, **over, "trades": len(tr),
                         "win_rate": wins / len(tr) * 100 if len(tr) else np.nan,
                         "exp_r": r.mean() if len(tr) else np.nan,
                         "pf": pnl[pnl > 0].sum() / -gl if gl < 0 else np.nan,
                         "pnl_1k": pnl.sum() if len(tr) else 0.0, "score": score(r)})
        if verbose:
            print(f"  {asset}: {len(configs)} configs  ({time.time() - t0:.0f}s)")
    trades = pd.concat(all_tr, ignore_index=True) if all_tr else pd.DataFrame()
    return trades, pd.DataFrame(rows)


def flag_spikes(metrics: pd.DataFrame, params: list[str], value_col: str = "score",
                spike_margin: float = 1.0) -> pd.DataFrame:
    """For each param, look along its 1-D slices; mark values that beat both neighbours by > margin."""
    m = metrics.copy()
    m["spike"] = False
    for p in params:
        others = [q for q in params if q != p]
        for _, grp in m.groupby(["asset"] + others, dropna=False):
            grp = grp.sort_values(p)
            v = grp[value_col].to_numpy(dtype=float)
            for k in range(1, len(v) - 1):
                if np.isfinite(v[k]) and v[k] - max(v[k - 1], v[k + 1]) > spike_margin and v[k] > 0 \
                        and (not np.isfinite(v[k - 1]) or v[k - 1] <= 0 or not np.isfinite(v[k + 1]) or v[k + 1] <= 0):
                    m.loc[grp.index[k], "spike"] = True
    return m


# ---------------------------------------------------------------------------
# walk-forward
# ---------------------------------------------------------------------------

def windows(start: pd.Timestamp, end: pd.Timestamp, is_months: int, oos_months: int) -> list[tuple]:
    out = []
    is_start = start
    while True:
        is_end = is_start + pd.DateOffset(months=is_months)
        oos_end = is_end + pd.DateOffset(months=oos_months)
        if is_end >= end:
            break
        out.append((is_start, is_end, min(oos_end, end)))
        is_start = is_start + pd.DateOffset(months=oos_months)
    return out


def walk_forward(trades: pd.DataFrame, asset: str, is_months: int = 12, oos_months: int = 3,
                 min_is_trades: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    trades: all configs' trades for one asset (from run_grid).
    Per window: pick the cfg with the best IS score, take its OOS trades.
    -> (oos trades with window id + chosen cfg, per-window log)
    """
    t = trades[trades["asset"] == asset].copy()
    if t.empty:
        return pd.DataFrame(), pd.DataFrame()
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    start = t["exit_time"].min().normalize().replace(day=1)
    end = t["exit_time"].max()
    oos_parts, log = [], []
    for w, (a, b, c) in enumerate(windows(start, end, is_months, oos_months)):
        is_t = t[(t["exit_time"] >= a) & (t["exit_time"] < b)]
        sc = is_t.groupby("cfg_id")["r_multiple"].apply(lambda s: score(s, min_is_trades))
        sc = sc.dropna()
        if sc.empty:
            log.append({"window": w, "is_start": a, "is_end": b, "oos_end": c, "cfg_id": None, "is_score": np.nan,
                        "is_trades": 0, "oos_trades": 0, "oos_r": np.nan, "oos_pnl_1k": 0.0})
            continue
        best = int(sc.idxmax())
        oos = t[(t["cfg_id"] == best) & (t["exit_time"] >= b) & (t["exit_time"] < c)].copy()
        oos["window"] = w
        oos_parts.append(oos)
        log.append({"window": w, "is_start": a, "is_end": b, "oos_end": c, "cfg_id": best, "is_score": float(sc.max()),
                    "is_trades": int((is_t["cfg_id"] == best).sum()), "is_exp_r": float(is_t.loc[is_t["cfg_id"] == best, "r_multiple"].mean()),
                    "oos_trades": len(oos), "oos_r": float(oos["r_multiple"].mean()) if len(oos) else np.nan,
                    "oos_pnl_1k": float(oos["pnl_usd"].sum())})
    oos_df = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
    return oos_df, pd.DataFrame(log)


def small_account(trades: pd.DataFrame, start: float = 200.0, margin: float = 200.0,
                  mode: str = "fixed_margin", risk_pct: float = 1.0, leverage: float = 5.0) -> pd.Series:
    """
    Replay a trade sequence on an account starting at `start`.

    fixed_margin     : P&L as recorded (at $margin x leverage notional);
                       once equity drops under the margin the position is
                       scaled to what the account can post.
    fixed_fractional : risk `risk_pct` of current equity to the stop; the
                       trade's R-multiple times that risk is the P&L, capped
                       where the implied notional exceeds leverage x equity.
    -> equity after each trade (index = exit_time)
    """
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades.sort_values("exit_time")
    eq, out = start, []
    if mode == "fixed_fractional":
        stop_frac = ((t["entry_price"] - t["stop_price"]).abs() / t["entry_price"]).to_numpy()
        for r, sf in zip(t["r_multiple"].to_numpy(), stop_frac):
            if eq <= 0 or not np.isfinite(r) or not np.isfinite(sf) or sf <= 0:
                out.append(eq)
                continue
            risk = eq * risk_pct / 100.0
            notional = risk / sf
            cap = eq * leverage
            if notional > cap:
                risk *= cap / notional
            eq += r * risk
            out.append(eq)
    else:
        for pnl in t["pnl_usd"]:
            scale = min(1.0, max(eq, 0.0) / margin)
            eq += pnl * scale
            out.append(eq)
    return pd.Series(out, index=pd.to_datetime(t["exit_time"], utc=True))
