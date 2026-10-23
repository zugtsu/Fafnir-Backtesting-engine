"""Performance metrics from a trade log and a per-bar equity curve."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bars_per_year(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 252.0
    span_days = (index[-1] - index[0]).total_seconds() / 86400.0
    return len(index) / span_days * 365.25 if span_days > 0 else 252.0


def summarize(trades: pd.DataFrame, equity: pd.Series) -> dict:
    start = float(equity.attrs.get("starting_equity", equity.iloc[0] if len(equity) else 0))
    out: dict = {"trades": int(len(trades)), "starting_equity": start}
    if len(equity):
        end = float(equity.iloc[-1])
        out["ending_equity"] = end
        out["net_pnl_usd"] = end - start
        out["return_pct"] = (end / start - 1) * 100 if start else np.nan
        peak = equity.cummax()
        dd = (equity - peak) / peak.where(peak > 0)
        out["max_drawdown_pct"] = float(-dd.min() * 100) if dd.notna().any() else np.nan
        rets = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        bpy = _bars_per_year(equity.index)
        if len(rets) > 2 and rets.std() > 0:
            out["sharpe"] = float(rets.mean() / rets.std() * np.sqrt(bpy))
            downside = rets[rets < 0].std()
            out["sortino"] = float(rets.mean() / downside * np.sqrt(bpy)) if downside and downside > 0 else np.nan
        else:
            out["sharpe"], out["sortino"] = np.nan, np.nan
        out["halted"] = bool(equity.attrs.get("halted", False))
        out["span_days"] = float((equity.index[-1] - equity.index[0]).total_seconds() / 86400.0)
    if len(trades) == 0:
        out.update({"win_rate": np.nan, "profit_factor": np.nan, "expectancy_r": np.nan,
                    "avg_win_usd": np.nan, "avg_loss_usd": np.nan, "avg_bars_held": np.nan,
                    "wins": 0, "losses": 0, "longs": 0, "shorts": 0, "gross_profit": 0.0, "gross_loss": 0.0})
        return out
    pnl = trades["pnl_usd"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    out["wins"], out["losses"] = int(len(wins)), int(len(losses))
    out["win_rate"] = float(len(wins) / len(pnl) * 100)
    out["gross_profit"], out["gross_loss"] = float(wins.sum()), float(losses.sum())
    out["profit_factor"] = float(wins.sum() / -losses.sum()) if losses.sum() < 0 else np.inf
    out["expectancy_r"] = float(trades["r_multiple"].mean())
    out["expectancy_usd"] = float(pnl.mean())
    out["avg_win_usd"] = float(wins.mean()) if len(wins) else 0.0
    out["avg_loss_usd"] = float(losses.mean()) if len(losses) else 0.0
    out["largest_win"], out["largest_loss"] = float(pnl.max()), float(pnl.min())
    out["avg_bars_held"] = float(trades["bars_held"].mean())
    out["longs"] = int((trades["direction"] == "long").sum())
    out["shorts"] = int((trades["direction"] == "short").sum())
    out["commissions_usd"] = float(trades["commission_usd"].sum())
    out["exit_reasons"] = trades["exit_reason"].value_counts().to_dict()
    out["partial_rate"] = float(trades["partial_taken"].mean() * 100)
    out["reach_618_in_trade_pct"] = float(trades["reached_618"].mean() * 100)
    out["reach_100_in_trade_pct"] = float(trades["reached_100"].mean() * 100)
    # streaks
    s = (pnl > 0).astype(int).to_numpy()
    best = worst = cur_w = cur_l = 0
    for v in s:
        cur_w, cur_l = (cur_w + 1, 0) if v else (0, cur_l + 1)
        best, worst = max(best, cur_w), max(worst, cur_l)
    out["max_win_streak"], out["max_loss_streak"] = best, worst
    return out


def monthly_pnl(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    t = trades.set_index(pd.DatetimeIndex(trades["exit_time"]))
    return t["pnl_usd"].resample("MS").sum()
