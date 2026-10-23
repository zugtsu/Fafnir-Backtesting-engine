"""
Monte Carlo trade reshuffle.

The order trades arrive in is one draw from many plausible orders; the
drawdown you observed is one sample. Reshuffling the trade P&L sequence
(with replacement, so streak lengths vary) and replaying it on the small
account gives the distribution of drawdowns and end equities the same
trade population could have produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _replay(pnls: np.ndarray, start: float, margin: float) -> tuple[float, float]:
    eq, peak, mdd = start, start, 0.0
    for p in pnls:
        eq += p * min(1.0, max(eq, 0.0) / margin)
        peak = max(peak, eq)
        if peak > 0:
            mdd = max(mdd, (peak - eq) / peak)
    return eq, mdd


def reshuffle(trades: pd.DataFrame, runs: int = 1000, start: float = 200.0, margin: float = 200.0,
              seed: int = 42, ruin_pct: float = 50.0) -> dict:
    if trades is None or trades.empty:
        return {"runs": 0}
    pnl = trades["pnl_usd"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    ends, dds = np.empty(runs), np.empty(runs)
    for k in range(runs):
        sample = rng.choice(pnl, size=len(pnl), replace=True)
        ends[k], dds[k] = _replay(sample, start, margin)
    actual_end, actual_dd = _replay(pnl, start, margin)
    q = lambda a, p: float(np.percentile(a, p))
    return {
        "runs": runs, "trades": int(len(pnl)), "start": start,
        "actual_end": actual_end, "actual_max_dd_pct": actual_dd * 100,
        "end_p5": q(ends, 5), "end_p25": q(ends, 25), "end_p50": q(ends, 50), "end_p75": q(ends, 75), "end_p95": q(ends, 95),
        "dd_p50_pct": q(dds, 50) * 100, "dd_p75_pct": q(dds, 75) * 100, "dd_p95_pct": q(dds, 95) * 100,
        "p_end_below_start_pct": float((ends < start).mean() * 100),
        "p_ruin_pct": float((dds >= ruin_pct / 100).mean() * 100),
        "ruin_definition": f"drawdown >= {ruin_pct:g}% of peak",
    }
