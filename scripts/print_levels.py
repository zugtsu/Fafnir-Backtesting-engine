"""
Print the Saty ATR level grid for an asset on a given date, so it can be
checked against the TradingView plot to the tick.

    python scripts/print_levels.py SPX 2024-11-29
    python scripts/print_levels.py EURUSD 2025-03-12 --tf 4h     # Multiday grid
    python scripts/print_levels.py BTCUSD 2025-01-06 --tf 1d     # Swing grid

The grid shown for date D is the one LIVE during D: previous period close
and previous period ATR(14, Wilder).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from saty_bt.config import asset_config, load_config          # noqa: E402
from saty_bt.data.calendar import period_key                   # noqa: E402
from saty_bt.data.store import load_bars                       # noqa: E402
from saty_bt.indicators import level_period_for                # noqa: E402
from saty_bt.indicators.atr_levels import LEVEL_NAMES, period_grid  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("asset")
    ap.add_argument("date", help="YYYY-MM-DD (trading-day key)")
    ap.add_argument("--tf", default="30m", help="chart timeframe -> picks the Saty mode")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    acfg = asset_config(cfg, args.asset)
    mode, period = level_period_for(cfg, args.tf)
    lsrc = acfg.get("levels_source") or {}
    level_asset = lsrc["asset"] if lsrc.get("enabled") else args.asset

    bars = load_bars(cfg, level_asset, period)
    if "period_key" not in bars:
        bars = bars.copy()
        bars["period_key"] = bars["day_key"].to_numpy()
    grid = period_grid(bars, cfg["indicators"]["atr_levels"])
    key = period_key(pd.DatetimeIndex([pd.Timestamp(args.date)]), period)[0]
    if key not in grid.index:
        print(f"no {period} period starting {key.date()} for {level_asset}; "
              f"range {grid.index[0].date()} .. {grid.index[-1].date()}")
        return 1
    row = grid.loc[key]
    prev = bars[bars["period_key"] == key]
    dec = 5 if acfg.get("class") == "fx" else 2

    print(f"{args.asset}  mode={mode}  level period={period}  levels from={level_asset}")
    print(f"grid live for period starting {key.date()}  (from previous {period} close + ATR)")
    print(f"  PDC  = {row['pdc']:.{dec}f}")
    print(f"  ATR  = {row['atr']:.{dec}f}   (Wilder {cfg['indicators']['atr_levels']['atr_length']})")
    print(f"  {'level':>8}  {'UP':>12}  {'DOWN':>12}")
    for frac, name in LEVEL_NAMES.items():
        if f"{name}_up" in row:
            tag = {"trigger": "trigger", "gg": "38.2 gate", "mid": "50 mid", "target": "61.8 tgt",
                   "full": "100 full"}.get(name, f"{frac:.3f}")
            print(f"  {tag:>8}  {row[f'{name}_up']:>12.{dec}f}  {row[f'{name}_down']:>12.{dec}f}")
    if len(prev):
        p = prev.iloc[0]
        print(f"\nthat period's own bar: O {p['open']:.{dec}f} H {p['high']:.{dec}f} "
              f"L {p['low']:.{dec}f} C {p['close']:.{dec}f}  "
              f"range/ATR = {(p['high'] - p['low']) / row['atr']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
