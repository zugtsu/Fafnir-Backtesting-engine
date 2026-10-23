"""The four Saty indicators plus the feature builder that stacks them."""

from __future__ import annotations

import pandas as pd

from ..config import asset_config, timeframe_minutes
from ..data.store import load_bars
from .atr_levels import atr_levels, period_grid
from .core import atr, ema, rma, swing_highs, swing_lows, true_range
from .phase_oscillator import phase_oscillator
from .pivot_ribbon import pivot_ribbon, time_warp
from .volume_stack import volume_stack
from .vwap import vwap
from .heikin_ashi import heikin_ashi
from .voss import voss

__all__ = ["atr_levels", "period_grid", "pivot_ribbon", "time_warp", "phase_oscillator",
           "volume_stack", "vwap", "heikin_ashi", "build_features", "atr", "ema", "rma", "true_range",
           "swing_lows", "swing_highs"]


def level_period_for(cfg: dict, tf: str) -> tuple[str, str]:
    mode = cfg["mode_by_timeframe"][tf]
    return mode, cfg["period_by_mode"][mode]


def build_features(cfg: dict, asset: str, tf: str, chart: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Everything the strategies read, on one chart frame:
    ATR levels (from the level-period bars of the asset's mode), ribbon,
    HTF Time Warp ribbon, phase oscillator, volume stack, swing pivots.
    All columns on row t are computed from bars completing at or before
    row t's close.
    """
    acfg = asset_config(cfg, asset)
    ind = cfg["indicators"]
    chart = load_bars(cfg, asset, tf) if chart is None else chart
    if "day_open" not in chart:                 # daily bars: the bar is the day
        chart = chart.copy()
        chart["day_open"] = chart.index
    mode, period = level_period_for(cfg, tf)

    lsrc = acfg.get("levels_source") or {}
    level_asset = lsrc.get("asset", asset) if lsrc.get("enabled") else asset
    period_bars = load_bars(cfg, level_asset, period)
    if "period_key" not in period_bars:
        period_bars = period_bars.copy()
        period_bars["period_key"] = period_bars["day_key"].to_numpy()

    f = atr_levels(chart, period_bars, period, ind["atr_levels"])
    f["mode"] = mode
    f["level_source"] = level_asset
    f = pivot_ribbon(f, ind["pivot_ribbon"])
    htf_tf = ind["pivot_ribbon"].get("time_warp_by_timeframe", {}).get(tf)
    if htf_tf:
        try:
            htf = load_bars(cfg, asset, htf_tf)
            f = time_warp(f, htf, ind["pivot_ribbon"])
            f["htf_tf"] = htf_tf
        except FileNotFoundError:
            f["htf_tf"] = None
    f = phase_oscillator(f, ind["phase_oscillator"])
    f = volume_stack(f, ind["volume_stack"])
    f = vwap(f, ind.get("vwap", {}))
    f = heikin_ashi(f)
    f = voss(f, ind.get("voss", {}))
    from .trend import adx as _adx, fisher as _fisher, macd as _macd, rsi as _rsi
    tcfg = ind.get("adx", {})
    f = f.join(_adx(f, int(tcfg.get("di_length", 14)), int(tcfg.get("adx_length", 14))))
    f = f.join(_fisher(f, int(ind.get("fisher", {}).get("length", 10))))
    mcfg = ind.get("macd", {})
    f = f.join(_macd(f, int(mcfg.get("fast", 12)), int(mcfg.get("slow", 26)), int(mcfg.get("signal", 9))))
    f = f.join(_rsi(f, int(ind.get("rsi", {}).get("length", 14))))
    n = int(cfg.get("strategies", {}).get("swing_bars", 2))
    f["swing_low"] = swing_lows(f, n)
    f["swing_high"] = swing_highs(f, n)
    f["chart_atr"] = atr(f, int(ind["phase_oscillator"].get("atr_length", 14)))
    f["asset"], f["tf"] = asset, tf
    return f
