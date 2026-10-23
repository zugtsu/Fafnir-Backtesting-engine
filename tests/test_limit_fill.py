"""Limit-at-level entry: fills at the level (or a better open), not the next open; expires unfilled."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saty_bt.engine import run_backtest
from saty_bt.strategies import STRATEGIES
from tests.test_strategies_engine import ACFG, CFG, NO_LIMITS, RISK, features


def _one_signal(f, i):
    sig = STRATEGIES["level_retest"](f, CFG["strategies"]["level_retest"], ACFG, CFG)
    sig.loc[:, "long_entry"] = False
    sig.loc[:, "short_entry"] = False
    sig.iloc[i, sig.columns.get_loc("long_entry")] = True
    return sig


def test_limit_fills_at_the_level_not_the_open():
    f = features(n_days=40).copy()
    i = 30 * 48
    sig = _one_signal(f, i)
    lvl = float(sig["entry_level_up"].iloc[i])
    # next bar opens above the level and dips to it: fill must be the level
    f.iloc[i + 1, f.columns.get_loc("open")] = lvl + 0.002
    f.iloc[i + 1, f.columns.get_loc("high")] = lvl + 0.003
    f.iloc[i + 1, f.columns.get_loc("low")] = lvl - 0.0005
    f.iloc[i + 1, f.columns.get_loc("close")] = lvl + 0.001
    scfg = {**CFG["strategies"]["level_retest"], "entry_fill": "limit", "limit_valid_bars": 3}
    tr, _ = run_backtest(f, sig, ACFG, CFG, scfg, RISK, NO_LIMITS, 1e6)
    assert len(tr) == 1
    assert tr["entry_price"].iloc[0] == pytest.approx(lvl)          # zero costs in ACFG
    assert tr["entry_fill"].iloc[0] == "limit" and tr["bars_to_fill"].iloc[0] == 1


def test_limit_expires_when_never_touched():
    f = features(n_days=40).copy()
    i = 30 * 48
    sig = _one_signal(f, i)
    lvl = float(sig["entry_level_up"].iloc[i])
    for k in range(1, 6):                                      # price stays well above the level
        for col in ("open", "high", "low", "close"):
            f.iloc[i + k, f.columns.get_loc(col)] = lvl + 0.01
    scfg = {**CFG["strategies"]["level_retest"], "entry_fill": "limit", "limit_valid_bars": 3}
    tr, _ = run_backtest(f, sig, ACFG, CFG, scfg, RISK, NO_LIMITS, 1e6)
    assert len(tr) == 0


def test_market_fill_is_next_open_by_contrast():
    f = features(n_days=40).copy()
    i = 30 * 48
    sig = _one_signal(f, i)
    sig["stop_mode"] = "atr"
    sig["long_stop"] = f["open"].shift(-1) - 0.01              # a valid stop below whatever the fill is
    scfg = {**CFG["strategies"]["level_retest"], "entry_fill": "market"}
    tr, _ = run_backtest(f, sig, ACFG, CFG, scfg, RISK, NO_LIMITS, 1e6)
    assert len(tr) == 1 and tr["entry_price"].iloc[0] == pytest.approx(f["open"].iloc[i + 1])
    assert tr["entry_fill"].iloc[0] == "market"
