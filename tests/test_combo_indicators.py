"""The indicator lab's `combo` system: up to five systems voting together."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from saty_bt.strategies import STRATEGIES
from saty_bt.strategies.indicator_lab import _hold

from test_strategies_engine import ACFG, CFG, features

LAB = STRATEGIES["indicator_lab"]


@pytest.fixture(scope="module")
def F():
    return features(40)


def _scfg(**kw):
    scfg = copy.deepcopy(CFG["strategies"]["indicator_lab"])
    scfg.update(kw)
    return scfg


def _block(f, start, spans):
    """A custom leg wanting the given side over [start+a, start+b) for each (a, b, side)."""
    s = pd.Series(0.0, index=f.index)
    for a, b, side in spans:
        s.iloc[start + a:start + b] = side
    return {"system": "custom", "custom_fn": lambda d, p, s=s: s}


def _start(f):
    return int(np.argmax(f["levels_valid"].to_numpy())) + 50


def test_hold_turns_events_into_a_held_side():
    idx = pd.RangeIndex(8)
    b = lambda *on: pd.Series([i in on for i in range(8)], index=idx)
    held = _hold(b(1), b(5), b(3), b())
    assert held.tolist() == [0, 1, 1, 0, 0, -1, -1, -1]


def test_single_leg_combo_matches_the_plain_system(F):
    for system in ("ma_cross", "rsi_revert", "donchian"):
        plain = LAB(F, _scfg(system=system), ACFG, CFG)
        for mode in ("all", "trigger"):
            combo = LAB(F, _scfg(system="combo", combo_mode=mode, combo=[{"system": system}]), ACFG, CFG)
            assert combo["long_entry"].sum() > 0, system
            if mode == "trigger":                       # trigger with one leg is the leg itself
                assert combo["long_entry"].equals(plain["long_entry"]), system
                assert combo["short_entry"].equals(plain["short_entry"]), system


def test_all_mode_trades_only_the_overlap(F):
    s = _start(F)
    legs = [_block(F, s, [(0, 30, 1)]), _block(F, s, [(10, 40, 1)])]
    sig = LAB(F, _scfg(system="combo", combo_mode="all", combo=legs), ACFG, CFG)
    assert sig["long_setup"].sum() == 1 and sig["long_setup"].iloc[s + 10]
    assert sig["long_exit"].iloc[s + 30] and not sig["long_exit"].iloc[s + 20]
    assert sig["short_setup"].sum() == 0


def test_at_least_mode_counts_votes(F):
    s = _start(F)
    legs = [_block(F, s, [(0, 30, 1)]), _block(F, s, [(10, 40, 1)]), _block(F, s, [(20, 50, -1)])]
    two = LAB(F, _scfg(system="combo", combo_mode="at_least", combo_need=2, combo=legs), ACFG, CFG)
    assert two["long_setup"].iloc[s + 10]                 # legs 1 and 2 long
    assert not two["long_exit"].iloc[s + 20]              # 2 long vs 1 short: still 2 votes
    assert two["long_exit"].iloc[s + 30]                  # leg 1 stops: only 1 long vote left
    three = LAB(F, _scfg(system="combo", combo_mode="at_least", combo_need=3, combo=legs), ACFG, CFG)
    assert three["long_setup"].sum() == 0 and three["short_setup"].sum() == 0


def test_trigger_mode_needs_the_others_to_agree(F):
    s = _start(F)
    trigger = _block(F, s, [(0, 5, 1), (40, 45, 1)])     # two long signals
    confirm = _block(F, s, [(30, 60, 1)])               # only agrees with the second
    sig = LAB(F, _scfg(system="combo", combo_mode="trigger", combo=[trigger, confirm]), ACFG, CFG)
    assert not sig["long_setup"].iloc[s] and sig["long_setup"].iloc[s + 40]
    assert sig["long_setup"].sum() == 1


def test_combo_rejects_more_than_five_legs(F):
    with pytest.raises(ValueError, match="1 to 5"):
        LAB(F, _scfg(system="combo", combo=[{"system": "roc"}] * 6), ACFG, CFG)
