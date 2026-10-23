"""The indicator lab's `custom` system: a plug-in callable drives entries and exits."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from saty_bt.strategies import STRATEGIES

from test_strategies_engine import ACFG, CFG, features


def _scfg(fn, **params):
    scfg = copy.deepcopy(CFG["strategies"]["indicator_lab"])
    scfg.update(system="custom", custom_fn=fn, custom_params=params)
    return scfg


def test_custom_state_flips_become_entries_and_zero_exits():
    f = features(40)
    # Entries only count once the daily levels have warmed up, so start well after that.
    start = int(np.argmax(f["levels_valid"].to_numpy())) + 50
    want = pd.Series(0.0, index=f.index)
    want.iloc[start:start + 20] = 1.0             # long block
    want.iloc[start + 100:start + 120] = -1.0     # short block
    sig = STRATEGIES["indicator_lab"](f, _scfg(lambda d, p: want), ACFG, CFG)
    assert sig["long_setup"].sum() == 1 and sig["long_setup"].iloc[start]
    assert sig["short_setup"].sum() == 1 and sig["short_setup"].iloc[start + 100]
    assert sig["long_exit"].iloc[start + 20]      # back to 0 = flat, so the long exits
    assert not sig["long_exit"].iloc[start + 10]


def test_custom_params_reach_the_callable_and_arrays_are_accepted():
    f = features(40)
    seen = {}

    def fn(d, p):
        seen.update(p)
        return np.sign(d["close"].diff(p["n"]).to_numpy())

    sig = STRATEGIES["indicator_lab"](f, _scfg(fn, n=5), ACFG, CFG)
    assert seen == {"n": 5}
    assert sig["long_setup"].sum() > 0 and sig["short_setup"].sum() > 0
