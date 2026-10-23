"""MADH and CMF checked against independent ports and known behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd

from saty_bt.strategies.indicator_lab import _cmf, _madh, _madh_literal
from saty_bt.validate import assert_causal


def _walk(n=600, seed=3):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.5, n))
    h = c + rng.uniform(0.05, 0.6, n)
    l = c - rng.uniform(0.05, 0.6, n)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c,
                         "volume": rng.uniform(100, 1000, n)},
                        index=pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC"))


def test_madh_matches_ehlers_line_for_line():
    df = _walk()
    for short, cyc in ((8, 27), (10, 30), (6, 20)):
        a, b = _madh(df.close, short, cyc), _madh_literal(df.close, short, cyc)
        assert a.notna().sum() == b.notna().sum()
        assert np.allclose(a.dropna(), b.dropna(), atol=1e-10)


def test_madh_reads_trend_and_flat_correctly():
    flat = pd.Series(np.full(200, 50.0))
    assert np.allclose(_madh(flat, 10, 30).dropna(), 0.0)
    up = pd.Series(np.linspace(100, 120, 200))
    assert (_madh(up, 10, 30).dropna() > 0).all()      # short average above long in an uptrend


def test_madh_and_cmf_are_causal():
    df = _walk(800)
    assert_causal(lambda d: d.assign(m=_madh(d.close, 10, 30)), df, ["m"], name="madh")
    assert_causal(lambda d: d.assign(c=_cmf(d, 20)), df, ["c"], name="cmf")


def test_cmf_is_bounded_and_needs_volume():
    df = _walk()
    c = _cmf(df, 20).dropna()
    assert c.between(-1, 1).all()
    assert _cmf(df.assign(volume=0.0), 20).isna().all()   # FX: no volume, no CMF


def test_confluence_readings_are_causal_and_skip_cmf_without_volume():
    from saty_bt.strategies.confluence import agreement, readings
    df = _walk(900)
    for name in ("smi", "ichimoku", "ribbon", "madh", "cmf"):
        assert_causal(lambda d, n=name: d.assign(x=readings(d, {"use": [n]})[n]), df, ["x"],
                      name=f"confluence:{name}")
    _, _, n_vol = agreement(df)
    _, _, n_fx = agreement(df.assign(volume=0.0))
    assert (n_vol, n_fx) == (5, 4)          # CMF only counts where volume exists
