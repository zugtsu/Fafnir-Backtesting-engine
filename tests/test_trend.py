"""ADX and Fisher Transform against literal bar-by-bar ports of the published formulas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saty_bt.indicators.trend import adx, fisher
from tests.test_indicators import bars


def test_adx_matches_wilder_recurrence():
    df = bars(300, seed=11)
    out = adx(df, 14, 14)
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    n = len(df)
    pdm = np.zeros(n); mdm = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        up, dn = h[i] - h[i - 1], l[i - 1] - l[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        mdm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    tr[0] = h[0] - l[0]

    def wilder(x, L):
        s = np.full(n, np.nan)
        s[L - 1] = x[:L].mean()
        for i in range(L, n):
            s[i] = (s[i - 1] * (L - 1) + x[i]) / L
        return s
    str_, spdm, smdm = wilder(tr, 14), wilder(pdm, 14), wilder(mdm, 14)
    pdi, mdi = 100 * spdm / str_, 100 * smdm / str_
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi)
    # ADX = Wilder smoothing of DX, seeded with the mean of the first 14 valid DX
    valid = np.where(~np.isnan(dx))[0]
    a = np.full(n, np.nan)
    s0 = valid[0] + 13
    a[s0] = dx[valid[0]:s0 + 1].mean()
    for i in range(s0 + 1, n):
        a[i] = (a[i - 1] * 13 + dx[i]) / 14
    np.testing.assert_allclose(out["plus_di"].to_numpy()[40:], pdi[40:], rtol=1e-9)
    np.testing.assert_allclose(out["adx"].to_numpy()[60:], a[60:], rtol=1e-9)
    assert out["adx"].dropna().between(0, 100).all()


def test_fisher_matches_pine_recurrence():
    df = bars(200, seed=5)
    out = fisher(df, 10)
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    n = len(df)
    v1 = 0.0; fish = 0.0
    expect = np.zeros(n)
    for i in range(n):
        if i < 9:
            expect[i] = 0.0
            continue
        w = hl2[i - 9:i + 1]
        x = (hl2[i] - w.min()) / (w.max() - w.min()) - 0.5
        v1 = 0.33 * 2 * x + 0.67 * v1
        v1 = min(max(v1, -0.999), 0.999)
        fish = 0.5 * np.log((1 + v1) / (1 - v1)) + 0.5 * fish
        expect[i] = fish
    np.testing.assert_allclose(out["fisher"].to_numpy()[9:], expect[9:], rtol=1e-9)
    assert (out["fisher_trigger"].to_numpy()[10:] == out["fisher"].to_numpy()[9:-1]).all()


def test_trend_indicators_are_causal():
    df = bars(300, seed=2)
    a, b = adx(df, 14, 14), adx(df.iloc[:-1], 14, 14)
    np.testing.assert_allclose(a["adx"].to_numpy()[:-1], b["adx"].to_numpy(), equal_nan=True)
    fa, fb = fisher(df, 10), fisher(df.iloc[:-1], 10)
    np.testing.assert_allclose(fa["fisher"].to_numpy()[:-1], fb["fisher"].to_numpy())
