"""
Engine and strategy tests on a synthetic feature frame: fills happen next
bar, stops resolve before targets, the close-through stop exits next
open, flat at the session close, and every strategy builder emits the
column contract the engine needs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saty_bt.config import asset_config, load_config
from saty_bt.engine import run_backtest
from saty_bt.indicators import atr_levels, phase_oscillator, pivot_ribbon, volume_stack
from saty_bt.indicators.core import atr, swing_highs, swing_lows
from saty_bt.strategies import STRATEGIES

CFG = load_config()
ACFG = asset_config(CFG, "EURUSD")
ACFG["costs"] = {"spread": 0.0, "slippage": 0.0, "slippage_bps": 0.0, "commission_bps": 0.0}
RISK = {"sizing": "fixed_margin", "margin_usd": 200, "leverage": 5}
NO_LIMITS = {"halt_drawdown_pct": 100, "daily_loss_limit_pct": 100, "consecutive_loss_limit": 10**9,
             "max_trades_per_asset_per_day": 100, "cooldown_bars_after_stop": 0, "min_stop_atr": 0.0}


def features(n_days=40, seed=3):
    """30m FX-like bars with a full feature set, deterministic."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 22:00", periods=n_days * 48, freq="30min", tz="UTC")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0004, len(idx)))
    opn = np.r_[close[0], close[:-1]]
    df = pd.DataFrame({"open": opn, "high": np.maximum(opn, close) + 0.0002, "low": np.minimum(opn, close) - 0.0002,
                       "close": close, "volume": np.nan}, index=idx)
    df.index.name = "timestamp"
    df["bar_close"] = df.index + pd.Timedelta("30min")
    df["day_open"] = (df.index - pd.Timedelta(hours=22)).normalize() + pd.Timedelta(hours=22)
    df["day_key"] = (df.index + pd.Timedelta(hours=2)).normalize().tz_localize(None)
    daily = df.groupby("day_key").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                      close=("close", "last"), volume=("volume", "sum"))
    daily.index = pd.DatetimeIndex(daily.index).tz_localize("UTC")
    daily["day_key"] = daily.index.tz_localize(None)
    daily["period_key"] = daily["day_key"]
    ind = CFG["indicators"]
    f = atr_levels(df, daily, "1d", ind["atr_levels"])
    f = pivot_ribbon(f, ind["pivot_ribbon"])
    f["htf_ribbon_bull"], f["htf_ribbon_bear"], f["htf_ribbon_state"] = f["ribbon_bull"], f["ribbon_bear"], f["ribbon_state"]
    f = phase_oscillator(f, ind["phase_oscillator"])
    f = volume_stack(f, ind["volume_stack"])
    f["swing_low"], f["swing_high"] = swing_lows(f, 2), swing_highs(f, 2)
    f["chart_atr"] = atr(f, 14)
    return f


@pytest.fixture(scope="module")
def F():
    return features()


def _run(f, sig, scfg):
    return run_backtest(f, sig, ACFG, CFG, scfg, RISK, NO_LIMITS, 1_000_000.0)


def test_every_strategy_emits_the_contract(F):
    need = {"long_entry", "short_entry", "long_setup", "short_setup", "long_stop", "short_stop", "long_target",
            "short_target", "long_runner", "short_runner", "long_trigger", "short_trigger", "stop_mode", "long_gap", "short_gap"}
    for name, fn in STRATEGIES.items():
        scfg = CFG["strategies"][name]
        sig = fn(F, scfg, ACFG, CFG)
        assert need <= set(sig.columns), name
        assert sig["long_entry"].dtype == bool and sig["short_entry"].dtype == bool, name
        tr, eq = _run(F, sig, scfg)
        assert len(eq) == len(F)
        if len(tr):
            assert (tr["exit_time"] >= tr["entry_time"]).all(), name


def test_fill_is_next_bar_open_and_no_same_bar_fill(F):
    scfg = dict(CFG["strategies"]["golden_gate"])
    sig = STRATEGIES["golden_gate"](F, scfg, ACFG, CFG)
    tr, _ = _run(F, sig, scfg)
    assert len(tr) > 0
    pos = {t: i for i, t in enumerate(F.index)}
    for _, row in tr.iterrows():
        si, ei = pos[row["signal_time"]], pos[row["entry_time"]]
        assert ei == si + 1
        assert row["entry_price"] == pytest.approx(F["open"].iloc[ei])


def test_flat_at_period_end_never_holds_overnight(F):
    scfg = dict(CFG["strategies"]["golden_gate"])
    sig = STRATEGIES["golden_gate"](F, scfg, ACFG, CFG)
    tr, _ = _run(F, sig, scfg)
    dk = F["day_key"]
    for _, row in tr.iterrows():
        assert dk.loc[row["entry_time"]] == dk.loc[row["exit_time"]]


def test_close_through_stop_exits_next_open(F):
    scfg = dict(CFG["strategies"]["golden_gate"])
    sig = STRATEGIES["golden_gate"](F, scfg, ACFG, CFG)
    tr, _ = _run(F, sig, scfg)
    ct = tr[tr["exit_reason"] == "stop_close_through"]
    assert len(ct) > 0
    pos = {t: i for i, t in enumerate(F.index)}
    for _, row in ct.iterrows():
        ei = pos[row["exit_time"]]
        # either filled at this bar's open (previous bar closed through) or at the close on the last bar of the day
        assert row["exit_price"] in (pytest.approx(F["open"].iloc[ei]), pytest.approx(F["close"].iloc[ei]))


def test_stop_beats_target_when_both_touched():
    """A hand-built bar that spans both the stop and the target must resolve as a stop."""
    f = features(n_days=6)
    sig = STRATEGIES["golden_gate"](f, CFG["strategies"]["golden_gate"], ACFG, CFG)
    sig[:] = sig  # keep dtypes
    i = 60
    sig.loc[:, "long_entry"] = False
    sig.loc[:, "short_entry"] = False
    sig.iloc[i, sig.columns.get_loc("long_entry")] = True
    sig["stop_mode"] = "atr"
    e = f["open"].iloc[i + 1]
    sig["long_stop"], sig["long_target"], sig["long_runner"] = e - 0.001, e + 0.001, e + 0.002
    f = f.copy()
    f.iloc[i + 1, f.columns.get_loc("high")] = e + 0.002
    f.iloc[i + 1, f.columns.get_loc("low")] = e - 0.002
    tr, _ = run_backtest(f, sig, ACFG, CFG, {"scale_out_pct": 50, "flat_at_period_end": True}, RISK, NO_LIMITS, 1e6)
    assert len(tr) == 1 and tr["exit_reason"].iloc[0] == "stop"
    assert tr["pnl_usd"].iloc[0] < 0


def test_min_stop_distance_skips_degenerate_entries():
    f = features(n_days=40)                      # enough days for the daily ATR to exist
    sig = STRATEGIES["golden_gate"](f, CFG["strategies"]["golden_gate"], ACFG, CFG)
    sig.loc[:, "long_entry"] = False
    sig.loc[:, "short_entry"] = False
    i = 30 * 48
    assert np.isfinite(f["atr"].iloc[i])
    sig.iloc[i, sig.columns.get_loc("long_entry")] = True
    sig["stop_mode"] = "atr"
    sig["long_stop"] = f["open"].shift(-1) - 1e-7          # a hair below the fill
    tr, _ = run_backtest(f, sig, ACFG, CFG, {"scale_out_pct": 50}, RISK, {**NO_LIMITS, "min_stop_atr": 0.05}, 1e6)
    assert len(tr) == 0


def test_random_baseline_is_deterministic(F):
    scfg = CFG["strategies"]["random_entry"]
    a = STRATEGIES["random_entry"](F, scfg, ACFG, CFG)
    b = STRATEGIES["random_entry"](F, scfg, ACFG, CFG)
    pd.testing.assert_series_equal(a["long_entry"], b["long_entry"])
    assert a["long_entry"].sum() + a["short_entry"].sum() > 0
