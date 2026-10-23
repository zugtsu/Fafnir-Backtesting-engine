"""Kaufman round tooling: efficiency ratio, KAMA, Kaufman's filtered KAMA
signal, the breakout channel, the ER filter, the shared bar simulator and the
matched-exposure null. Synthetic data only - no strategy is run on prices."""
import numpy as np
import pandas as pd
import pytest

from saty_bt.data.rates import financing_usd, parse_table
from saty_bt.indicators.core import atr as wilder_atr
from saty_bt.indicators.kaufman import (breakout_position, breakout_signals, efficiency_ratio,
                                        er_pass, kama, kama_filter, kama_state)
from saty_bt.sim import (Costs, FXFinancing, matched_exposure_null, open_to_open, series_stats,
                         simulate)
from saty_bt.validate import assert_causal


def _s(vals, start="2024-01-01", freq="1D"):
    return pd.Series(np.asarray(vals, float),
                     index=pd.date_range(start, periods=len(vals), freq=freq, tz="UTC"))


def _walk(n=1500, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.concatenate([[100.0], c[:-1]]) * np.exp(rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n)))
    idx = pd.date_range("2016-01-01", periods=n, freq="1D", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


# ── efficiency ratio and KAMA ────────────────────────────────────────

def test_er_straight_line_is_one_and_warmup_is_nan():
    er = efficiency_ratio(_s(100 + 0.5 * np.arange(60)), 10)
    assert er.iloc[:10].isna().all()
    assert np.allclose(er.iloc[10:], 1.0)
    down = efficiency_ratio(_s(100 - 0.5 * np.arange(60)), 20)
    assert np.allclose(down.iloc[20:], 1.0)


def test_er_alternating_is_noise_and_flat_is_zero():
    alt = _s(np.tile([1.0, 2.0], 40))
    assert np.allclose(efficiency_ratio(alt, 10).iloc[10:], 0.0)          # even n: exactly 0
    assert np.allclose(efficiency_ratio(alt, 9).iloc[9:], 1 / 9)          # odd n: one net step in 9
    assert (efficiency_ratio(_s(np.full(40, 5.0)), 10).iloc[10:] == 0.0).all()   # 0/0 -> 0


def _er_literal(c, n):
    out = [np.nan] * len(c)
    for t in range(n, len(c)):
        den = sum(abs(c[i] - c[i - 1]) for i in range(t - n + 1, t + 1))
        out[t] = abs(c[t] - c[t - n]) / den if den > 0 else 0.0
    return np.array(out)


def _kama_literal(c, n, fast=2, slow=30):
    er = _er_literal(c, n)
    out = [np.nan] * len(c)
    out[n] = c[n]
    for t in range(n + 1, len(c)):
        sc = (er[t] * (2 / (fast + 1) - 2 / (slow + 1)) + 2 / (slow + 1)) ** 2
        out[t] = out[t - 1] + sc * (c[t] - out[t - 1])
    return np.array(out)


@pytest.mark.parametrize("n", [10, 20])
def test_er_and_kama_match_a_literal_loop(n):
    c = _walk(400).close
    cl = list(c.to_numpy())
    assert np.allclose(efficiency_ratio(c, n).to_numpy(), _er_literal(cl, n), equal_nan=True, rtol=1e-12)
    k = kama(c, n).to_numpy()
    lit = _kama_literal(cl, n)
    assert np.isnan(k[:n]).all() and k[n] == cl[n]
    assert np.allclose(k, lit, equal_nan=True, rtol=1e-12, atol=0)


# ── Kaufman's filtered KAMA signal ───────────────────────────────────

def test_kama_state_k0_any_turn_counts():
    K = _s([10, 11, 12, 11.5, 11, 11.2, 13, 12.9, 12.95, 12.8])
    # tracking starts at row 2 (first SD of 2 changes); every turn flips
    assert list(kama_state(K, k=0.0, sd_n=2)) == [0, 0, 0, -1, -1, 1, 1, -1, 1, -1]


def test_kama_state_filter_blocks_small_turns():
    K = _s([0, 1, 2, 3, 2.5, 2.0, 1.0, 1.5, 3.0, 2.9])
    f = kama_filter(K, k=1.0, sd_n=2)
    assert np.allclose(f, [np.nan, np.nan, 0, 0, 0.75, 0, 0.25, 0.75, 0.5, 0.8], equal_nan=True)
    # row 4: 3 -> 2.5 is a 0.5 drop < f 0.75 (blocked); row 5: f = 0, drop 1.0 -> short;
    # row 7: +0.5 off the low 1.0 < f 0.75 (blocked); row 8: +2.0 >= 0.5 -> long;
    # row 9: -0.1 < 0.8 (blocked)
    assert list(kama_state(K, k=1.0, sd_n=2)) == [0, 0, 0, 1, 1, -1, -1, -1, 1, 1]
    assert list(kama_state(K, k=0.0, sd_n=2)) == [0, 0, 0, 1, -1, -1, -1, 1, 1, -1]


def test_kama_state_warmup_is_seed_plus_sd_n():
    c = _walk(200).close
    K = kama(c, 10)
    st = kama_state(K, k=0.1)
    first_filter = 10 + 20           # KAMA seeded at row 10, 20 changes later
    assert kama_filter(K, 0.1).iloc[:first_filter].isna().all()
    assert np.isfinite(kama_filter(K, 0.1).iloc[first_filter])
    assert (st.iloc[:first_filter + 1] == 0).all()
    assert set(st.unique()) <= {-1, 0, 1} and (st.iloc[-50:] != 0).all()


# ── breakout channel, ER filter, breakout position ───────────────────

def test_breakout_channel_uses_prior_closes():
    c = _s([1, 2, 3, 4, 3.5, 2, 1, 5])
    s = breakout_signals(c, entry_n=3, exit_n=2)
    assert list(s.long_entry) == [False, False, False, True, False, False, False, True]
    assert list(s.short_entry) == [False, False, False, False, False, True, True, False]
    assert list(s.long_exit) == [False, False, False, False, False, True, True, False]
    assert list(s.short_exit) == [False, False, True, True, False, False, False, True]
    assert s.entry_hi.iloc[3] == 3.0                       # max of rows 0..2, not row 3
    assert list(breakout_position(s)) == [0, 0, 0, 1, 1, -1, -1, 1]
    assert list(breakout_position(s, long_only=True)) == [0, 0, 0, 1, 1, 0, 0, 1]
    gate = pd.Series(True, index=c.index)
    gate.iloc[5] = False                                    # the short entry at row 5 is filtered out
    assert list(breakout_position(s, allow=gate)) == [0, 0, 0, 1, 1, 0, -1, 1]


def test_er_pass_trailing_quantile_includes_t():
    er = _s([np.nan, np.nan, .1, .2, .3, .4, .1, .9])
    assert list(er_pass(er, window=4, q=0.5)) == [False] * 5 + [True, False, True]
    x = _walk(600).close.pct_change().abs()
    p = er_pass(x, window=50, q=0.67)
    t = 400
    assert p.iloc[t] == (x.iloc[t] > np.quantile(x.iloc[t - 49:t + 1].to_numpy(), 0.67))


def test_kaufman_features_are_causal():
    b = _walk(1500)

    def fn(df):
        c = df["close"]
        er = efficiency_ratio(c, 20)
        K = kama(c, 10)
        sig = breakout_signals(c, 20, 10)
        ok = er_pass(er, 250, 0.5)
        return pd.DataFrame({
            "er": er, "kama": K,
            "state0": kama_state(K, 0.0), "state1": kama_state(K, 1.0),
            "le": sig.long_entry.astype(int), "se": sig.short_entry.astype(int),
            "lx": sig.long_exit.astype(int), "sx": sig.short_exit.astype(int),
            "er_pass": ok.astype(int),
            "pos": breakout_position(sig, allow=ok),
        }, index=df.index)

    assert_causal(fn, b, ["er", "kama", "state0", "state1", "le", "se", "lx", "sx", "er_pass", "pos"],
                  tol=0.0, name="kaufman")


# ── simulator ────────────────────────────────────────────────────────

def _bars(o, h=None, l=None, c=None, freq="1D", start="2024-01-01"):
    o = np.asarray(o, float)
    c = o if c is None else np.asarray(c, float)
    h = np.maximum(o, c) + 0.5 if h is None else np.asarray(h, float)
    l = np.minimum(o, c) - 0.5 if l is None else np.asarray(l, float)
    idx = pd.date_range(start, periods=len(o), freq=freq, tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def _on(b, vals):
    return pd.Series(np.asarray(vals, float), index=b.index)


def test_exact_stop_and_reverse_at_one_open():
    b = _bars(100 + np.arange(10.0), c=100.5 + np.arange(10.0))
    want = _on(b, [0, 1, 1, 1, -1, -1, -1, 0, 0, 0])
    t, x = simulate(b, want, _on(b, np.ones(10)))
    assert len(t) == 2
    a, s = t.iloc[0], t.iloc[1]
    assert (a.direction, s.direction) == (1, -1)
    assert a.entry_time == b.index[2] and a.signal_time == b.index[1] and a.entry_price == 102.0
    assert a.exit_time == s.entry_time == b.index[5]           # same open
    assert a.exit_price == s.entry_price == 105.0
    assert a.reason == "reverse" and s.reason == "signal"
    assert s.exit_time == b.index[8] and s.exit_price == 108.0
    assert list(x.pos) == [0, 0, 1, 1, 1, -1, -1, -1, 0, 0]
    assert a.gross == pytest.approx(3 / 102) and s.gross == pytest.approx(-3 / 105)


def test_long_only_goes_flat_instead_of_short():
    b = _bars(100 + np.arange(10.0))
    want = _on(b, [0, 1, 1, 1, -1, -1, -1, 0, 0, 0])
    t, x = simulate(b, want, _on(b, np.ones(10)), long_only=True)
    assert len(t) == 1 and t.direction.iloc[0] == 1
    assert t.exit_time.iloc[0] == b.index[5] and t.reason.iloc[0] == "signal"
    assert list(x.pos) == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_stop_fills_at_stop_and_r_is_minus_one():
    o = np.full(8, 100.0)
    low = o - 0.5
    low[3] = 89.0                                              # stop = 100 - 5 x 2 = 90
    b = _bars(o, l=low)
    t, x = simulate(b, _on(b, [0, 1, 1, 1, 1, 1, 1, 1]), _on(b, np.full(8, 2.0)))
    r = t.iloc[0]
    assert r.reason == "stop" and r.exit_price == 90.0 and r.stop_price == 90.0
    assert r.exit_time == b.index[4]                           # end of the stopped bar
    assert r.r_multiple == pytest.approx(-1.0) and r.ret == pytest.approx(-0.10)
    assert list(x.pos) == [0, 0, 1, 1, 0, 0, 0, 0] and x.pos_close.iloc[3] == 0
    # with a bar_close column the stop is stamped at that bar's close
    b2 = b.assign(bar_close=b.index + pd.Timedelta(hours=7))
    assert simulate(b2, _on(b2, [0, 1, 1, 1, 1, 1, 1, 1]), _on(b2, np.full(8, 2.0))).trades.exit_time.iloc[0] \
        == b.index[3] + pd.Timedelta(hours=7)


def test_stop_checked_on_entry_bar_after_the_open():
    o = np.full(6, 100.0)
    low = o - 0.5
    low[2] = 80.0
    b = _bars(o, l=low)
    t, _ = simulate(b, _on(b, [0, 1, 1, 1, 1, 1]), _on(b, np.full(6, 2.0)))
    assert t.entry_bar.iloc[0] == t.exit_bar.iloc[0] == 2 and t.exit_price.iloc[0] == 90.0


def test_gap_through_stop_fills_at_open_short_side_too():
    o = np.full(8, 100.0)
    o[3] = 85.0
    b = _bars(o)
    t, _ = simulate(b, _on(b, [0, 1, 1, 1, 1, 1, 1, 1]), _on(b, np.full(8, 2.0)))
    r = t.iloc[0]
    assert r.reason == "gap" and r.exit_price == 85.0 and r.exit_time == b.index[3]
    assert r.r_multiple == pytest.approx(-1.5)
    o = np.full(8, 100.0)
    o[4] = 112.0
    b = _bars(o)
    t, _ = simulate(b, _on(b, [0, -1, -1, -1, -1, -1, -1, -1]), _on(b, np.full(8, 2.0)))
    assert t.reason.iloc[0] == "gap" and t.exit_price.iloc[0] == 112.0
    assert t.r_multiple.iloc[0] == pytest.approx(-1.2)


def test_no_reentry_after_stop_until_desired_changes():
    o = np.full(14, 100.0)
    low = o - 0.5
    low[3] = 89.0
    b = _bars(o, l=low)
    atr = _on(b, np.full(14, 2.0))
    want = _on(b, [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1])
    t, x = simulate(b, want, atr)
    assert list(t.entry_bar) == [2, 10]                        # flat 4..9, back after 1 -> 0 -> 1
    assert (x.pos.iloc[4:10] == 0).all() and x.pos.iloc[10] == 1
    assert t.reason.iloc[-1] == "end" and t.exit_time.iloc[-1] == b.index[-1] + pd.Timedelta(days=1)
    # a flip straight to the other side is a change: enter short at the next open
    want2 = _on(b, [0, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1])
    t2, _ = simulate(b, want2, atr)
    assert list(t2.entry_bar) == [2, 7] and list(t2.direction) == [1, -1]
    # long-only: -1 maps to 0, which releases the lock; the next +1 re-enters
    want3 = _on(b, [0, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1])
    assert list(simulate(b, want3, atr, long_only=True).trades.entry_bar) == [2, 7]


def test_costs_charged_once_per_side():
    b = _bars(np.full(8, 100.0))
    want = _on(b, [0, 1, 1, -1, -1, 0, 0, 0])
    cst = Costs(spread=0.02, fee_side=0.001, hold_bps_hour=1.0)
    t, x = simulate(b, want, _on(b, np.ones(8)), costs=cst)
    side = 0.01 / 100 + 0.001
    assert list(t.entry_bar) == [2, 4] and list(t.exit_bar) == [4, 6]
    assert np.allclose(t.hours, [48.0, 48.0])
    assert np.allclose(t.cost, 2 * side + 48 * 1e-4)
    assert np.allclose(t.ret, -(2 * side + 48 * 1e-4)) and np.allclose(t.gross, 0.0)
    assert np.allclose(t.pnl_usd, t.ret * 10_000)
    # bar 2: one side + a day of holding; bar 4: exit side + entry side + a day; bar 6: exit side only
    assert x.cost.iloc[2] == pytest.approx(side + 24e-4)
    assert x.cost.iloc[4] == pytest.approx(2 * side + 24e-4)
    assert x.cost.iloc[6] == pytest.approx(side)
    assert x.ret.sum() == pytest.approx(t.ret.sum(), abs=1e-15)
    t2, _ = simulate(b, want, _on(b, np.ones(8)), costs=cst.scaled(2.0))
    assert np.allclose(t2.ret, 2 * t.ret)
    frac = simulate(b, want, _on(b, np.ones(8)), costs=Costs(spread_frac=0.0002)).trades
    assert np.allclose(frac.ret, -0.0002)
    with pytest.raises(ValueError):
        Costs(spread=0.01, spread_frac=0.001)


def test_exit_fee_is_on_exit_notional():
    # a winner from 100 to 300: a 0.4% fee is 0.4% of $100 in and 0.4% of $300 out,
    # i.e. 1.6% of the entry notional in all; the price-unit spread stays fixed
    b = _bars(np.r_[np.full(3, 100.0), np.full(5, 300.0)])
    want = _on(b, [0, 1, 1, 1, 1, 0, 0, 0])
    t = simulate(b, want, _on(b, np.full(8, 1000.0)), costs=Costs(fee_side=0.004)).trades
    assert t.cost.iloc[0] == pytest.approx(0.004 + 0.004 * 3.0)
    assert t.ret.iloc[0] == pytest.approx(2.0 - 0.016)
    s = simulate(b, want, _on(b, np.full(8, 1000.0)), costs=Costs(spread=0.02)).trades
    assert s.cost.iloc[0] == pytest.approx(0.02 / 100)       # half spread each side, in price units
    f = simulate(b, want, _on(b, np.full(8, 1000.0)), costs=Costs(spread_frac=0.0002)).trades
    assert f.cost.iloc[0] == pytest.approx(0.0001 + 0.0001 * 3.0)


def test_bar_returns_sum_to_trade_returns():
    b = _walk(800, seed=3)
    rng = np.random.default_rng(5)
    want = pd.Series(np.repeat(rng.choice([-1, 0, 1], 80), 10).astype(float), index=b.index)
    a = wilder_atr(b, 14) * 0.5                                 # tight stops: plenty of stop/gap exits
    t, x = simulate(b, want, a, costs=Costs(spread_frac=0.0004, fee_side=0.0002, hold_bps_hour=0.1))
    assert {"stop", "reverse", "signal"} <= set(t.reason)
    assert x.ret.sum() == pytest.approx(t.ret.sum(), rel=1e-10)
    assert x.gross.sum() == pytest.approx(t.gross.sum(), rel=1e-10)
    for _, r in t.iterrows():                                   # each trade's gross telescopes on its bars
        seg = x.gross.iloc[r.entry_bar:r.exit_bar + 1]
        if r.exit_bar > r.entry_bar and (t.exit_bar == r.entry_bar).sum() == 0 \
                and (t.entry_bar == r.exit_bar).sum() == 0:
            assert seg.sum() == pytest.approx(r.gross, abs=1e-12)
    assert ((x.pos != 0) | (x.pos_close == 0)).all()
    assert (t.entry_time.to_numpy()[1:] >= t.exit_time.to_numpy()[:-1]).all()


def test_simulator_is_causal():
    # cutting the future off cannot change any trade that closed before the cut,
    # nor any per-bar value before the last kept bar
    b = _walk(900, seed=4)
    rng = np.random.default_rng(8)
    want = pd.Series(np.repeat(rng.choice([-1, 0, 1], 90), 10).astype(float), index=b.index)
    a = wilder_atr(b, 14)
    cst = Costs(spread_frac=0.0003, fee_side=0.0001, hold_bps_hour=0.05)
    full_t, full_x = simulate(b, want, a, costs=cst)
    for cut in (300, 555, 777):
        t, x = simulate(b.iloc[:cut], want.iloc[:cut], a.iloc[:cut], costs=cst)
        done = t[t.reason != "end"].reset_index(drop=True)
        ref = full_t.iloc[:len(done)].reset_index(drop=True)
        pd.testing.assert_frame_equal(done, ref)
        pd.testing.assert_frame_equal(x.iloc[:cut - 1][["pos", "pos_close", "gross"]],
                                      full_x.iloc[:cut - 1][["pos", "pos_close", "gross"]])


def test_entry_skipped_without_atr():
    b = _bars(np.full(6, 100.0))
    a = _on(b, [np.nan, np.nan, 1, 1, 1, 1])
    t, _ = simulate(b, _on(b, [1, 1, 1, 1, 1, 1]), a)
    assert list(t.entry_bar) == [3] and t.attrs["skipped_no_atr"] == 2
    with pytest.raises(ValueError):
        simulate(b, _on(b, [0, 2, 0, 0, 0, 0]), a)
    with pytest.raises(ValueError):
        simulate(b, pd.Series(0.0, index=b.index[:-1]), a)


TABLE = parse_table(pd.DataFrame([
    {"ccy": "USD", "announced_date": "2024-01-01", "announced_time_local": "00:00", "tz": "America/New_York", "rate_pct": 5.0},
    {"ccy": "JPY", "announced_date": "2024-01-01", "announced_time_local": "00:00", "tz": "Asia/Tokyo", "rate_pct": 0.0},
]))


def test_fx_financing_sign_amount_and_bar():
    # HistData-style stored stamps (New York wall + 5h), 4h bars from Mon 2024-07-08 02:00
    b = _bars(np.full(36, 150.0), freq="4h", start="2024-07-08 02:00")
    b["bar_close"] = b.index + pd.Timedelta(hours=4)
    a = _on(b, np.ones(36))
    fx = FXFinancing("USDJPY", markup_pct=0.0, table=TABLE)
    want = _on(b, [1] * 30 + [0] * 6)
    tl, xl = simulate(b, want, a, costs=Costs(spread=0.014, fx=fx))
    ts, _ = simulate(b, -want, a, costs=Costs(spread=0.014, fx=fx))
    assert tl.financing.iloc[0] > 0 and ts.financing.iloc[0] == pytest.approx(-tl.financing.iloc[0])
    r = tl.iloc[0]
    exp = financing_usd(10_000, "USD", "JPY", 1, r.entry_time, r.exit_time, 0.0,
                        asset="USDJPY", table=TABLE) / 10_000
    assert r.financing == pytest.approx(exp)
    # entry Mon 06:00 stored (01:00 NY), exit Sat 06:00 stored (01:00 NY): rolls Mon..Fri, Wed x3
    assert r.financing == pytest.approx((1 + 1 + 3 + 1 + 1) * 5.0 / 100 / 365)
    # booked on the bar stamped 18:00 (13:00-17:00 New York), one per weekday
    fb = xl.financing[xl.financing != 0]
    assert set(fb.index.hour) == {18} and len(fb) == 5
    assert xl.financing.sum() == pytest.approx(r.financing)
    assert r.ret == pytest.approx(r.gross - 0.014 / 150 + r.financing)
    # markup is charged both ways and doubles under scaled(2); the differential does not
    fx1 = FXFinancing("USDJPY", markup_pct=1.0, table=TABLE)
    l1 = simulate(b, want, a, costs=Costs(fx=fx1)).trades.financing.iloc[0]
    l2 = simulate(b, want, a, costs=Costs(fx=fx1).scaled(2.0)).trades.financing.iloc[0]
    assert l1 == pytest.approx(7 * 4.0 / 100 / 365) and l2 == pytest.approx(7 * 3.0 / 100 / 365)
    # bars already on real UTC: no second conversion
    bu = b.copy()
    bu.index = bu.index - pd.Timedelta(hours=1)                 # summer: real UTC = stored - 1h
    bu["bar_close"] = bu.index + pd.Timedelta(hours=4)
    fxu = FXFinancing("USDJPY", markup_pct=0.0, table=TABLE, stored_stamps=False)
    tu, xu = simulate(bu, _on(bu, want.to_numpy()), _on(bu, np.ones(36)), costs=Costs(fx=fxu))
    assert set(xu.financing[xu.financing != 0].index.hour) == {17}  # 17:00-21:00 UTC holds 17:00 NY
    assert tu.financing.iloc[0] == pytest.approx(r.financing)


# ── matched-exposure null and stats ──────────────────────────────────

def _runs(n, seed, on=(5, 40), off=(5, 60)):
    """Persistent in/out series: alternating out-gaps and in-runs."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n, bool)
    i = 0
    while i < n:
        a, b = rng.integers(*on), rng.integers(*off)
        i += b
        x[i:i + a] = True
        i += a
    return x


def _n_entries(m):
    m = np.atleast_2d(m)
    return m[:, :1].sum(axis=1) + (m[:, 1:] & ~m[:, :-1]).sum(axis=1)


def _idx(n):
    return pd.date_range("2018-01-01", periods=n, freq="1D", tz="UTC")


def test_null_keeps_invested_fraction_and_entry_count():
    # little persistence, so plain 20-bar block shuffles match the entry count often
    n = 900
    x = np.random.default_rng(2).random(n) < 0.4
    idx = _idx(n)
    r = pd.Series(np.random.default_rng(3).normal(0.0005, 0.02, n), index=idx)
    res = matched_exposure_null(pd.Series(x, index=idx), r, cost_per_entry=0.008, n=300,
                                block=20, seed=7, return_series=True)
    S = res["series"]
    assert res["method"] == "block" and res["blocks"] == 45
    assert S.shape == (300, n) and res["n"] == 300
    assert (S.sum(axis=1) == x.sum()).all()
    assert (_n_entries(S) == res["entries"]).all() and res["entries"] == _n_entries(x)[0]
    assert 0 < res["acceptance_rate"] < 1 and res["draws"] >= 300
    assert len(np.unique(S, axis=0)) == 300                    # actually random
    # the rule is scored exactly like the draws
    e = x.copy()
    e[1:] = x[1:] & ~x[:-1]
    s = x * r.to_numpy() - 0.008 * e
    assert res["observed"]["log_return"] == pytest.approx(np.log1p(s).sum())
    assert res["observed"]["sharpe"] == pytest.approx(s.mean() / s.std(ddof=1))
    assert res["p_log"] == pytest.approx(np.mean(res["log_return"] >= res["observed"]["log_return"]))
    assert res["q95_sharpe"] == pytest.approx(np.quantile(res["sharpe"], 0.95))
    again = matched_exposure_null(pd.Series(x, index=idx), r, 0.008, n=300, block=20, seed=7)
    assert np.array_equal(again["log_return"], res["log_return"])


def test_null_fails_loudly_when_blocks_cannot_match_a_persistent_rule():
    # block edges cut holding runs, so shuffles carry ~10 more entries than this rule
    n = 900
    x = pd.Series(_runs(n, 2), index=_idx(n))
    r = pd.Series(0.0, index=x.index)
    with pytest.raises(RuntimeError, match="matched the entry count"):
        matched_exposure_null(x, r, 0.0, n=100, max_draws=20_000)
    with pytest.warns(RuntimeWarning):
        part = matched_exposure_null(x, r, 0.0, n=100, max_draws=20_000, allow_partial=True)
    assert part["n"] < 100 and part["shuffle_entries_median"] > part["entries"]


@pytest.mark.parametrize("lead_in", [False, True])
def test_null_snap_variant_matches_exactly(lead_in):
    n = 900
    x = _runs(n, 2)
    x[:7] = lead_in                                             # also: starting in the market
    idx = _idx(n)
    r = pd.Series(np.random.default_rng(4).normal(0, 0.02, n), index=idx)
    res = matched_exposure_null(pd.Series(x, index=idx), r, 0.004, n=500, seed=3, snap=True,
                                return_series=True)
    S = res["series"]
    assert res["method"] == "snap" and res["acceptance_rate"] == 1.0 and res["n"] == 500
    assert (S.sum(axis=1) == x.sum()).all() and (_n_entries(S) == _n_entries(x)[0]).all()
    assert len(np.unique(S, axis=0)) == 500
    if lead_in:
        assert S[:, :7].all()                                   # the bar-0 block stays first
    # every snapped block edge is an out-of-market bar
    from saty_bt.sim import _block_starts
    st = _block_starts(x, 20, snap=True)
    assert (~x[st[1:]]).all() and np.all(np.diff(st) > 0)


def test_null_pooled_equal_weight_columns():
    n = 600
    idx = _idx(n)
    X = pd.DataFrame({"A": _runs(n, 4), "B": _runs(n, 5)}, index=idx)
    R = pd.DataFrame(np.random.default_rng(6).normal(0, 0.01, (n, 2)), index=idx, columns=["A", "B"])
    res = matched_exposure_null(X, R, {"A": 0.0002, "B": 0.0004}, n=100, seed=1,
                                periods_per_year=252, snap=True, return_series=True)
    for c in "AB":
        S = res["series"][c]
        assert (S.sum(axis=1) == X[c].sum()).all()
        assert (_n_entries(S) == res["entries"][c]).all()
    sA = X.A.to_numpy() * R.A.to_numpy()
    sB = X.B.to_numpy() * R.B.to_numpy()
    eA = X.A.to_numpy() & ~np.r_[False, X.A.to_numpy()[:-1]]
    eB = X.B.to_numpy() & ~np.r_[False, X.B.to_numpy()[:-1]]
    s = (sA - 0.0002 * eA + sB - 0.0004 * eB) / 2
    assert res["observed"]["sharpe"] == pytest.approx(s.mean() / s.std(ddof=1) * np.sqrt(252))
    assert len(res["sharpe"]) == 100
    # the null's portfolio is the equal-weight mean of the two columns' draws
    SA, SB = res["series"]["A"], res["series"]["B"]
    eSA, eSB = SA.copy(), SB.copy()
    eSA[:, 1:] = SA[:, 1:] & ~SA[:, :-1]
    eSB[:, 1:] = SB[:, 1:] & ~SB[:, :-1]
    p = (SA * R.A.to_numpy() - 0.0002 * eSA + SB * R.B.to_numpy() - 0.0004 * eSB) / 2
    assert np.allclose(res["log_return"], np.log1p(p).sum(axis=1))


def test_series_stats_and_open_to_open():
    st = series_stats(pd.Series([0.1, -0.5, 0.2]))
    assert st["total_return"] == pytest.approx(1.1 * 0.5 * 1.2 - 1)
    assert st["max_drawdown"] == pytest.approx(-0.5)
    assert st["log_return"] == pytest.approx(np.log(1.1 * 0.5 * 1.2))
    assert series_stats(pd.Series([-0.1, 0.05]))["max_drawdown"] == pytest.approx(-0.1)
    b = _bars([100.0, 110.0, 99.0])
    assert np.allclose(open_to_open(b).to_numpy(), [0.1, -0.1, np.nan], equal_nan=True)


def test_sim_pos_scored_by_null_matches_sim_without_stops_or_costs():
    # with no stops hit and no costs, the null's scorer on sim.bars.pos with
    # open-to-open returns reproduces the simulator's own gross (to compounding)
    b = _walk(400, seed=9)
    rng = np.random.default_rng(2)
    want = pd.Series(np.repeat(rng.choice([0, 1], 40), 10).astype(float), index=b.index)
    t, x = simulate(b, want, pd.Series(1e6, index=b.index), long_only=True)
    o2o = open_to_open(b).fillna(0.0)
    held = (x.pos > 0)
    # sum of open-to-open simple returns over each trade vs its gross: same path, different compounding
    s = sum(np.prod(1 + o2o.iloc[r.entry_bar:r.exit_bar]) - 1 for _, r in t[t.reason != "end"].iterrows())
    g = t[t.reason != "end"].gross.sum()
    assert s == pytest.approx(g, rel=1e-9)
    assert held.sum() == sum(r.exit_bar - r.entry_bar for _, r in t[t.reason != "end"].iterrows()) \
        + sum(len(b) - r.entry_bar for _, r in t[t.reason == "end"].iterrows())
