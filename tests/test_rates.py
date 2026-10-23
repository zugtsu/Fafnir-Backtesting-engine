"""Rollover financing: which 17:00 New York instants are charged, which rate
is in force at each, the Wednesday triple, the sign, and the HistData clock."""
import pandas as pd
import pytest

from saty_bt.data.rates import financing_usd, parse_table, rate_at, rollovers

TABLE = parse_table(pd.DataFrame([
    {"ccy": "USD", "announced_date": "2024-01-01", "announced_time_local": "00:00", "tz": "America/New_York", "rate_pct": 5.0},
    # a cut announced 14:00 NY on Wednesday 2024-07-10: the 17:00 roll that day already uses it
    {"ccy": "USD", "announced_date": "2024-07-10", "announced_time_local": "14:00", "tz": "America/New_York", "rate_pct": 4.0},
    {"ccy": "JPY", "announced_date": "2024-01-01", "announced_time_local": "00:00", "tz": "Asia/Tokyo", "rate_pct": 0.0},
]))


def ny(s):
    return pd.Timestamp(s, tz="America/New_York")


def test_rollovers_weekdays_and_triple_wednesday():
    inst, w = rollovers(ny("2024-07-08 10:00"), ny("2024-07-11 10:00"))      # Mon -> Thu
    assert list(inst.tz_convert("America/New_York").strftime("%a %H:%M")) == ["Mon 17:00", "Tue 17:00", "Wed 17:00"]
    assert list(w) == [1, 1, 3]
    inst, w = rollovers(ny("2024-07-12 16:00"), ny("2024-07-15 10:00"))      # Fri -> Mon
    assert list(w) == [1]
    assert list(rollovers(ny("2024-07-12 16:00"), ny("2024-07-15 10:00"), "triple_fri")[1]) == [3]


def test_rollover_boundaries_and_dst():
    # entry exactly at 17:00 is after that roll; exit exactly at 17:00 pays it
    assert len(rollovers(ny("2024-07-09 17:00"), ny("2024-07-10 16:59"))[0]) == 0
    assert len(rollovers(ny("2024-07-09 16:59"), ny("2024-07-09 17:00"))[0]) == 1
    summer = rollovers(ny("2024-07-09 12:00"), ny("2024-07-09 18:00"))[0]
    winter = rollovers(ny("2024-01-09 12:00"), ny("2024-01-09 18:00"))[0]
    assert summer[0].hour == 21 and winter[0].hour == 22                     # real UTC


def test_rate_in_force_is_last_announcement_before():
    r = rate_at("USD", pd.DatetimeIndex([ny("2024-07-09 17:00"), ny("2024-07-10 17:00")]).tz_convert("UTC"), TABLE)
    assert list(r) == [5.0, 4.0]
    with pytest.raises(ValueError):
        rate_at("USD", pd.DatetimeIndex([ny("2023-06-01 17:00")]).tz_convert("UTC"), TABLE)


def test_sign_markup_and_amount():
    kw = dict(entry=ny("2024-07-08 10:00").tz_convert("UTC"), exit=ny("2024-07-09 10:00").tz_convert("UTC"),
              asset=None, table=TABLE)
    # one roll (Mon), USD 5% vs JPY 0%: long USDJPY earns, short pays; markup always costs
    long_ = financing_usd(10_000, "USD", "JPY", +1, markup_pct=0.0, **kw)
    short = financing_usd(10_000, "USD", "JPY", -1, markup_pct=0.0, **kw)
    assert long_ == pytest.approx(10_000 * 5.0 / 100 / 365)
    assert short == pytest.approx(-long_)
    assert financing_usd(10_000, "USD", "JPY", +1, markup_pct=1.0, **kw) == pytest.approx(10_000 * 4.0 / 100 / 365)
    assert financing_usd(10_000, "USD", "JPY", -1, markup_pct=1.0, **kw) == pytest.approx(-10_000 * 6.0 / 100 / 365)


def test_histdata_stamps_are_converted():
    # HistData stamp = NY wall + 5h. Tuesday 2024-07-09 stamps 21:00 -> 23:00 are
    # 16:00 -> 18:00 New York (summer): one roll. Read as real UTC they would be
    # 17:00 -> 19:00 EDT and miss it.
    e, x = pd.Timestamp("2024-07-09 21:00", tz="UTC"), pd.Timestamp("2024-07-09 23:00", tz="UTC")
    as_hist = financing_usd(10_000, "USD", "JPY", +1, e, x, 0.0, asset="USDJPY", table=TABLE)
    as_utc = financing_usd(10_000, "USD", "JPY", +1, e, x, 0.0, asset=None, table=TABLE)
    assert as_hist == pytest.approx(10_000 * 5.0 / 100 / 365) and as_utc == 0.0


def test_shipped_table_covers_the_data():
    from saty_bt.data.rates import load_table
    t = load_table()
    assert set(t) == {"USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF"}
    first = pd.DatetimeIndex([ny("2014-12-01 17:00")]).tz_convert("UTC")
    for c in t:
        assert t[c][0].is_monotonic_increasing
        rate_at(c, first)                                    # in force from the start of the data
    peak = pd.DatetimeIndex([ny("2023-10-02 17:00")]).tz_convert("UTC")
    assert rate_at("USD", peak)[0] == 5.375 and rate_at("JPY", peak)[0] == -0.1


def test_engine_books_financing_into_pnl():
    import pytest
    from saty_bt.config import load_config, resolve_path
    if not (resolve_path(load_config(), "store") / "USDJPY").exists():
        pytest.skip("USDJPY data not built here")
    from saty_bt.sweep import run_grid
    spec = {"name": "t", "system": "ma_cross", "n_fast": 50, "n_slow": 200, "ma_kind": "sma",
            "stop_atr_k": 3.0, "target_r": 2.0}
    out = {}
    for mode in ("none", "rate_diff"):
        cfg = load_config()
        c = cfg["assets"]["USDJPY"]["costs"]
        c.update(spread=0.014, commission_bps=0.0, holding_bps_per_hour=0.0, financing=mode,
                 financing_markup_pct=1.0)
        out[mode], _ = run_grid(cfg, "indicator_lab", ["USDJPY"], "4h", {"spec": [spec]}, verbose=False)
    a, b = out["none"], out["rate_diff"]
    assert len(a) == len(b) and a.financing_usd.abs().sum() == 0
    assert (b.pnl_usd - a.pnl_usd).sum() == pytest.approx(b.financing_usd.sum(), rel=1e-9, abs=1e-6)
    longs_2023 = b[(b.direction == "long") & (pd.to_datetime(b.entry_time).dt.year == 2023)]
    assert (longs_2023.financing_usd > 0).all()              # USD 5%+ over JPY -0.1%, less 1%
