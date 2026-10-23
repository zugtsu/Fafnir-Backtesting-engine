"""The HistData clock is New York + 5h; true_utc() must put scheduled US
releases back on their real UTC minute in both seasons."""
import numpy as np
import pandas as pd
import pytest

from saty_bt.config import load_config
from saty_bt.data.clock import true_utc, with_true_utc
from saty_bt.data.store import load_bars


def test_conversion_arithmetic():
    s = pd.DatetimeIndex(["2024-07-05 13:30", "2024-01-05 13:30"], tz="UTC")
    t = true_utc(s, "USDJPY")
    assert list(t.strftime("%H:%M")) == ["12:30", "13:30"]
    assert (true_utc(s, "SPX") == s).all()


def _nfp_minute(d: pd.DataFrame, summer: bool) -> str:
    r = (d.close / d.open - 1).abs()
    ny = r.index.tz_convert("America/New_York").strftime("%z") == "-0400"
    x = r[ny if summer else ~ny]
    x = x[(x.index.dayofweek == 4) & (x.index.day <= 7)]
    (h, m), _ = next(iter(x.groupby([x.index.hour, x.index.minute]).mean()
                          .sort_values(ascending=False).items()))
    return f"{h:02d}:{m:02d}"


@pytest.mark.parametrize("asset", ["USDJPY", "EURUSD"])
def test_jobs_report_lands_on_real_utc(asset):
    try:
        raw = load_bars(load_config(), asset, "3m")
    except FileNotFoundError:
        pytest.skip("no 3m store")
    raw = raw[raw.index >= "2022-01-01"]
    assert _nfp_minute(raw, True) == "13:30"          # the stored clock: wrong in summer
    fixed = with_true_utc(raw, asset)
    assert _nfp_minute(fixed, True) == "12:30"        # 08:30 EDT
    assert _nfp_minute(fixed, False) == "13:30"       # 08:30 EST
    assert fixed.index.is_monotonic_increasing and not fixed.index.has_duplicates
