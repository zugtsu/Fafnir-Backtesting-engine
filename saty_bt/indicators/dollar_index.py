"""
A synthetic U.S. Dollar Index built from our own FX pairs (Kaufman, Trading
Systems and Methods 5th ed., Ch 2 - index construction).

ICE's DXY is a GEOMETRIC average of the dollar against six currencies:

    DXY = 50.14348112 x EURUSD^-0.576 x USDJPY^0.136 x GBPUSD^-0.119
                      x USDCAD^0.091  x USDSEK^0.042 x USDCHF^0.036

i.e. a weighted sum of log(units of each currency per dollar). We build the
same thing from the store's pairs:

    EUR 57.6%  (EURUSD inverted: USD per EUR -> EUR per USD)
    JPY 13.6%  (USDJPY as is)
    GBP 11.9%  (GBPUSD inverted)
    CAD  9.1%  (USDCAD as is)
    SEK  4.2%  ** NOT IN OUR DATA - DROPPED **
    CHF  3.6%  (USDCHF as is)

THE SWEDISH KRONA IS MISSING. Its 4.2% is dropped and the other five weights
are rescaled to sum to 1 (each divided by 0.958): EUR 60.1%, JPY 14.2%,
GBP 12.4%, CAD 9.5%, CHF 3.8%. SEK trades close to EUR, so the loss is small,
but this is NOT the ICE index and its level is not DXY's - it is rebased to
100 at the first stamp where every pair exists. Compare changes, not levels.

    log_index = sum_c  w_c x log(currency c per USD)
    dxy       = base x exp(log_index - log_index at the first complete stamp)

Alignment: on the pairs' own stored stamps (all five are HistData, so they
share one clock - New York wall time + 5h; saty_bt.data.clock.true_utc gives
real UTC). A stamp where ANY pair has no bar gives NaN - a missing bar is not
a price, so nothing is forward-filled. EURUSD, GBPUSD, USDCAD and USDCHF start
2021-01-03 in the store, so the index exists 2021-2026 only; the 2015-2020
USDJPY rows come back NaN.

Causal: each row uses only that row's closes, and the rebasing anchor is the
FIRST complete row, which truncating the future cannot move
(saty_bt.validate.assert_causal checks this in the tests and the report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ICE U.S. Dollar Index weights (sum to 1.0)
DXY_WEIGHTS = {"EUR": 0.576, "JPY": 0.136, "GBP": 0.119, "CAD": 0.091, "SEK": 0.042, "CHF": 0.036}
# currency -> (our pair, invert?)   invert = the pair is quoted USD per currency
COMPONENTS = {"EUR": ("EURUSD", True), "JPY": ("USDJPY", False), "GBP": ("GBPUSD", True),
              "CAD": ("USDCAD", False), "CHF": ("USDCHF", False)}
DROPPED = ("SEK",)          # not in the store
PAIRS = tuple(p for p, _ in COMPONENTS.values())


def index_weights() -> dict[str, float]:
    """DXY weights without SEK, rescaled to sum to 1."""
    keep = {c: w for c, w in DXY_WEIGHTS.items() if c not in DROPPED}
    tot = sum(keep.values())
    return {c: w / tot for c, w in keep.items()}


def dollar_index(closes: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """
    closes: one column per pair (EURUSD, GBPUSD, USDJPY, USDCAD, USDCHF) on a
    shared index; NaN where a pair has no bar. Returns, on the same index:
      ln_EUR .. ln_CHF   log(currency per USD) of each component
      log_index          the weighted sum (NaN unless every component exists)
      dxy                base x exp(log_index - first complete log_index)
    The dollar strengthening against a currency raises its ln_ column and
    therefore the index.
    """
    missing = [p for p in PAIRS if p not in closes.columns]
    if missing:
        raise KeyError(f"dollar_index needs {missing}")
    w = index_weights()
    out = pd.DataFrame(index=closes.index)
    total = pd.Series(0.0, index=closes.index)
    for ccy, (pair, invert) in COMPONENTS.items():
        x = pd.to_numeric(closes[pair], errors="coerce").astype(float)
        x = x.where(x > 0)
        ln = -np.log(x) if invert else np.log(x)
        out[f"ln_{ccy}"] = ln
        total = total + w[ccy] * ln                 # NaN in any component -> NaN
    out["log_index"] = total
    ok = np.flatnonzero(total.notna().to_numpy())
    anchor = float(total.iloc[ok[0]]) if len(ok) else np.nan
    out["dxy"] = float(base) * np.exp(total - anchor)
    return out


def load_closes(cfg: dict, tf: str = "1h") -> pd.DataFrame:
    """The five pairs' closes from the store, outer-joined on stored stamps."""
    from ..data.store import load_bars
    cols = {p: pd.to_numeric(load_bars(cfg, p, tf)["close"], errors="coerce") for p in PAIRS}
    return pd.concat(cols, axis=1, join="outer").sort_index()


def load_dollar_index(cfg: dict, tf: str = "1h", base: float = 100.0) -> pd.DataFrame:
    """dollar_index() on the store's bars (stored HistData stamps)."""
    return dollar_index(load_closes(cfg, tf), base=base)
