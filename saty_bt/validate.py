"""
The checks that would have caught our worst bugs, as reusable functions.

Two of them are here because we shipped the bug, not because we were careful.

  assert_causal    Twice in this project a feature read the future and printed
                   a result that did not exist - daily bars stamped at the
                   session OPEN, and 4h bars reindexed forward. Profit factor
                   3.05 and 1.77, both fictional. The test is cheap: recompute
                   the feature on a truncated history and demand that every
                   value up to the cut is bit-identical. A feature that looks
                   ahead cannot pass, because the future it read is gone.

  audit_bars       The bar data itself has lied to us: HistData ships a
                   volume column of zeros (which silently killed VWAP
                   weighting and every volume filter), and its clock is a
                   fixed UTC-5 with no DST. Frozen bars, duplicate stamps and
                   gaps all produce strategies that trade an artefact.

  cost_curve       Our binding constraint is almost never the signal, it is
                   the spread. 1h FX had 3 of 6 families above PF 1.0 before
                   cost and 0 of 6 after. A result without a cost curve is
                   not a result.

Nothing here knows about a strategy. They are guards on the layer underneath.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd


def assert_causal(fn: Callable[[pd.DataFrame], pd.DataFrame], bars: pd.DataFrame,
                  cols: Sequence[str], cuts: Iterable[int] = (0.25, 0.5, 0.75),
                  tol: float = 1e-9, name: str = "feature") -> pd.DataFrame:
    """
    Recompute `fn` on truncated history and compare the overlap.

    A causal feature cannot tell the difference, because everything it is
    allowed to see is still there. A feature that peeks - a forward reindex, an
    unshifted resample, a bar stamped at its own open - will move.

    `cuts` are fractions of the history. Returns a frame of the worst absolute
    difference per column per cut, and raises AssertionError if any exceeds
    `tol`.
    """
    full = fn(bars)
    missing = [c for c in cols if c not in full.columns]
    if missing:
        raise KeyError(f"{name}: {missing} not produced by fn")

    rows = []
    for frac in cuts:
        cut = int(len(bars) * frac)
        if cut < 50:
            continue
        part = fn(bars.iloc[:cut])
        for c in cols:
            a = pd.to_numeric(full[c].iloc[:cut], errors="coerce")
            b = pd.to_numeric(part[c], errors="coerce")
            # a value that is NaN in one and not the other is also a leak
            both = a.notna() & b.notna()
            mismatch_na = int((a.notna() ^ b.notna()).sum())
            d = float((a[both] - b[both]).abs().max()) if both.any() else 0.0
            rows.append({"cut": cut, "column": c, "max_abs_diff": d,
                         "na_mismatch": mismatch_na})
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"{name}: history too short to truncate")
    bad = out[(out.max_abs_diff > tol) | (out.na_mismatch > 0)]
    if len(bad):
        raise AssertionError(
            f"{name} IS NOT CAUSAL - a future bar changed a past value:\n{bad.to_string(index=False)}")
    return out


def audit_bars(bars: pd.DataFrame, tf_minutes: int | None = None,
               name: str = "bars") -> dict:
    """
    What is actually wrong with this price series. Returns a dict of findings;
    nothing raises, because some of these are normal for FX (weekend gaps) and
    fatal for crypto (which trades through).
    """
    idx = pd.DatetimeIndex(bars.index)
    o, h, l, c = (pd.to_numeric(bars[k], errors="coerce") for k in ("open", "high", "low", "close"))
    v = pd.to_numeric(bars["volume"], errors="coerce") if "volume" in bars else pd.Series(np.nan, index=idx)

    d = idx.to_series().diff().dropna()
    step = pd.Timedelta(minutes=tf_minutes) if tf_minutes else (d.mode().iloc[0] if len(d) else pd.NaT)
    gaps = d[d > step] if step is not pd.NaT else d.iloc[:0]

    frozen = (h == l) & (o == c)
    ohlc_bad = (h < l) | (h < o) | (h < c) | (l > o) | (l > c)

    out = {
        "name": name,
        "bars": len(bars),
        "span": (idx.min(), idx.max()),
        "bar_step": step,
        "duplicate_timestamps": int(idx.duplicated().sum()),
        "out_of_order": int((d < pd.Timedelta(0)).sum()),
        "gaps": int(len(gaps)),
        "largest_gap": gaps.max() if len(gaps) else pd.Timedelta(0),
        "frozen_bars": int(frozen.sum()),
        "frozen_pct": float(frozen.mean() * 100),
        "impossible_ohlc": int(ohlc_bad.sum()),
        "nan_close": int(c.isna().sum()),
        "volume_present_pct": float(v.notna().mean() * 100),
        "volume_nonzero_pct": float((v > 0).mean() * 100),
        "hours_utc_seen": sorted(set(idx.hour.tolist())),
    }
    # a series that only ever starts at one hour is a daily bar pretending
    out["single_stamp_hour"] = len(out["hours_utc_seen"]) == 1
    return out


def print_audit(a: dict) -> None:
    print(f"  {a['name']:<12} {a['bars']:>7} bars  {a['span'][0]:%Y-%m-%d} .. {a['span'][1]:%Y-%m-%d}"
          f"  step {a['bar_step']}")
    flags = []
    if a["duplicate_timestamps"]:
        flags.append(f"{a['duplicate_timestamps']} duplicate stamps")
    if a["out_of_order"]:
        flags.append(f"{a['out_of_order']} out of order")
    if a["impossible_ohlc"]:
        flags.append(f"{a['impossible_ohlc']} impossible OHLC")
    if a["nan_close"]:
        flags.append(f"{a['nan_close']} NaN closes")
    if a["frozen_pct"] > 1:
        flags.append(f"{a['frozen_pct']:.1f}% frozen bars")
    if a["volume_nonzero_pct"] < 50:
        flags.append(f"volume unusable ({a['volume_nonzero_pct']:.0f}% nonzero)")
    if a["single_stamp_hour"]:
        flags.append("every bar stamped at one hour - is this really intraday?")
    print(f"               gaps {a['gaps']} (largest {a['largest_gap']})"
          + (("  |  " + " · ".join(flags)) if flags else "  |  clean"))


def cost_curve(run: Callable[[float], pd.Series], multiples: Sequence[float] = (0.0, 0.5, 1.0, 2.0, 4.0)
               ) -> pd.DataFrame:
    """
    `run(multiple) -> per-trade pnl` at that multiple of the modelled cost.
    Returns profit factor and total at each, plus the breakeven multiple - the
    honest headline number for how much worse execution a result can absorb.
    """
    rows = []
    for m in multiples:
        p = pd.to_numeric(run(m), errors="coerce").dropna()
        w = p > 0
        neg = p[~w].sum()
        rows.append({"cost_multiple": m, "trades": len(p),
                     "pf": float(p[w].sum() / -neg) if neg < 0 else np.nan,
                     "total": float(p.sum()), "per_trade": float(p.mean()) if len(p) else np.nan})
    d = pd.DataFrame(rows)
    ok = d[d.pf >= 1.0]
    d.attrs["breakeven_multiple"] = float(ok.cost_multiple.max()) if len(ok) else 0.0
    return d
