"""
Canonical bar frame and integrity audit.

Canonical form (everything downstream assumes it):
    index : DatetimeIndex named 'timestamp', tz=UTC, sorted, unique,
            stamped at the bar OPEN
    columns: open, high, low, close (float64), volume (float64, NaN when the
            source carries no volume)

Nothing here repairs data. `audit()` reports what is wrong so the caller can
exclude bars instead of guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]
COLUMNS = OHLC + ["volume"]

_ALIASES = {
    "ts": "timestamp", "time": "timestamp", "date": "timestamp",
    "datetime": "timestamp", "gmt time": "timestamp", "Gmt time": "timestamp",
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Volume": "volume", "vol": "volume", "Adj Close": "adj_close",
}


class SchemaError(ValueError):
    pass


def normalize(df: pd.DataFrame, *, tz_naive_as: str | None = None) -> pd.DataFrame:
    """
    Coerce a loosely-shaped OHLCV frame into canonical form.

    tz_naive_as: timezone to assume for naive timestamps. None means naive
    timestamps are an error - we never guess UTC silently.
    """
    df = df.copy()
    df = df.rename(columns={c: _ALIASES.get(c, c) for c in df.columns})
    df.columns = [c.lower() for c in df.columns]

    if "timestamp" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "timestamp"})
        else:
            raise SchemaError("no 'timestamp' column or DatetimeIndex")

    missing = [c for c in OHLC if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = np.nan

    ts = pd.to_datetime(df["timestamp"], utc=False)
    if ts.dt.tz is None:
        if tz_naive_as is None:
            raise SchemaError("naive timestamps; pass tz_naive_as=...")
        ts = ts.dt.tz_localize(tz_naive_as, ambiguous="NaT", nonexistent="NaT")
    ts = ts.dt.tz_convert("UTC")
    df["timestamp"] = ts

    df = df[df["timestamp"].notna()]
    out = df[["timestamp"] + COLUMNS].set_index("timestamp").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.astype("float64")
    out.index.name = "timestamp"
    return out


def validate(df: pd.DataFrame) -> None:
    """Raise SchemaError unless `df` is canonical."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise SchemaError("index is not a DatetimeIndex")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise SchemaError(f"index tz must be UTC, got {df.index.tz}")
    if df.index.name != "timestamp":
        raise SchemaError("index must be named 'timestamp'")
    if not df.index.is_monotonic_increasing:
        raise SchemaError("index not sorted")
    if df.index.has_duplicates:
        raise SchemaError("duplicate timestamps")
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")


@dataclass
class IntegrityReport:
    label: str
    bars: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    null_bars: int = 0
    inverted: int = 0             # high < low, or O/C outside H/L
    non_positive: int = 0
    zero_volume_bars: int = 0
    nan_volume_bars: int = 0
    gaps: list = field(default_factory=list)   # (after_ts, before_ts, missing_bars)
    bad_index: list = field(default_factory=list)

    @property
    def excluded(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.bad_index, tz="UTC")

    def to_dict(self) -> dict:
        return {
            "label": self.label, "bars": self.bars,
            "first": str(self.first), "last": str(self.last),
            "null_bars": self.null_bars, "inverted": self.inverted,
            "non_positive": self.non_positive,
            "zero_volume_bars": self.zero_volume_bars,
            "nan_volume_bars": self.nan_volume_bars,
            "gaps": len(self.gaps),
            "largest_gap": str(max((g[1] - g[0] for g in self.gaps), default=pd.Timedelta(0))),
        }

    def summary(self) -> str:
        d = self.to_dict()
        return (f"{self.label:>14}: {d['bars']:>8,} bars  {d['first']} -> {d['last']}  "
                f"nulls={d['null_bars']} inverted={d['inverted']} badpx={d['non_positive']} "
                f"gaps={d['gaps']} (max {d['largest_gap']}) zerovol={d['zero_volume_bars']}")


def audit(df: pd.DataFrame, label: str, step: pd.Timedelta,
          max_gap: pd.Timedelta | None = None) -> IntegrityReport:
    """
    Inspect a canonical frame. `step` is the nominal bar spacing; gaps larger
    than `max_gap` (default 2 steps) are recorded - session closes included,
    so the caller decides which gaps are expected.
    """
    max_gap = max_gap or 2 * step
    rep = IntegrityReport(label=label, bars=len(df),
                          first=df.index[0] if len(df) else None,
                          last=df.index[-1] if len(df) else None)
    if df.empty:
        return rep

    nulls = df[OHLC].isna().any(axis=1)
    valid = df[~nulls]
    bad_px = (valid[OHLC] <= 0).any(axis=1)
    inv = ((valid["high"] < valid["low"])
           | (valid["open"] > valid["high"]) | (valid["open"] < valid["low"])
           | (valid["close"] > valid["high"]) | (valid["close"] < valid["low"]))

    rep.null_bars = int(nulls.sum())
    rep.non_positive = int(bad_px.sum())
    rep.inverted = int(inv.sum())
    rep.zero_volume_bars = int((df["volume"] == 0).sum())
    rep.nan_volume_bars = int(df["volume"].isna().sum())
    rep.bad_index = list(df.index[nulls]) + list(valid.index[bad_px | inv])

    deltas = df.index.to_series().diff()
    for ts, d in deltas[deltas > max_gap].items():
        rep.gaps.append((ts - d, ts, int(d / step) - 1))
    return rep


def infer_step(df: pd.DataFrame) -> pd.Timedelta:
    """Most common spacing between consecutive bars."""
    if len(df) < 3:
        raise SchemaError("too few bars to infer step")
    d = df.index.to_series().diff().dropna()
    return pd.Timedelta(d.mode().iloc[0])
