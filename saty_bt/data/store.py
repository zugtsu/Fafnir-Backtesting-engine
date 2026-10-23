"""
Parquet store: data_store/<ASSET>/<tf>.parquet plus <ASSET>/meta.json.

meta.json records where every series came from (source kind, symbol,
fetched_at, volume kind, integrity summary) so a results file can cite it.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..config import resolve_path
from .schema import validate


def store_dir(cfg: dict, asset: str) -> Path:
    return resolve_path(cfg, "store") / asset


# Windows filenames are case-insensitive, so the monthly/quarterly level
# periods ("1M", "3M") would overwrite the 1-minute/3-minute chart frames.
_FILE_NAME = {"1M": "1mo", "3M": "3mo"}


def store_path(cfg: dict, asset: str, tf: str) -> Path:
    return store_dir(cfg, asset) / f"{_FILE_NAME.get(tf, tf)}.parquet"


def save_bars(cfg: dict, asset: str, tf: str, df: pd.DataFrame) -> Path:
    validate(df[["open", "high", "low", "close", "volume"]])
    p = store_path(cfg, asset, tf)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, engine="pyarrow", index=True)
    return p


def load_bars(cfg: dict, asset: str, tf: str) -> pd.DataFrame:
    p = store_path(cfg, asset, tf)
    if not p.exists():
        raise FileNotFoundError(f"{p} - run `python -m saty_bt.data.build` first")
    df = pd.read_parquet(p, engine="pyarrow")
    df.index = pd.DatetimeIndex(df.index, tz="UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    df.index.name = "timestamp"
    return df


def load_csv_or_parquet(path: str | Path, tz_naive_as: str | None = None) -> pd.DataFrame:
    """User-supplied file with columns timestamp,open,high,low,close,volume."""
    from .schema import normalize
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    else:
        raw = pd.read_csv(path)
    return normalize(raw, tz_naive_as=tz_naive_as)


def save_meta(cfg: dict, asset: str, meta: dict) -> Path:
    p = store_dir(cfg, asset) / "meta.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["written_at"] = datetime.now(timezone.utc).isoformat()
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return p


def load_meta(cfg: dict, asset: str) -> dict:
    p = store_dir(cfg, asset) / "meta.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def list_assets(cfg: dict) -> list[str]:
    root = resolve_path(cfg, "store")
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "meta.json").exists())
