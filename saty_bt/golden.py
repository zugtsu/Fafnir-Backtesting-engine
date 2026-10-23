"""
Golden masters: the exact trade lists of every strategy we currently believe.

Why: this engine has been edited repeatedly with nothing to catch a regression.
The Heikin Ashi swap silently missed five systems and was caught only because
someone noticed a suspicious zero. Every spec below is frozen to a fixture in
tests/golden/, and tests/test_golden.py fails - naming the first trade that
moved - if a code change alters a single fill.

TWO KINDS OF FAILURE, deliberately kept apart:

  data changed   the bars underneath moved (new history imported, store
                 rebuilt). Not a bug. Each fixture records a fingerprint of the
                 data it was captured on, and the test says "data changed"
                 rather than showing a confusing trade diff.
  code changed   same data, different trades. That is a regression - or an
                 intended strategy change, in which case recapture on purpose:

                     python scripts/capture_golden.py <name> --why "..."

The specs restate their constants instead of importing them from scripts/, so
editing a research script can never silently redefine what a fixture means.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .config import load_config
from .data.store import load_bars
from .standalone import load_bitstamp_long, rubber_band, storm_chaser
from .sweep import run_grid

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "golden"

# OANDA retail spreads used for the FX research (fx_hunt.py, 2026-09-17)
OANDA = {"USDJPY": 0.014, "AUDUSD": 0.00014, "EURUSD": 0.00010,
         "GBPUSD": 0.00014, "USDCAD": 0.00020, "USDCHF": 0.00018}
FINANCING = 0.14 / 24
SIX = ["USDJPY", "AUDUSD", "EURUSD", "GBPUSD", "USDCAD", "USDCHF"]
SLOW_TREND = {"name": "50/200 sma stop3 tgt2", "system": "ma_cross", "n_fast": 50,
              "n_slow": 200, "ma_kind": "sma", "stop_atr_k": 3.0, "target_r": 2.0}


@dataclass(frozen=True)
class Spec:
    name: str
    what: str
    inputs: tuple[tuple[str, str], ...]        # (asset, tf) pairs the data fingerprint covers
    run: Callable[[], pd.DataFrame]


def _engine(strategy: str, assets: list[str], tf: str, space: dict, cfg: dict) -> pd.DataFrame:
    tr, _ = run_grid(cfg, strategy, assets, tf, space, verbose=False)
    keep = ["asset", "direction", "entry_time", "exit_time", "entry_price",
            "exit_price", "exit_reason", "r_multiple", "pnl_usd"]
    tr = tr[keep].copy()
    return tr.sort_values(["entry_time", "asset"]).reset_index(drop=True)


def _o2() -> pd.DataFrame:
    cfg = load_config()
    cfg["risk"]["max_trades_per_asset_per_day"] = 1
    return _engine("usdjpy_london_orb", ["USDJPY"], "30m", {"flat_hour_utc": [21]}, cfg)


def _slow_trend() -> pd.DataFrame:
    cfg = load_config()
    for k, sp in OANDA.items():
        c = cfg["assets"][k]["costs"]
        c["spread"], c["commission_bps"] = sp, 0.0
        c["holding_bps_per_hour"] = FINANCING
    return _engine("indicator_lab", SIX, "4h", {"spec": [SLOW_TREND]}, cfg)


def _h3() -> pd.DataFrame:
    return _engine("spx_h2_final", ["SPX"], "1h", {}, load_config())


def _rubber_band() -> pd.DataFrame:
    from .indicators import build_features
    f = build_features(load_config(), "AUDUSD", "4h")
    t = rubber_band(f[f.index >= "2021-01-01"])
    return t.rename(columns={"t": "exit_time"}).reset_index(drop=True)


def _storm() -> pd.DataFrame:
    t = storm_chaser(load_bitstamp_long(ROOT))
    return t.rename(columns={"t": "exit_time"}).reset_index(drop=True)


SPECS: dict[str, Spec] = {s.name: s for s in (
    Spec("o2_london_orb", "USDJPY 30m London box break, 1 trade/day, flat 21:00 UTC",
         (("USDJPY", "30m"),), _o2),
    Spec("slow_trend_50_200", "50/200 SMA cross, 4h, six majors, OANDA costs",
         tuple((a, "4h") for a in SIX), _slow_trend),
    Spec("h3_spx", "spx_h2_final on SPX 1h", (("SPX", "1h"),), _h3),
    Spec("rubber_band", "AUDUSD 4h +/-2 SD fade, exit at mean, 30-bar cap, from 2021",
         (("AUDUSD", "4h"),), _rubber_band),
    Spec("storm_chaser", "BTC 1h vol-gated 50-bar breakout, Bitstamp 2013-2026",
         (("bitstamp_long", "1h"),), _storm),
)}


def fingerprint(spec: Spec) -> dict:
    """What the data looked like: span, bar count and a hash of the closes."""
    out = {}
    for asset, tf in spec.inputs:
        if asset == "bitstamp_long":
            b = load_bitstamp_long(ROOT)
        else:
            b = load_bars(load_config(), asset, tf)
        closes = pd.to_numeric(b["close"], errors="coerce").to_numpy(float)
        h = hashlib.sha256(np.round(closes, 8).tobytes()).hexdigest()[:16]
        out[f"{asset}/{tf}"] = {"first": str(b.index.min()), "last": str(b.index.max()),
                                "bars": int(len(b)), "close_sha": h}
    return out


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip() or "?"
    except Exception:
        return "?"


def capture(name: str, why: str) -> dict:
    spec = SPECS[name]
    FIXTURES.mkdir(parents=True, exist_ok=True)
    tr = spec.run()
    tr.to_csv(FIXTURES / f"{name}.csv", index=False)
    meta = {"name": name, "what": spec.what, "why": why,
            "captured_at": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
            "git_commit": _git_commit(), "trades": int(len(tr)),
            "data": fingerprint(spec)}
    (FIXTURES / f"{name}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def compare(name: str, atol: float = 1e-9) -> tuple[str, str]:
    """-> (status, message); status is 'ok', 'data_changed' or 'code_changed'"""
    spec = SPECS[name]
    meta = json.loads((FIXTURES / f"{name}.meta.json").read_text(encoding="utf-8"))
    now = fingerprint(spec)
    if now != meta["data"]:
        moved = [k for k in now if now.get(k) != meta["data"].get(k)]
        return "data_changed", (f"{name}: the DATA moved, not the code ({', '.join(moved)}). "
                                f"Was {meta['data']}, now {now}. If intended, recapture: "
                                f"python scripts/capture_golden.py {name} --why \"...\"")

    want = pd.read_csv(FIXTURES / f"{name}.csv")
    got = spec.run()
    got = pd.read_csv(__import__("io").StringIO(got.to_csv(index=False)))   # same dtypes as a reload
    if list(got.columns) != list(want.columns):
        return "code_changed", f"{name}: columns changed {list(want.columns)} -> {list(got.columns)}"
    if len(got) != len(want):
        n = min(len(got), len(want))
        first = _first_diff(want.iloc[:n], got.iloc[:n], atol)
        return "code_changed", (f"{name}: trade count {len(want)} -> {len(got)}"
                                + (f"; first difference at row {first}" if first is not None else
                                   "; the first rows all match, so trades were added or removed at the end"))
    first = _first_diff(want, got, atol)
    if first is not None:
        return "code_changed", (f"{name}: trade {first} moved\n  was: {want.iloc[first].to_dict()}\n"
                                f"  now: {got.iloc[first].to_dict()}")
    return "ok", f"{name}: {len(got)} trades identical"


def _first_diff(a: pd.DataFrame, b: pd.DataFrame, atol: float) -> int | None:
    """Earliest row that differs in ANY column - the trade where the paths split."""
    bad = np.zeros(len(a), dtype=bool)
    for c in a.columns:
        x, y = a[c], b[c]
        if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
            bad |= ~np.isclose(x.to_numpy(float), y.to_numpy(float), rtol=0, atol=atol, equal_nan=True)
        else:
            bad |= (x.astype(str) != y.astype(str)).to_numpy()
    return int(np.argmax(bad)) if bad.any() else None
