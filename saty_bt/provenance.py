"""
Where did this number come from?

A CSV in results/ used to be an orphan: nothing recorded which code, which
settings or which data produced it, so a six-month-old result was either
trusted on faith or thrown away. save_result() writes the table AND appends one
row to results/registry.csv saying exactly what produced it.

The `dirty` column matters more than it looks. A git commit hash only
describes the code if the working tree matched it; a run made with uncommitted
edits is stamped dirty=True, and the hash is then a lower bound, not a
description.

    from saty_bt.provenance import save_result
    save_result(table, "fx_hunt_4h", cfg=cfg, seed=0, note="53 pre-registered configs")

Retrofit scripts as they are touched; there is no need to convert them all.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
REGISTRY = RESULTS / "registry.csv"
FIELDS = ["when_utc", "name", "path", "rows", "git_commit", "dirty", "config_sha",
          "seed", "command", "note"]


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def config_sha(cfg: dict | None) -> str:
    """Hash of the settings actually used - pass the cfg AFTER any in-script edits."""
    if cfg is None:
        return ""
    body = {k: v for k, v in cfg.items() if not str(k).startswith("_")}
    blob = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def stamp(cfg: dict | None = None, seed: int | None = None, note: str = "") -> dict:
    return {
        "when_utc": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD") or "?",
        "dirty": bool(_git("status", "--porcelain")),
        "config_sha": config_sha(cfg),
        "seed": "" if seed is None else seed,
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])[:300],
        "note": note,
    }


def save_result(df: pd.DataFrame, name: str, cfg: dict | None = None,
                seed: int | None = None, note: str = "") -> Path:
    """Write results/<name>.csv and log what produced it. Returns the path."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.csv"
    df.to_csv(path, index=False)
    row = {"name": name, "path": str(path.relative_to(ROOT)), "rows": len(df),
           **stamp(cfg, seed, note)}
    new = not REGISTRY.exists()
    with REGISTRY.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    return path


def history(name: str | None = None) -> pd.DataFrame:
    """Every logged run, newest last - optionally just one result's lineage."""
    if not REGISTRY.exists():
        return pd.DataFrame(columns=FIELDS)
    r = pd.read_csv(REGISTRY)
    return r[r.name == name] if name else r
