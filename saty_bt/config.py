"""
Config loading, per-asset merging, hashing and seeding.

`config.yaml` is the only place a parameter may live. Code reads it through
`load_config()`; results files carry `config_hash()` so a number can always be
traced back to the exact configuration that produced it.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = str(path)
    cfg["_root"] = str(path.parent)
    return cfg


def _canonical(cfg: dict) -> str:
    public = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return json.dumps(public, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(cfg: dict) -> str:
    """sha256 of the config with private keys (paths) removed - short form."""
    return hashlib.sha256(_canonical(cfg).encode()).hexdigest()[:12]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def asset_config(cfg: dict, name: str) -> dict:
    """Asset block with `asset_defaults` filled in and its session resolved."""
    block = cfg["assets"].get(name)
    if block is None:
        block = cfg.get("auxiliary", {}).get(name)
    if block is None:
        raise KeyError(f"asset '{name}' not in config")
    merged = _deep_merge(cfg.get("asset_defaults", {}), block)
    merged["name"] = name
    sess_name = merged["session"]
    if isinstance(sess_name, str):
        merged["session_name"] = sess_name
        merged["session"] = copy.deepcopy(cfg["sessions"][sess_name])
    return merged


def resolve_path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else Path(cfg["_root"]) / p


def seed_everything(cfg: dict) -> int:
    seed = int(cfg.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    return seed


def timeframe_minutes(tf: str) -> int:
    """'30m' -> 30, '1h' -> 60, '4h' -> 240, '1d' -> 1440."""
    tf = tf.strip().lower()
    n, unit = int(tf[:-1]), tf[-1]
    return n * {"m": 1, "h": 60, "d": 1440}[unit]


def is_intraday(tf: str) -> bool:
    return timeframe_minutes(tf) < 1440


def public_config(cfg: dict) -> dict[str, Any]:
    return {k: v for k, v in cfg.items() if not k.startswith("_")}
