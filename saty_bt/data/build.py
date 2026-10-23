"""
Build the parquet store from config.yaml.

    python -m saty_bt.data.build                 # every asset
    python -m saty_bt.data.build --assets SPX EURUSD
    python -m saty_bt.data.build --no-network    # skip yfinance / ccxt sources

For each asset: fetch the base series, audit it, resample every chart
timeframe plus the level periods (1d, 1w, 1M, 3M, 1Y), attach proxy volume
where configured, and write meta.json with the source and audit results.

yfinance-sourced assets are special: 30m and 1h come from separate capped
calls, 1d from the full daily history. Higher frames derive from 1h.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import numpy as np
import pandas as pd

from ..config import (asset_config, config_hash, load_config, resolve_path,
                      timeframe_minutes)
from .calendar import SessionSpec
from .fetchers import fetch
from .resample import aggregate_period, resample_all, resample_daily, resample_intraday
from .schema import audit, infer_step
from .store import load_bars, save_bars, save_meta, store_path

LEVEL_PERIODS = ("1w", "1M", "3M", "1Y")


def _audit_dict(df: pd.DataFrame, label: str) -> dict:
    step = infer_step(df)
    return audit(df, label, step).to_dict()


def build_from_base(cfg: dict, asset: str, base: pd.DataFrame, spec: SessionSpec,
                    timeframes: list[str]) -> dict[str, pd.DataFrame]:
    frames = resample_all(base, timeframes, spec, LEVEL_PERIODS)
    return frames


def build_yfinance(cfg: dict, acfg: dict, spec: SessionSpec, timeframes: list[str]
                   ) -> tuple[dict[str, pd.DataFrame], dict]:
    """Equities: intraday from capped calls, daily from the long series."""
    symbol = acfg["source"]["symbol"]
    frames: dict[str, pd.DataFrame] = {}
    sources: dict = {}
    daily_raw, src_d = fetch(cfg, acfg["source"], "1d")
    sources["1d"] = src_d
    # daily from yfinance is already one bar per session; add day_key/bar_close
    sf_daily = daily_raw.copy()
    from .calendar import session_frame
    sf = session_frame(sf_daily.index, spec)
    sf_daily["day_key"] = sf["day_key"].to_numpy()
    sf_daily["bar_close"] = sf["day_close"].to_numpy()
    sf_daily["early_close"] = sf["early_close"].to_numpy()
    sf_daily["partial"] = False
    sf_daily = sf_daily[sf["in_session"].to_numpy()]
    frames["1d"] = sf_daily
    for p in LEVEL_PERIODS:
        frames[p] = aggregate_period(sf_daily, p)

    want_intraday = [tf for tf in timeframes if timeframe_minutes(tf) < 1440]
    if want_intraday:
        try:
            h1, src_h = fetch(cfg, acfg["source"], "1h")
            sources["1h"] = src_h
            for tf in want_intraday:
                m = timeframe_minutes(tf)
                if m >= 60 and m % 60 == 0:
                    frames[tf] = resample_intraday(h1, tf, spec, pd.Timedelta("1h"))
        except Exception as e:      # noqa: BLE001
            sources["1h_error"] = str(e)
        if "30m" in want_intraday:
            try:
                m30, src_m = fetch(cfg, acfg["source"], "30m")
                sources["30m"] = src_m
                frames["30m"] = resample_intraday(m30, "30m", spec, pd.Timedelta("30min"))
            except Exception as e:  # noqa: BLE001
                sources["30m_error"] = str(e)
    return frames, sources


def attach_proxy_volume(cfg: dict, acfg: dict, frames: dict[str, pd.DataFrame],
                        network: bool) -> dict:
    """
    SPX option (a)/(b): copy the proxy's volume onto matching bar timestamps.
    Bars without a proxy bar get NaN volume; the gate treats NaN as
    'unavailable' and the results flag it.
    """
    vcfg = acfg["volume"]
    info = {"mode": vcfg["mode"]}
    if vcfg["mode"] != "proxy":
        return info
    proxy = vcfg["proxy_symbol"]
    aux_name = {"SPY": "SPY", "ES=F": "ES", "^GSPC": "GSPC"}.get(proxy, proxy)
    info.update({"proxy_symbol": proxy, "coverage": {}})
    for tf, df in frames.items():
        if tf in LEVEL_PERIODS:
            continue
        try:
            pv = load_bars(cfg, aux_name, tf)["volume"]
        except FileNotFoundError:
            info["coverage"][tf] = "proxy series missing"
            df["volume"] = np.nan
            continue
        vol = pv.reindex(df.index)
        df["volume"] = vol.to_numpy()
        n = int(vol.notna().sum())
        info["coverage"][tf] = {"bars_with_proxy": n, "bars_total": int(len(df)),
                                "first": str(vol.dropna().index.min()) if n else None,
                                "last": str(vol.dropna().index.max()) if n else None}
    # regenerate the level periods from the daily frame so their volume matches
    for p in LEVEL_PERIODS:
        frames[p] = aggregate_period(frames["1d"], p)
    return info


def build_asset(cfg: dict, asset: str, network: bool = True, verbose: bool = True) -> dict:
    acfg = asset_config(cfg, asset)
    spec = SessionSpec.from_config(acfg["session"])
    timeframes = acfg.get("timeframes", cfg["timeframes"])
    kind = acfg["source"]["kind"]
    meta: dict = {"asset": asset, "class": acfg.get("class"), "session": acfg["session_name"],
                  "session_spec": acfg["session"], "config_hash": config_hash(cfg),
                  "volume_cfg": acfg["volume"], "volume_gate": acfg.get("volume_gate")}

    if kind in ("yfinance", "ccxt") and not network:
        raise RuntimeError(f"{asset}: network source {kind} skipped (--no-network)")

    if kind == "yfinance":
        frames, sources = build_yfinance(cfg, acfg, spec, timeframes)
        meta["source"] = sources
        meta["base_step"] = "yfinance per-interval"
    else:
        base, src = fetch(cfg, acfg["source"])
        step = infer_step(base)
        meta["source"] = src
        meta["base_step"] = str(step)
        meta["base_audit"] = _audit_dict(base, f"{asset} base")
        frames = build_from_base(cfg, asset, base, spec, timeframes)

    meta["volume"] = attach_proxy_volume(cfg, acfg, frames, network)
    if acfg["volume"]["mode"] == "disabled":
        for df in frames.values():
            df["volume"] = np.nan

    meta["frames"] = {}
    for tf, df in frames.items():
        if df.empty:
            meta["frames"][tf] = {"bars": 0}
            continue
        p = save_bars(cfg, asset, tf, df)
        a = _audit_dict(df[["open", "high", "low", "close", "volume"]], f"{asset} {tf}") \
            if len(df) > 3 else {"bars": len(df)}
        meta["frames"][tf] = {"path": str(p.relative_to(resolve_path(cfg, "store"))), **a}
        if verbose:
            print(f"  {asset:>7} {tf:>3}: {len(df):>8,} bars  {df.index[0]} -> {df.index[-1]}"
                  f"  vol={'nan' if df['volume'].isna().all() else 'ok'}")
    save_meta(cfg, asset, meta)
    return meta


def build_auxiliary(cfg: dict, name: str, verbose: bool = True) -> dict:
    """Proxy / level series: fetched like an equity, stored under its own name."""
    acfg = asset_config(cfg, name)
    spec = SessionSpec.from_config(acfg["session"])
    frames, sources = build_yfinance(cfg, acfg, spec, cfg["timeframes"])
    meta = {"asset": name, "role": "auxiliary", "source": sources, "frames": {}}
    for tf, df in frames.items():
        if df.empty:
            continue
        p = save_bars(cfg, name, tf, df)
        meta["frames"][tf] = {"path": str(p.relative_to(resolve_path(cfg, "store"))),
                              "bars": int(len(df)), "first": str(df.index[0]), "last": str(df.index[-1])}
        if verbose:
            print(f"  {name:>7} {tf:>3}: {len(df):>8,} bars  {df.index[0]} -> {df.index[-1]}")
    save_meta(cfg, name, meta)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--no-network", action="store_true")
    ap.add_argument("--skip-aux", action="store_true")
    ap.add_argument("--debug", action="store_true", help="print tracebacks on failure")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    print(f"config {cfg['_path']}  hash={config_hash(cfg)}")
    failures = []

    if not args.no_network and not args.skip_aux:
        for name in cfg.get("auxiliary", {}):
            if args.assets and name not in args.assets:
                continue
            try:
                print(f"[aux] {name}")
                build_auxiliary(cfg, name)
            except Exception as e:  # noqa: BLE001
                failures.append((name, str(e)))
                print(f"  !! {name}: {e}")

    for asset in cfg["assets"]:
        if args.assets and asset not in args.assets:
            continue
        try:
            print(f"[asset] {asset}")
            build_asset(cfg, asset, network=not args.no_network)
        except Exception as e:  # noqa: BLE001
            failures.append((asset, str(e)))
            print(f"  !! {asset}: {e}")
            if args.debug:
                traceback.print_exc()

    if failures:
        print("\nFAILED:")
        for n, e in failures:
            print(f"  {n}: {e}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
