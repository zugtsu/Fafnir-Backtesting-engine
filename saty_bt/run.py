"""
Run a strategy across assets and write results.

    python -m saty_bt.run --strategy golden_gate --tf 30m
    python -m saty_bt.run --tf 30m --assets SPX EURUSD --equity 200 --sizing fixed_margin --margin 200 --leverage 5
    python -m saty_bt.run --tf 30m --gates-off volume oscillator      # ablation
    python -m saty_bt.run --tf 30m --set strategies.golden_gate.stop=pdc

Outputs results/<run_id>/: trades.csv, equity_<asset>.parquet, summary.json,
level_reach_events.csv, level_reach.json, and a line in results/index.md.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import asset_config, config_hash, load_config, resolve_path, seed_everything
from .engine import run_backtest
from .indicators import build_features
from .reports.level_reach import base_rates, breakdown, level_reach_events
from .reports.metrics import summarize
from .strategies import STRATEGIES


def _set_path(cfg: dict, dotted: str, value: str) -> None:
    keys = dotted.split(".")
    d = cfg
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    v: object = value
    for cast in (int, float):
        try:
            v = cast(value)
            break
        except ValueError:
            pass
    if isinstance(v, str) and v.lower() in ("true", "false"):
        v = v.lower() == "true"
    d[keys[-1]] = v


def run_one(cfg: dict, strategy: str, asset: str, tf: str, risk_over: dict,
            starting_equity: float | None, gates_off: list[str], verbose: bool = True) -> dict:
    acfg = asset_config(cfg, asset)
    scfg = copy.deepcopy(cfg["strategies"][strategy])
    for g in gates_off:
        scfg["gates"][g] = False
    rcfg = {**acfg.get("risk", {}), **{k: v for k, v in risk_over.items() if k in ("sizing", "margin_usd", "leverage", "risk_pct")}}

    f = build_features(cfg, asset, tf)
    sig = STRATEGIES[strategy](f, scfg, acfg, cfg)
    trades, eq = run_backtest(f, sig, acfg, cfg, scfg, rcfg, risk_over, starting_equity)
    m = summarize(trades, eq)
    ev = level_reach_events(f)
    br = base_rates(ev)
    setups = int(sig["long_setup"].sum() + sig["short_setup"].sum())
    entries = int(sig["long_entry"].sum() + sig["short_entry"].sum())
    blocks = pd.concat([sig.loc[sig["long_setup"], "long_block"], sig.loc[sig["short_setup"], "short_block"]])
    m.update({"asset": asset, "tf": tf, "strategy": strategy, "setups": setups, "signals": entries,
              "block_reasons": blocks[blocks != ""].value_counts().to_dict(),
              "level_reach": br, "sizing": rcfg, "volume_gate": acfg.get("volume_gate"),
              "volume_kind": acfg["volume"].get("kind"), "level_source": f["level_source"].iloc[0],
              "bars": int(len(f)), "first_bar": str(f.index[0]), "last_bar": str(f.index[-1])})
    if verbose:
        print(f"  {asset:>7} {tf}: trades={m['trades']:4d} win={m.get('win_rate', float('nan')):5.1f}% "
              f"PF={m.get('profit_factor', float('nan')):5.2f} exp={m.get('expectancy_r', float('nan')):+.2f}R "
              f"pnl=${m.get('net_pnl_usd', 0):+9.2f} dd={m.get('max_drawdown_pct', float('nan')):5.1f}% "
              f"| P(61.8|38.2)={br['p_618_given_382']:.1f}% n={br['gate_closes']}")
    return {"metrics": m, "trades": trades, "equity": eq, "events": ev, "signals": sig, "features": f}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--strategy", default="golden_gate")
    ap.add_argument("--tf", default="30m")
    ap.add_argument("--assets", nargs="*", default=None)
    ap.add_argument("--equity", type=float, default=None, help="starting equity override")
    ap.add_argument("--sizing", choices=["fixed_fractional", "fixed_margin"], default=None)
    ap.add_argument("--margin", type=float, default=None)
    ap.add_argument("--leverage", type=float, default=None)
    ap.add_argument("--risk-pct", type=float, default=None)
    ap.add_argument("--gates-off", nargs="*", default=[])
    ap.add_argument("--set", action="append", default=[], help="dotted.config.key=value")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    for s in args.set:
        k, v = s.split("=", 1)
        _set_path(cfg, k, v)
    seed_everything(cfg)
    chash = config_hash(cfg)
    risk_over = {k: v for k, v in {"sizing": args.sizing, "margin_usd": args.margin,
                                   "leverage": args.leverage, "risk_pct": args.risk_pct}.items() if v is not None}
    assets = args.assets or list(cfg["assets"])
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + f"_{args.strategy}_{args.tf}" + (f"_{args.tag}" if args.tag else "")
    out_dir = resolve_path(cfg, "results") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id}  config_hash={chash}  gates_off={args.gates_off}  risk_over={risk_over}")

    all_trades, all_events, summary = [], [], {}
    for asset in assets:
        try:
            r = run_one(cfg, args.strategy, asset, args.tf, risk_over, args.equity, args.gates_off)
        except Exception as e:  # noqa: BLE001
            print(f"  {asset:>7}: FAILED {e}")
            summary[asset] = {"error": str(e)}
            continue
        summary[asset] = r["metrics"]
        if len(r["trades"]):
            all_trades.append(r["trades"])
        ev = r["events"]
        ev["asset"] = asset
        all_events.append(ev)
        r["equity"].to_frame().to_parquet(out_dir / f"equity_{asset}.parquet")
        for by in ("session_hour", "ribbon_state", "htf_ribbon_state", "osc_zone", "stack_winner",
                   "rel_vol_bucket", "range_util_bucket", "direction"):
            t = breakdown(ev, by)
            if len(t):
                summary[asset].setdefault("level_reach_by", {})[by] = t.reset_index().to_dict(orient="records")

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    trades.to_csv(out_dir / "trades.csv", index=False)
    pd.concat(all_events, ignore_index=True).to_csv(out_dir / "level_reach_events.csv", index=False)
    meta = {"run_id": run_id, "config_hash": chash, "strategy": args.strategy, "tf": args.tf,
            "gates_off": args.gates_off, "risk_override": risk_over, "starting_equity": args.equity,
            "set": args.set, "generated_at": datetime.now(timezone.utc).isoformat(),
            "assets": summary}
    with open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o))

    idx_path = resolve_path(cfg, "results") / "index.md"
    if not idx_path.exists():
        idx_path.write_text("# Runs\n\n| run | config | strategy | tf | assets | trades | net P&L | avg win% | gates off |\n|---|---|---|---|---|---|---|---|---|\n", encoding="utf-8")
    ok = [m for m in summary.values() if "error" not in m]
    n_tr = sum(m["trades"] for m in ok)
    pnl = sum(m.get("net_pnl_usd", 0) for m in ok)
    wr = np.nanmean([m.get("win_rate", np.nan) for m in ok]) if ok else np.nan
    with open(idx_path, "a", encoding="utf-8") as fh:
        fh.write(f"| {run_id} | {chash} | {args.strategy} | {args.tf} | {len(ok)} | {n_tr} | {pnl:+.2f} | {wr:.1f} | {' '.join(args.gates_off) or '-'} |\n")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
