"""
Optimise Strategy A honestly.

    python -m saty_bt.optimize --tf 30m --assets SPX EURUSD USDJPY GBPUSD AUDUSD BTCUSD

Stage 1  structure grid: entry variant x stop type x target x partial x
         three gates on/off. Every config over the full history.
Stage 2  walk-forward on that grid: 12 months in-sample picks the config,
         the next 3 months are scored out-of-sample, rolling.
Stage 3  numeric sweep (trigger, gate, target, ATR length) around each
         asset's best structure, with overfit spike flags, and a second
         walk-forward over that grid.

Outputs results/optimize_<ts>/ with grid1.csv, grid3.csv, wf1_<asset>.csv,
wf3_<asset>.csv, oos1_trades.csv, oos3_trades.csv, summary.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import config_hash, load_config, resolve_path, seed_everything
from .sweep import flag_spikes, run_grid, score, small_account, walk_forward

SPACE1 = {
    "entry": ["gate", "trigger"],
    "entry_fill": ["market", "limit"],
    "stop": ["trigger", "trigger_touch", "swing", "pdc", "atr"],
    "target_runner": [(0.618, 1.0), (1.0, 1.618)],
    "scale_out_pct": [0, 50],
    "gate:ribbon": [True, False],
    "gate:time_warp": [True, False],
    "gate:time_of_day": [True, False],
}
# RTH assets also sweep the first-N-hours window (only meaningful with the gate on)
SPACE1_RTH_EXTRA = {"tod_hours": [2, 3, 4]}


def space_for(cfg: dict, asset: str) -> dict:
    from .config import asset_config
    s = dict(SPACE1)
    if asset_config(cfg, asset)["session"]["kind"] == "rth_us":
        s.update(SPACE1_RTH_EXTRA)
    return s
SPACE3_NUMERIC = {
    "trigger_frac": [0.20, 0.236, 0.27, 0.30],
    "gate_frac": [0.35, 0.382, 0.42, 0.45],
    "target_frac": [0.5, 0.618, 0.7, 0.786],
    "atr_length": [10, 14, 17, 20],
}


def _metrics(tr: pd.DataFrame) -> dict:
    if tr is None or tr.empty:
        return {"trades": 0}
    pnl, r = tr["pnl_usd"], tr["r_multiple"]
    gl = pnl[pnl <= 0].sum()
    eq = small_account(tr)
    eqf = small_account(tr, mode="fixed_fractional", risk_pct=1.0, leverage=5.0)
    return {"trades": int(len(tr)), "win_rate": float((pnl > 0).mean() * 100), "exp_r": float(r.mean()),
            "pf": float(pnl[pnl > 0].sum() / -gl) if gl < 0 else float("inf"),
            "pnl_1k_notional": float(pnl.sum()), "score": score(r),
            "account200_end": float(eq.iloc[-1]) if len(eq) else 200.0,
            "account200_max_dd_pct": float(((eq.cummax() - eq) / eq.cummax()).max() * 100) if len(eq) else 0.0,
            "account200_ff_end": float(eqf.iloc[-1]) if len(eqf) else 200.0,
            "account200_ff_max_dd_pct": float(((eqf.cummax() - eqf) / eqf.cummax()).max() * 100) if len(eqf) else 0.0,
            "limit_fill_share_pct": float((tr["entry_fill"] == "limit").mean() * 100) if "entry_fill" in tr else 0.0,
            "first": str(tr["exit_time"].min())[:10], "last": str(tr["exit_time"].max())[:10]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--strategy", default="golden_gate")
    ap.add_argument("--tf", default="30m")
    ap.add_argument("--assets", nargs="*", default=["SPX", "EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "BTCUSD"])
    ap.add_argument("--is-months", type=int, default=None)
    ap.add_argument("--oos-months", type=int, default=None)
    ap.add_argument("--skip-stage3", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    seed_everything(cfg)
    wf = cfg.get("walk_forward", {})
    is_m = args.is_months or int(wf.get("in_sample_months", 12))
    oos_m = args.oos_months or int(wf.get("out_of_sample_months", 3))
    out = resolve_path(cfg, "results") / f"optimize_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{args.strategy}_{args.tf}"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"config_hash": config_hash(cfg), "tf": args.tf, "strategy": args.strategy,
               "is_months": is_m, "oos_months": oos_m, "space1": {k: [str(x) for x in v] for k, v in SPACE1.items()},
               "space3": SPACE3_NUMERIC, "assets": {}}

    print(f"stage 1: structure grid ({np.prod([len(v) for v in SPACE1.values()])} configs per asset, x3 hours on RTH assets)")
    parts_t, parts_m = [], []
    for a in args.assets:
        t_, m_ = run_grid(cfg, args.strategy, [a], args.tf, space_for(cfg, a))
        parts_t.append(t_)
        parts_m.append(m_)
    tr1 = pd.concat([p for p in parts_t if len(p)], ignore_index=True)
    m1 = pd.concat(parts_m, ignore_index=True)
    if "tod_hours" not in m1:
        m1["tod_hours"] = np.nan
    m1["target_runner"] = m1["target_runner"].astype(str)
    m1.to_csv(out / "grid1.csv", index=False)
    tr1.to_csv(out / "grid1_trades.csv", index=False)

    oos1_all = []
    for a in args.assets:
        ma = m1[m1["asset"] == a]
        best = ma.sort_values("score", ascending=False).iloc[0]
        base_mask = ((ma["entry"] == "gate") & (ma["entry_fill"] == "market") & (ma["stop"] == "trigger")
                     & (ma["target_runner"] == "(0.618, 1.0)") & (ma["scale_out_pct"] == 50)
                     & ma["gate:ribbon"] & ma["gate:time_warp"] & ma["gate:time_of_day"])
        if ma["tod_hours"].notna().any():
            base_mask &= ma["tod_hours"] == 2
        base = ma[base_mask].iloc[0]
        oos, log = walk_forward(tr1, a, is_m, oos_m)
        log.to_csv(out / f"wf1_{a}.csv", index=False)
        if len(oos):
            oos1_all.append(oos)
        n_pos = int((ma["exp_r"] > 0).sum())
        summary["assets"][a] = {
            "baseline_saty": {k: (None if pd.isna(v) else v) for k, v in base[["trades", "win_rate", "exp_r", "pf", "pnl_1k", "score"]].to_dict().items()},
            "stage1_best": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.drop(["asset"]).to_dict().items()},
            "stage1_configs_positive_exp": n_pos, "stage1_configs": int(len(ma)),
            "stage1_in_sample_best": _metrics(tr1[(tr1["asset"] == a) & (tr1["cfg_id"] == best["cfg_id"])]),
            "wf1_oos": _metrics(oos), "wf1_windows": int(len(log)),
            "wf1_cfg_changes": int(log["cfg_id"].nunique()) if len(log) else 0,
        }
        b = summary["assets"][a]
        print(f"  {a:>7}: baseline exp={base['exp_r']:+.2f}R n={int(base['trades'])} | IS-best exp={best['exp_r']:+.2f}R "
              f"n={int(best['trades'])} score={best['score']:.2f} ({best['entry']}/{best['entry_fill']}/{best['stop']}/tr{best['target_runner']}/so{best['scale_out_pct']}"
              f"/rib{int(best['gate:ribbon'])}/tw{int(best['gate:time_warp'])}/tod{int(best['gate:time_of_day'])}h{best['tod_hours']}) "
              f"| WF OOS exp={b['wf1_oos'].get('exp_r', float('nan')):+.2f}R n={b['wf1_oos']['trades']} "
              f"pnl@1k=${b['wf1_oos'].get('pnl_1k_notional', 0):+.0f} $200->${b['wf1_oos'].get('account200_end', 200):.0f} "
              f"(1%risk ${b['wf1_oos'].get('account200_ff_end', 200):.0f})")
    if oos1_all:
        pd.concat(oos1_all, ignore_index=True).to_csv(out / "oos1_trades.csv", index=False)

    if not args.skip_stage3:
        print("stage 3: numeric sweep around each asset's best structure")
        oos3_all, m3_all = [], []
        for a in args.assets:
            best = summary["assets"][a]["stage1_best"]
            fixed = {"entry": best["entry"], "entry_fill": best["entry_fill"], "stop": best["stop"],
                     "scale_out_pct": int(best["scale_out_pct"]),
                     "gate:ribbon": bool(best["gate:ribbon"]), "gate:time_warp": bool(best["gate:time_warp"]),
                     "gate:time_of_day": bool(best["gate:time_of_day"])}
            if best.get("tod_hours") is not None and not pd.isna(best.get("tod_hours")):
                fixed["tod_hours"] = float(best["tod_hours"])
            runner = 1.0 if best["target_runner"].startswith("(0.618") else 1.618
            space = {k: [v] for k, v in fixed.items()}
            space.update(SPACE3_NUMERIC)
            space["runner_frac"] = [runner]
            tr3, m3 = run_grid(cfg, args.strategy, [a], args.tf, space, verbose=False)
            m3 = flag_spikes(m3, list(SPACE3_NUMERIC))
            m3_all.append(m3)
            best3 = m3.sort_values("score", ascending=False).iloc[0]
            oos, log = walk_forward(tr3, a, is_m, oos_m)
            log.to_csv(out / f"wf3_{a}.csv", index=False)
            if len(oos):
                oos3_all.append(oos)
            # plateau check: mean score of the 3x3x3x3 neighbourhood vs the peak
            summary["assets"][a].update({
                "stage3_best": {k: (v.item() if hasattr(v, "item") else v) for k, v in best3.drop(["asset"]).to_dict().items()},
                "stage3_in_sample_best": _metrics(tr3[tr3["cfg_id"] == best3["cfg_id"]]),
                "stage3_spikes": int(m3["spike"].sum()), "stage3_best_is_spike": bool(best3["spike"]),
                "stage3_median_score": float(m3["score"].median()),
                "wf3_oos": _metrics(oos), "wf3_windows": int(len(log)),
            })
            b = summary["assets"][a]
            print(f"  {a:>7}: IS-best exp={best3['exp_r']:+.2f}R n={int(best3['trades'])} score={best3['score']:.2f} "
                  f"(trig {best3['trigger_frac']} gate {best3['gate_frac']} tgt {best3['target_frac']} atr {int(best3['atr_length'])}) "
                  f"spike={bool(best3['spike'])} | WF OOS exp={b['wf3_oos'].get('exp_r', float('nan')):+.2f}R n={b['wf3_oos']['trades']} "
                  f"$200->${b['wf3_oos'].get('account200_end', 200):.0f} (1%risk ${b['wf3_oos'].get('account200_ff_end', 200):.0f}) "
                  f"limit fills {b['wf3_oos'].get('limit_fill_share_pct', 0):.0f}%")
        pd.concat(m3_all, ignore_index=True).to_csv(out / "grid3.csv", index=False)
        if oos3_all:
            pd.concat(oos3_all, ignore_index=True).to_csv(out / "oos3_trades.csv", index=False)

    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=lambda o: None if (isinstance(o, float) and np.isnan(o)) else str(o))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
