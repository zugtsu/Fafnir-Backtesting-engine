"""
One gate every candidate has to clear before it is called a strategy.

The OpenAI prompt this came from specifies deployment discipline beautifully -
risk limits, kill switches, calibration scoring, rollback paths - and never
once requires the thing being deployed to have an edge. Every gate in it is
operational. A system built exactly to that spec will find a well-logged,
fully risk-managed way to lose money, because the base rate of a real edge is
about one percent and nothing in the pipeline measures against it.

This is the missing block, with our own numbers behind each line:

  TWO REGIMES       O2 scored 1.25 on 2021-2026 and 0.93 on 2015-2020. The
                    rubber band went 1.49 to 0.96. Half-to-half profit factor
                    correlation across 53 pre-registered configs was NEGATIVE
                    (-0.16 on 4h, -0.34 on 1h). One regime tells you nothing.

  MULTIPLE TESTING  Of 499 cells, 5 cleared both regimes where chance predicts
                    10. Unless you count what you tried, a survivor is not
                    evidence. The shuffle test is the arithmetic.

  CONCENTRATION     Our best FX result dropped from +448 to negative when the
                    top 20 of 391 trades were removed. Trend following is
                    meant to be concentrated; a result carried by five trades
                    is five trades.

  SIGNIFICANCE      That same result: t = 1.36 against a bar of 2.0, bootstrap
                    interval on profit factor [1.00, 1.47]. It passed every
                    structural test and still could not be told from luck.

  COST              1h FX: 3 of 6 families above 1.0 before cost, 0 of 6
                    after. Breakeven cost multiple is the headline, not a
                    footnote.

  HELD OUT          Markets the search never touched. Everything else can be
                    gamed by trying again; this cannot.

Nothing here is clever. It is a checklist that refuses to be talked out of a
number, which is the only thing that has worked in this project.

    from saty_bt.promote import gate, report
    r = gate(trades, split="2021-01-01", n_configs_tried=53)
    report(r)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .stats import deflated_sharpe, pbo_test, pf_interval


def pf(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    w = x > 0
    neg = x[~w].sum()
    return float(x[w].sum() / -neg) if neg < 0 else np.nan


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    hard: bool = True          # a soft check is reported but does not block


@dataclass
class Verdict:
    checks: list[Check] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.hard)

    def add(self, name: str, ok: bool, detail: str, hard: bool = True) -> None:
        self.checks.append(Check(name, bool(ok), detail, hard))


def gate(trades: pd.DataFrame, *, split, bar: float = 1.10, min_trades: int = 30,
         n_configs_tried: int | None = None, population_pf: pd.DataFrame | None = None,
         held_out: pd.DataFrame | None = None, cost_curve_df: pd.DataFrame | None = None,
         t_bar: float = 2.0, drop_best: int = 20, seed: int = 0,
         returns: pd.DataFrame | None = None, chosen: str | None = None,
         pbo_S: int = 16, pbo_reps: int = 200, n_trials_dsr: int | None = None) -> Verdict:
    """
    trades          must carry exit_time, pnl_usd, r_multiple, and asset
    split           the regime boundary
    population_pf   every cell tried, with columns pf_old/pf_new, for the
                    shuffle test. Without it the multiple-testing check can
                    only report the count, not price it.
    held_out        trades on markets that were never in the search
    cost_curve_df   output of validate.cost_curve
    returns         stats.returns_matrix over EVERY config tried (periods x
                    configs); with `chosen` (its column) it unlocks the PBO
                    and deflated-Sharpe checks
    """
    v = Verdict()
    t = trades.copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], utc=True)
    sp = pd.Timestamp(split, tz="UTC") if not isinstance(split, pd.Timestamp) else split
    old, new = t[t.exit_time < sp], t[t.exit_time >= sp]

    # Score in R wherever trades carry one: risk-based sizing is how these are
    # traded, and dollars at a fixed notional can disagree badly. Found
    # 2026-10-06 on the user's system: 2015-2020 PF 1.34 in dollars, 0.96 in R.
    unit = "r_multiple" if "r_multiple" in t and t.r_multiple.notna().any() else "pnl_usd"
    ulab = "in R" if unit == "r_multiple" else "in $"

    # 1 - two regimes, each on its own
    p_old, p_new = pf(old[unit]), pf(new[unit])
    ok = (len(old) >= min_trades and len(new) >= min_trades
          and p_old >= bar and p_new >= bar)
    v.add("two regimes", ok,
          f"{ulab}: before {sp:%Y-%m}: {len(old)} trades PF {p_old:.2f} | "
          f"after: {len(new)} trades PF {p_new:.2f} | need {bar:.2f} and "
          f"{min_trades} trades in BOTH")
    v.stats |= {"pf_old": p_old, "pf_new": p_new, "n_old": len(old), "n_new": len(new)}

    # 2 - no single market carrying the whole thing
    if "asset" in t and t.asset.nunique() > 1:
        by = t.groupby("asset").pnl_usd.sum()
        tot = by.sum()
        share = float(by.max() / tot) if tot > 0 else np.nan
        losers = int((by < 0).sum())
        v.add("not one market", tot > 0 and share <= 0.75,
              f"{t.asset.nunique()} markets, biggest contributes {share*100:.0f}% "
              f"of profit, {losers} lose money | need <= 75%")
        v.stats["market_share_max"] = share
    else:
        v.add("not one market", True,
              "single market - cross-market concentration cannot be tested; "
              "a time-concentration check is ENGINE_PLAN step 10", hard=False)

    # 3 - multiple testing
    if n_configs_tried is None:
        v.add("multiple testing", False,
              "n_configs_tried not supplied - a survivor out of an uncounted "
              "search is not evidence")
    elif population_pf is not None and {"pf_old", "pf_new"} <= set(population_pf.columns):
        e = population_pf.dropna(subset=["pf_old", "pf_new"])
        winners = int(((e.pf_old >= bar) & (e.pf_new >= bar)).sum())
        p1, p2 = float((e.pf_old >= bar).mean()), float((e.pf_new >= bar).mean())
        expect = len(e) * p1 * p2
        rng = np.random.default_rng(seed)
        sh = np.array([int(((e.pf_old.to_numpy() >= bar) &
                            (rng.permutation(e.pf_new.to_numpy()) >= bar)).sum())
                       for _ in range(2000)])
        pctile = float((sh < winners).mean() * 100)
        # When most configs pass each half, at least a+b-N must pass both by
        # counting alone, and no shuffle can land lower: the verdict is forced.
        # (Found on Storm Chaser 2026-10-05: 126 + 141 - 144 = 123 = observed.)
        forced = winners <= int((e.pf_old >= bar).sum() + (e.pf_new >= bar).sum() - len(e))
        superseded = returns is not None and chosen is not None
        why = (" | UNINFORMATIVE: this many passes forces the count by arithmetic" if forced else
               " | context only: the PBO check below measures selection directly" if superseded
               else " | need 95th+")
        v.add("multiple testing", winners > 0 and pctile >= 95,
              f"{winners} of {len(e)} cells cleared both; chance predicts "
              f"{expect:.1f}; {pctile:.0f}th percentile of a shuffle test{why}",
              hard=not (forced or superseded))
        v.stats |= {"survivors": winners, "expected_by_chance": expect, "shuffle_pctile": pctile}
    else:
        v.add("multiple testing", True,
              f"{n_configs_tried} configurations tried, counted in advance; no "
              f"population supplied so this is not priced", hard=False)

    # 4 - concentration
    s = t[unit].sort_values(ascending=False)
    kept = t[~t.index.isin(s.head(drop_best).index)]
    ok = pf(kept[unit]) >= 1.0
    v.add(f"survives losing its best {drop_best}", ok,
          f"{ulab}: PF {pf(t[unit]):.2f} -> {pf(kept[unit]):.2f} without the top "
          f"{drop_best} of {len(t)} trades | need >= 1.00")
    v.stats["pf_ex_best"] = pf(kept[unit])

    # 5 - significance
    # t is scale-free, so pnl serves when a strategy has no fixed R (pct-return systems)
    col = "r_multiple" if "r_multiple" in t and t.r_multiple.notna().any() else "pnl_usd"
    r = pd.to_numeric(t[col], errors="coerce").dropna()
    tstat = float(r.mean() / r.std() * np.sqrt(len(r))) if len(r) > 2 and r.std() else np.nan
    # BLOCK bootstrap on the time-ordered trades: winners cluster, and an iid
    # resample breaks the clusters up and reports intervals that are too tight
    ordered = t.sort_values("exit_time")[unit]          # same unit as the other checks
    bi = pf_interval(ordered, seed=seed + 1)
    ii = pf_interval(ordered, mean_block=1.0, seed=seed + 1)
    lo, hi = bi["lo"], bi["hi"]
    v.add("distinguishable from luck", np.isfinite(tstat) and tstat >= t_bar and lo > 1.0,
          f"t = {tstat:.2f} (need {t_bar:.1f}) | block-bootstrap 90% PF {ulab} [{lo:.2f}, {hi:.2f}] "
          f"(iid would say [{ii['lo']:.2f}, {ii['hi']:.2f}]) - lower bound must clear 1.00")
    v.stats |= {"t": tstat, "pf_lo": lo, "pf_hi": hi, "pf_lo_iid": ii["lo"], "pf_hi_iid": ii["hi"]}

    # 5b - overfitting of the SELECTION, if every tried config's returns are supplied
    if returns is not None and chosen is not None and chosen in returns.columns:
        pt = pbo_test(returns, S=pbo_S, reps=pbo_reps, seed=seed)
        # Two ways to pass, from Bailey et al. 2014: picking the in-sample best is
        # informative (PBO beats its zero-edge null), OR whatever wins in-sample
        # almost never loses out of sample (a broad plateau, where picking cannot
        # help because every neighbour is good). Noise false-passes the second
        # route at most 3.5% of the time even for 144 configs at r~0.9.
        v.add("selection not overfit (PBO)", pt["null_p"] <= 0.05 or pt["prob_oos_loss"] <= 0.05,
              f"PBO {pt['pbo']:.2f} (zero-edge versions score that low {pt['null_p']*100:.0f}% "
              f"of the time) | in-sample winner loses out of sample in "
              f"{pt['prob_oos_loss']*100:.0f}% of {pt['n_splits']} splits | need either <= 5%")
        srs = (returns.mean() / returns.std(ddof=1)).to_numpy()
        ds = deflated_sharpe(returns[chosen], srs, n_trials=n_trials_dsr)
        v.add("survives deflation", np.isfinite(ds["dsr"]) and ds["dsr"] >= 0.95,
              f"weekly Sharpe {ds['sharpe']:.3f} vs {ds['sr_threshold']:.3f} expected from the best "
              f"of {ds['n_trials']} noise tries -> deflated {ds['dsr']*100:.0f}% "
              f"(naive {ds['psr_vs_zero']*100:.0f}%) | need 95%")
        v.stats |= {"pbo": pt["pbo"], "pbo_null_p": pt["null_p"],
                    "prob_oos_loss": pt["prob_oos_loss"], "dsr": ds["dsr"],
                    "psr": ds["psr_vs_zero"]}
    else:
        v.add("selection not overfit (PBO)", False,
              "no returns matrix of the configs tried - selection bias is unmeasured", hard=False)

    # 6 - cost
    if cost_curve_df is not None and len(cost_curve_df):
        be = cost_curve_df.attrs.get("breakeven_multiple", np.nan)
        v.add("absorbs worse execution", np.isfinite(be) and be >= 2.0,
              f"still profitable at {be:g}x modelled cost | need 2x")
        v.stats["breakeven_cost_multiple"] = be
    else:
        v.add("absorbs worse execution", False,
              "no cost curve supplied - cost has been the binding constraint "
              "in every FX result here")

    # 7 - markets the search never saw
    if held_out is not None and len(held_out):
        hp = pf(held_out.pnl_usd)
        per = held_out.groupby("asset").pnl_usd.apply(pf) if "asset" in held_out else pd.Series(dtype=float)
        v.add("held-out markets", hp >= 1.0,
              f"{len(held_out)} trades on {held_out.asset.nunique() if 'asset' in held_out else '?'} "
              f"untouched markets, PF {hp:.2f}"
              + (" (" + ", ".join(f"{k} {x:.2f}" for k, x in per.items()) + ")" if len(per) else ""))
        v.stats["pf_held_out"] = hp
    else:
        v.add("held-out markets", False,
              "nothing was held out - every market was in the search", hard=False)
    return v


def report(v: Verdict, title: str = "PROMOTION GATE") -> bool:
    w = 92
    print("=" * w)
    print(title)
    print("=" * w)
    for c in v.checks:
        mark = "PASS" if c.passed else ("FAIL" if c.hard else "note")
        print(f"  [{mark:<4}] {c.name:<28} {c.detail}")
    hard = [c for c in v.checks if c.hard]
    n_ok = sum(c.passed for c in hard)
    print("-" * w)
    if v.passed:
        print(f"  PROMOTE - cleared all {len(hard)} hard checks.")
    else:
        failed = [c.name for c in hard if not c.passed]
        print(f"  DO NOT PROMOTE - {n_ok}/{len(hard)} hard checks. Failed: {', '.join(failed)}")
    print("=" * w)
    return v.passed
