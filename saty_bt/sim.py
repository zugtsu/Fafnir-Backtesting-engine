"""
Bar-by-bar position simulator shared by every tester of the Kaufman round
(EDGE_PREREG_KAUFMAN.md), plus the matched-exposure null.

One piece of fill arithmetic, written and tested once, so the FX, BTC and
stock results differ only in their signals and their costs.

Semantics of simulate():
  * `desired` (+1 / 0 / -1) is known at each bar's CLOSE and acted on at the
    NEXT bar's OPEN. Nothing is ever acted on at the bar that produced it.
  * A change of desired position exits at that open and, if the new value is
    non-zero, enters the other way at the SAME open (exact stop-and-reverse).
    long_only maps -1 to 0 (go flat instead of short).
  * Disaster stop: entry -/+ stop_atr x ATR of the SIGNAL bar (the bar whose
    close was acted on). Checked from the entry bar's open onward: on later
    bars a gap through the stop fills at the OPEN ("gap"); otherwise a high/
    low through it fills AT the stop ("stop"). On the entry bar only the
    intrabar check applies (the open is the fill). 1R = that distance.
  * After a stop (or gap) the simulator stays flat until the desired
    position CHANGES from the stopped direction - no re-entry on the same
    state. A desired flip straight to the other side enters at once.
  * Entries whose signal-bar ATR is NaN or <= 0 are not taken (counted in
    trades.attrs['skipped_no_atr']).
  * An open position at the end of the data is closed at the last close
    (reason "end"), with exit costs.
  * Signal exits are stamped at the bar open where they fill. Stop exits are
    stamped at the END of the stopped bar (its `bar_close` column when the
    bars carry one, else the next bar's stamp): the stop filled somewhere
    inside that bar, and holding cost and FX rollovers are charged through
    its end.
  * Costs, as fractions of the entry notional: half the spread plus fee_side
    at each fill (spread in price units / entry price, or spread_frac),
    hold_bps_hour per hour held, and optional real FX financing
    (rates.financing_usd terms: every 17:00 New York rollover R with
    entry < R <= exit, rate differential less markup) on $10,000.

Outputs: a trades frame and a per-bar frame whose `ret` column sums exactly
to the trades' `ret` (gross marked to market at each close, costs booked on
the bar where they occur, financing on the bar containing the rollover).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
import pandas as pd

from .data.clock import true_utc
from .data.rates import rate_at, rollovers

NOTIONAL = 10_000.0


@dataclass(frozen=True)
class FXFinancing:
    """Real rollover financing for a spot FX position (OANDA-style).
    base/quote default to asset[:3] / asset[3:]. stored_stamps=True means the
    bars carry the store's stamps and are converted with clock.true_utc(asset)
    (HistData pairs: New York wall + 5h); pass False if the bars were already
    re-indexed with clock.with_true_utc."""
    asset: str
    markup_pct: float = 1.0
    base: str | None = None
    quote: str | None = None
    weekend: str = "triple_wed"
    day_count: int = 365
    stored_stamps: bool = True
    table: dict | None = field(default=None, compare=False, hash=False, repr=False)

    @property
    def ccys(self) -> tuple[str, str]:
        return (self.base or self.asset[:3]), (self.quote or self.asset[3:6])


@dataclass(frozen=True)
class Costs:
    """spread: FULL spread in price units (charged half per fill), OR
    spread_frac: full spread as a fraction of price. fee_side: fraction of
    notional per fill. hold_bps_hour: basis points per hour held.
    fx: optional FXFinancing."""
    spread: float | None = None
    spread_frac: float = 0.0
    fee_side: float = 0.0
    hold_bps_hour: float = 0.0
    fx: FXFinancing | None = None

    def __post_init__(self):
        if self.spread is not None and self.spread_frac:
            raise ValueError("give spread (price units) OR spread_frac, not both")

    def side_frac(self, entry_price: float) -> float:
        half = (self.spread / 2.0 / entry_price) if self.spread is not None else self.spread_frac / 2.0
        return half + self.fee_side

    def exit_frac(self, entry_price: float, exit_price: float) -> float:
        """Exit-side cost as a fraction of the ENTRY notional: a percentage fee
        is charged on what the position is worth at exit (found by the Kaufman
        BTC verifier 2026-10-06 - a 20 -> 162 trade's exit fee is 8x the entry's)."""
        r = exit_price / entry_price
        if self.spread is not None:
            half = self.spread / 2.0 / entry_price
        else:
            half = self.spread_frac / 2.0 * r
        return half + self.fee_side * r

    def scaled(self, m: float) -> "Costs":
        """Every cost x m: spread, fee, holding and the financing MARKUP. The
        rate differential itself is not a cost and is not scaled."""
        return replace(self, spread=None if self.spread is None else self.spread * m,
                       spread_frac=self.spread_frac * m, fee_side=self.fee_side * m,
                       hold_bps_hour=self.hold_bps_hour * m,
                       fx=None if self.fx is None else replace(self.fx, markup_pct=self.fx.markup_pct * m))


class SimResult(NamedTuple):
    trades: pd.DataFrame
    bars: pd.DataFrame


TRADE_COLS = ["entry_time", "exit_time", "direction", "entry_price", "exit_price", "reason",
              "gross", "ret", "r_multiple", "hours", "pnl_usd", "signal_time", "stop_price",
              "stop_dist", "cost", "financing", "entry_bar", "exit_bar"]


def _aligned(s: pd.Series, index: pd.DatetimeIndex, name: str) -> np.ndarray:
    if not isinstance(s, pd.Series) or not s.index.equals(index):
        raise ValueError(f"{name} must be a Series on exactly the bars' index (align it first)")
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")


def _bar_end(bars: pd.DataFrame) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(bars.index)
    if "bar_close" in bars:
        be = pd.DatetimeIndex(pd.to_datetime(bars["bar_close"], utc=True))
        if not be.isna().any():
            return be.tz_convert(idx.tz)
    if len(idx) < 2:
        return idx
    step = (idx[1:] - idx[:-1]).to_series().median()
    return idx[1:].append(pd.DatetimeIndex([idx[-1] + step]))


def simulate(bars: pd.DataFrame, desired: pd.Series, atr: pd.Series, *,
             long_only: bool = False, stop_atr: float = 5.0, costs: Costs | None = None,
             notional: float = NOTIONAL, asset: str | None = None) -> SimResult:
    """
    bars     OHLC on a tz-aware index, stamped at bar OPEN (optional
             `bar_close` column). These are the FILL bars.
    desired  +1 / 0 / -1 per bar, known at that bar's close (NaN -> 0), on
             exactly bars.index.
    atr      per bar, on exactly bars.index; the stop uses the signal bar's.
    Returns SimResult(trades, bars) - see the module docstring.
    """
    costs = costs or Costs()
    idx = pd.DatetimeIndex(bars.index)
    if idx.tz is None:
        raise ValueError("bars must have a tz-aware index")
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("bars index must be strictly increasing")
    O, H, L, C = (pd.to_numeric(bars[k], errors="coerce").to_numpy(dtype="float64")
                  for k in ("open", "high", "low", "close"))
    if not (np.isfinite(O).all() and np.isfinite(H).all() and np.isfinite(L).all() and np.isfinite(C).all()):
        raise ValueError("bars contain NaN prices")
    want = np.nan_to_num(_aligned(desired, idx, "desired"), nan=0.0)
    if not np.isin(want, (-1.0, 0.0, 1.0)).all():
        raise ValueError("desired must be in {-1, 0, +1}")
    if long_only:
        want = np.where(want < 0, 0.0, want)
    A = _aligned(atr, idx, "atr")
    N = len(idx)
    bend = _bar_end(bars)

    pos_open = np.zeros(N, dtype=np.int64)
    pos_close = np.zeros(N, dtype=np.int64)
    gross_bar = np.zeros(N)
    cost_bar = np.zeros(N)
    rows = []
    cur = None          # [dir, entry_bar, entry_price, stop, dist, signal_bar, mark]
    lock = None
    skipped = 0

    def close_trade(j: int, px: float, when: pd.Timestamp, reason: str) -> None:
        d, e_bar, pe, stop, dist, s_bar, mark = cur
        gross_bar[j] += d * (px - mark) / pe
        side = costs.exit_frac(pe, px)
        cost_bar[j] += side
        rows.append({"entry_time": idx[e_bar], "exit_time": when, "direction": d,
                     "entry_price": pe, "exit_price": px, "reason": reason,
                     "gross": d * (px - pe) / pe, "signal_time": idx[s_bar],
                     "stop_price": stop, "stop_dist": dist, "side_cost": side,
                     "entry_cost": costs.side_frac(pe),
                     "entry_bar": e_bar, "exit_bar": j})

    for j in range(N):
        if j >= 1:
            tgt = int(want[j - 1])
            if lock is not None:
                if tgt != lock:
                    lock = None
                else:
                    tgt = 0
            if cur is not None and tgt != cur[0]:
                close_trade(j, O[j], idx[j], "reverse" if tgt != 0 else "signal")
                cur = None
            if cur is None and tgt != 0:
                a = A[j - 1]
                if np.isfinite(a) and a > 0:
                    dist = stop_atr * a
                    cur = [tgt, j, O[j], O[j] - tgt * dist, dist, j - 1, O[j]]
                    cost_bar[j] += costs.side_frac(O[j])
                else:
                    skipped += 1
            elif cur is not None and cur[1] < j:
                d, stop = cur[0], cur[3]
                if (d > 0 and O[j] <= stop) or (d < 0 and O[j] >= stop):
                    close_trade(j, O[j], idx[j], "gap")
                    lock, cur = d, None
        pos_open[j] = cur[0] if cur is not None else 0
        if cur is not None:
            d, stop = cur[0], cur[3]
            if (d > 0 and L[j] <= stop) or (d < 0 and H[j] >= stop):
                close_trade(j, stop, bend[j], "stop")
                lock, cur = d, None
            else:
                gross_bar[j] += d * (C[j] - cur[6]) / cur[2]
                cur[6] = C[j]
        pos_close[j] = cur[0] if cur is not None else 0
    if cur is not None:
        cur[6] = C[N - 1]                       # already marked to the last close
        close_trade(N - 1, C[N - 1], bend[N - 1], "end")
        cur = None

    t = pd.DataFrame(rows)
    hold_bar = np.zeros(N)
    fin_bar = np.zeros(N)
    if len(t):
        t["entry_time"] = pd.DatetimeIndex(t["entry_time"])
        t["exit_time"] = pd.DatetimeIndex(t["exit_time"])
        t["hours"] = (t["exit_time"] - t["entry_time"]).dt.total_seconds() / 3600.0
        # holding, attributed bar by bar: full bars from entry to the exit bar,
        # then the part of the exit bar held (0 for an exit at its open)
        bar_h = np.zeros(N)
        if N > 1:
            bar_h[:-1] = np.asarray((idx[1:] - idx[:-1]).total_seconds(), float) / 3600.0
        cover = np.zeros(N + 1)
        eb, xb = t["entry_bar"].to_numpy(int), t["exit_bar"].to_numpy(int)
        np.add.at(cover, eb, 1.0)
        np.add.at(cover, xb, -1.0)
        held = np.cumsum(cover)[:N] > 0
        part = np.asarray((pd.DatetimeIndex(t["exit_time"]) - idx[xb]).total_seconds(), float) / 3600.0
        hours_bar = np.where(held, bar_h, 0.0)
        np.add.at(hours_bar, xb, part)
        hold_bar = hours_bar * costs.hold_bps_hour / 1e4
        t["financing"] = 0.0
        if costs.fx is not None:
            t["financing"], fin_bar = _financing(t, idx, costs.fx, N)
        t["cost"] = t["entry_cost"] + t["side_cost"] + t["hours"] * costs.hold_bps_hour / 1e4
        t["ret"] = t["gross"] - t["cost"] + t["financing"]
        t["r_multiple"] = t["ret"] / (t["stop_dist"] / t["entry_price"])
        t["pnl_usd"] = t["ret"] * notional
        t = t[TRADE_COLS]
    else:
        t = pd.DataFrame({c: pd.Series(dtype="float64") for c in TRADE_COLS})
    if asset is not None:
        t["asset"] = asset
    t.attrs["skipped_no_atr"] = skipped
    t.attrs["notional"] = notional

    b = pd.DataFrame({"pos": pos_open, "pos_close": pos_close, "gross": gross_bar,
                      "cost": cost_bar + hold_bar, "financing": fin_bar}, index=idx)
    b["ret"] = b["gross"] - b["cost"] + b["financing"]
    return SimResult(t, b)


def _financing(t: pd.DataFrame, idx: pd.DatetimeIndex, fx: FXFinancing, N: int):
    """Per-trade financing (fraction of notional) and its per-bar attribution.
    Same terms as rates.financing_usd, computed once over the whole span."""
    conv = (lambda x: true_utc(pd.DatetimeIndex(x), fx.asset)) if fx.stored_stamps \
        else (lambda x: pd.DatetimeIndex(x).tz_convert("UTC"))
    bars_u = conv(idx)
    e_u, x_u = conv(t["entry_time"]), conv(t["exit_time"])
    inst, w = rollovers(e_u.min(), x_u.max(), fx.weekend)
    per_trade = np.zeros(len(t))
    fin_bar = np.zeros(N)
    if not len(inst):
        return per_trade, fin_bar
    base, quote = fx.ccys
    diff = rate_at(base, inst, fx.table) - rate_at(quote, inst, fx.table)
    lo = inst.searchsorted(e_u, side="right")
    hi = inst.searchsorted(x_u, side="right")
    d = t["direction"].to_numpy(float)
    for i in range(len(t)):
        if hi[i] <= lo[i]:
            continue
        sl = slice(lo[i], hi[i])
        terms = w[sl] * (d[i] * diff[sl] - fx.markup_pct) / 100.0 / fx.day_count
        per_trade[i] = float(np.sum(terms))
        where = bars_u.searchsorted(inst[sl], side="left") - 1
        np.add.at(fin_bar, where, terms)
    return per_trade, fin_bar


# ── helpers for nulls and benchmarks ──────────────────────────────────

def open_to_open(bars: pd.DataFrame) -> pd.Series:
    """Return of holding from this bar's open to the next bar's open - the
    return a position in sim.bars['pos'] earns (the last bar is NaN)."""
    o = pd.to_numeric(bars["open"], errors="coerce")
    return (o.shift(-1) / o - 1.0).rename("o2o")


def series_stats(ret: pd.Series, periods_per_year: float | None = None) -> dict:
    """Compounded total return, log return, Sharpe (annualised if
    periods_per_year is given) and worst drawdown of a per-period return
    series (NaN -> 0). Buy-and-hold: series_stats(open_to_open(bars))."""
    r = pd.to_numeric(pd.Series(ret), errors="coerce").fillna(0.0).to_numpy(float)
    if not len(r):
        return {"total_return": 0.0, "log_return": 0.0, "sharpe": np.nan, "max_drawdown": 0.0, "periods": 0}
    eq = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    sd = r.std(ddof=1) if len(r) > 1 else np.nan
    sharpe = float(r.mean() / sd) if sd and np.isfinite(sd) and sd > 0 else np.nan
    if periods_per_year:
        sharpe *= np.sqrt(periods_per_year)
    return {"total_return": float(eq[-1] - 1.0), "log_return": float(np.sum(np.log1p(r))),
            "sharpe": sharpe, "max_drawdown": float(np.min(eq / peak - 1.0)), "periods": len(r)}


def _entries(x: np.ndarray) -> np.ndarray:
    """Entry indicator along the last axis: in now and not in on the previous bar."""
    e = x.copy()
    e[..., 1:] = x[..., 1:] & ~x[..., :-1]
    return e


def _score(X: np.ndarray, r: np.ndarray, cost: float) -> np.ndarray:
    """Per-bar return of in/out series X (draws x bars): in x r - cost at entries."""
    return X * r - cost * _entries(X)


def _summ(s: np.ndarray, ppy: float | None):
    lr = np.sum(np.log1p(s), axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = s.mean(axis=-1) / s.std(axis=-1, ddof=1)
    if ppy:
        sh = sh * np.sqrt(ppy)
    return lr, sh


def _block_starts(x: np.ndarray, block: int, phase: int = 0, snap: bool = False) -> np.ndarray:
    N = len(x)
    edges = list(range(phase if phase else block, N, block))
    if snap:                       # move each edge forward to the next out-of-market bar
        out = np.flatnonzero(~x)
        pos = np.searchsorted(out, edges)
        edges = sorted({int(out[p]) for p in pos if p < len(out)})
    return np.asarray([0] + [e for e in edges if 0 < e < N], dtype=np.int64)


def _shuffle_matched(x: np.ndarray, n: int, block: int, rng: np.random.Generator,
                     max_draws: int, phase: int = 0, tol: int = 0, snap: bool = False,
                     batch: int = 2000):
    """Block shuffle of bool series x: cut into consecutive `block`-bar blocks
    from bar 0 (phase > 0: the first block is `phase` bars), permute the
    block ORDER, keep a draw when its entry count is within `tol` of x's
    (rejection). The invested fraction is preserved exactly by construction.
    Entry counts are computed from block ends without building the series.
    snap=True moves every block edge to the next bar where x is out, so no
    block but the first starts in the market (and that one, if it does, stays
    first): entries are then preserved exactly and every draw is accepted.
    Returns (accepted draws x bars (at most n), draws tried, draws that
    matched, median entries of all draws, number of blocks)."""
    N = len(x)
    target = int(_entries(x).sum())
    starts = _block_starts(x, block, phase, snap)
    ends = np.r_[starts[1:], N] - 1
    lens = ends - starts + 1
    nb = len(starts)
    pin_first = bool(snap and x[0])
    first, last = x[starts].astype(np.int64), x[ends].astype(np.int64)
    is_start = np.zeros(N, bool)
    is_start[starts] = True
    trans = np.zeros(N, bool)
    trans[1:] = x[1:] & ~x[:-1]
    within = int((trans & ~is_start).sum())
    keep, got, draws, seen = [], 0, 0, []
    while got < n and draws < max_draws:
        if pin_first:
            perms = np.argsort(rng.random((batch, nb - 1)), axis=1) + 1
            perms = np.concatenate([np.zeros((batch, 1), dtype=perms.dtype), perms], axis=1)
        else:
            perms = np.argsort(rng.random((batch, nb)), axis=1)
        ent = within + first[perms[:, 0]]
        if nb > 1:
            ent = ent + np.sum(first[perms[:, 1:]] * (1 - last[perms[:, :-1]]), axis=1)
        draws += batch
        if len(seen) < 20:
            seen.append(ent)
        ok = perms[np.abs(ent - target) <= tol]
        if len(ok):
            keep.append(ok)
            got += len(ok)
        # hopeless: at this rate max_draws cannot reach even half of n
        if draws >= 100_000 and got < n and got * max_draws / draws < 0.5 * n:
            break
    med = float(np.median(np.concatenate(seen))) if seen else np.nan
    if not keep:
        return np.zeros((0, N), bool), draws, got, med, nb
    P = np.concatenate(keep)[:n]
    out = np.empty((len(P), N), bool)
    ar = np.arange(N)
    for i, p in enumerate(P):          # bar k of the draw = bar (start of its block + offset) of x
        L = lens[p]
        shift = starts[p] - np.concatenate([[0], np.cumsum(L)[:-1]])
        out[i] = x[np.repeat(shift, L) + ar]
    return out, draws, got, med, nb


def matched_exposure_null(in_market, bar_returns, cost_per_entry, n: int = 1000, block: int = 20,
                          seed: int = 0, max_draws: int = 4_000_000,
                          periods_per_year: float | None = None, phase: int = 0, tol: int = 0,
                          snap: bool = False, allow_partial: bool = False,
                          return_series: bool = False) -> dict:
    """
    Random in/out series with the same invested fraction (block shuffle:
    consecutive `block`-bar blocks from bar 0, block order permuted) and the
    same number of entries (exact by default, by rejection), scored on the
    same bar returns at the same cost. As pre-registered (and as H3 did it).

    A block shuffle cuts holding runs at block edges, so its draws usually
    have MORE entries than a persistent rule; exact matching can then accept
    very few draws. If fewer than n match within max_draws per column it
    raises RuntimeError (with the acceptance numbers) unless allow_partial,
    which warns and returns what matched. phase > 0 (first block shortened)
    and tol > 0 (entry count within tol) are diagnostics / declared
    deviations, not the pre-registered test.

    snap=True is the closest FEASIBLE variant, and a declared deviation: the
    same nominal `block`-bar edges, each moved forward to the next bar where
    the rule is out of the market (a block that starts in the market at bar
    0 stays first). No shuffle can then create or merge an entry, so every
    draw matches exactly (acceptance 1.0) and in-runs are never cut.

    in_market      bool per bar: a Series, or a DataFrame with one column per
                   market (pooled equal weight: each column is shuffled and
                   matched on its own, the portfolio bar return is the mean
                   over columns)
    bar_returns    same shape and index (NaN -> 0); for the simulator use
                   in_market = res.bars.pos > 0, bar_returns = open_to_open(bars)
    cost_per_entry ROUND-TRIP cost fraction charged on each entry bar (scalar,
                   or one per column)

    Scoring, for the rule and every null draw alike: s_t = in_t x r_t -
    cost x entry_t; log return = sum log(1 + s_t); Sharpe = mean / SD
    (ddof 1) of s_t, x sqrt(periods_per_year) when given.

    Returns a dict: log_return, sharpe (arrays, one per accepted draw),
    observed {log_return, sharpe}, p_log / p_sharpe (share of the null at or
    above the rule), q95_log / q95_sharpe, pass95 (rule >= both 95th
    percentiles), and per column: acceptance_rate (matched / tried), draws,
    invested_fraction, entries (the rule's), shuffle_entries_median (all
    block shuffles, matched or not, for context), blocks; method ("block" or
    "snap"); n (draws kept). Run it once per regime (slice the inputs).
    """
    X = in_market.to_frame() if isinstance(in_market, pd.Series) else in_market
    R = bar_returns.to_frame() if isinstance(bar_returns, pd.Series) else bar_returns
    if X.shape != R.shape or not X.index.equals(R.index):
        raise ValueError("in_market and bar_returns must share index and shape")
    m, N = X.shape[1], X.shape[0]
    cols = list(X.columns)
    if isinstance(cost_per_entry, (pd.Series, dict)):
        costs = [float(cost_per_entry[c]) for c in cols]
    elif np.ndim(cost_per_entry) == 0:
        costs = [float(cost_per_entry)] * m
    else:
        costs = [float(c) for c in cost_per_entry]
    rng = np.random.default_rng(seed)

    obs_s = np.zeros(N)
    nulls, series = [], {}
    info = {"acceptance_rate": {}, "draws": {}, "invested_fraction": {}, "entries": {},
            "shuffle_entries_median": {}, "blocks": {}}
    for j, c in enumerate(cols):
        x = X[c].fillna(False).to_numpy(bool)
        r = pd.to_numeric(R.iloc[:, j], errors="coerce").fillna(0.0).to_numpy(float)
        obs_s += _score(x, r, costs[j]) / m
        Xn, draws, matched, med, nb = _shuffle_matched(x, n, block, rng, max_draws, phase=phase, tol=tol,
                                              snap=snap)
        info["acceptance_rate"][c] = matched / draws if draws else np.nan
        info["draws"][c] = draws
        info["invested_fraction"][c] = float(x.mean()) if N else np.nan
        info["entries"][c] = int(_entries(x).sum())
        info["shuffle_entries_median"][c] = med
        info["blocks"][c] = nb
        nulls.append((Xn, r, costs[j]))
        if return_series:
            series[c] = Xn
    k = min(len(Xn) for Xn, _, _ in nulls)
    if k < n:
        msg = (f"matched_exposure_null: only {k} of {n} draws matched the entry count "
               f"(block shuffles tried {info['draws']}, cap {max_draws}) | rule entries {info['entries']}, "
               f"median shuffle entries {info['shuffle_entries_median']}, "
               f"acceptance {info['acceptance_rate']}")
        if not allow_partial:
            raise RuntimeError(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    S = np.zeros((k, N))
    for Xn, r, cst in nulls:
        S += _score(Xn[:k], r, cst) / m
    lr, sh = _summ(S, periods_per_year)
    olr, osh = _summ(obs_s[None, :], periods_per_year)
    olr, osh = float(olr[0]), float(osh[0])
    q95_l = float(np.quantile(lr, 0.95)) if k else np.nan
    q95_s = float(np.quantile(sh, 0.95)) if k else np.nan
    out = {"log_return": lr, "sharpe": sh, "observed": {"log_return": olr, "sharpe": osh},
           "p_log": float(np.mean(lr >= olr)) if k else np.nan,
           "p_sharpe": float(np.mean(sh >= osh)) if k else np.nan,
           "q95_log": q95_l, "q95_sharpe": q95_s,
           "pass95": bool(k and olr >= q95_l and osh >= q95_s), "n": k,
           "method": "snap" if snap else "block", **info}
    if m == 1:
        for key in ("acceptance_rate", "draws", "invested_fraction", "entries", "shuffle_entries_median",
                    "blocks"):
            out[key] = out[key][cols[0]]
    if return_series:
        out["series"] = series if m > 1 else series[cols[0]]
    return out
