"""
The statistics that keep a backtest honest about its own uncertainty.

  stationary_bootstrap   Resampling trades one at a time pretends they are
                         independent. They are not: a trending month hands a
                         trend system a run of winners together, and an
                         iid resample breaks that run up and so understates how
                         bad a stretch can get. The stationary bootstrap
                         (Politis & Romano 1994) resamples BLOCKS of
                         consecutive trades with random geometric lengths,
                         keeping the clustering. Its intervals are wider, and
                         wider is the honest answer.

  returns_matrix         Trade lists -> one column of per-period P&L per
                         configuration, on a shared calendar. What PBO and the
                         deflated Sharpe need, and what the engine never made.

  pbo                    Probability of Backtest Overfitting via
                         combinatorially symmetric cross-validation (Bailey,
                         Borwein, Lopez de Prado & Zhu 2014). Split the history
                         into S blocks; for every way of choosing half of them
                         as in-sample, take the in-sample best and see where it
                         ranks out of sample. PBO is the share of splits where
                         it lands at or below the median. Pure noise gives 0.5;
                         a real, persistent edge gives near 0. It is the formal
                         version of our "5 survivors where chance predicts 10".

  deflated_sharpe        A Sharpe ratio picked as the best of N tries is biased
                         upward by the picking. The deflated Sharpe (Bailey &
                         Lopez de Prado 2014) asks how likely the true Sharpe
                         beats the best you would expect from N tries of pure
                         noise, after correcting for skew and fat tails.

Only numpy and the standard library (statistics.NormalDist), on purpose.
"""

from __future__ import annotations

import math
from itertools import combinations
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329


# ── bootstrap ─────────────────────────────────────────────────────────

def stationary_bootstrap(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """One resample of indices 0..n-1: runs of consecutive indices (wrapping)
    whose lengths are geometric with the given mean."""
    if n <= 0:
        return np.empty(0, dtype=int)
    p = 1.0 / max(mean_block, 1.0)
    starts = rng.random(n) < p
    starts[0] = True
    run_id = np.cumsum(starts) - 1
    run_pos = np.flatnonzero(starts)
    first = rng.integers(0, n, size=len(run_pos))
    offset = np.arange(n) - run_pos[run_id]
    return (first[run_id] + offset) % n


def _pf(x: np.ndarray) -> float:
    pos = x[x > 0].sum()
    neg = -x[x <= 0].sum()
    return float(pos / neg) if neg > 0 else np.nan


def pf_interval(pnl, reps: int = 2000, mean_block: float | None = None,
                seed: int = 0, q=(5, 95)) -> dict:
    """
    Profit-factor interval from a TIME-ORDERED pnl sequence. mean_block=None
    uses n ** (1/3), the usual rule; mean_block=1 reproduces the iid bootstrap.
    """
    x = np.asarray(pd.to_numeric(pd.Series(pnl), errors="coerce").dropna(), dtype=float)
    n = len(x)
    L = n ** (1 / 3) if mean_block is None else mean_block
    rng = np.random.default_rng(seed)
    s = np.array([_pf(x[stationary_bootstrap(n, L, rng)]) for _ in range(reps)])
    s = s[np.isfinite(s)]
    lo, hi = np.percentile(s, q) if len(s) else (np.nan, np.nan)
    return {"lo": float(lo), "hi": float(hi), "below_1": float((s < 1.0).mean()) if len(s) else np.nan,
            "mean_block": float(L), "n": n}


# ── returns matrix ────────────────────────────────────────────────────

def returns_matrix(trades: pd.DataFrame, by: str = "cfg", value: str = "pnl_usd",
                   when: str = "exit_time", freq: str = "W") -> pd.DataFrame:
    """
    One column per configuration, one row per calendar period, P&L booked at
    exit. Periods with no exits are 0 - a strategy that is flat earned nothing.
    Every column shares the same full calendar, which is what CSCV requires.
    """
    t = trades[[by, value, when]].copy()
    t[when] = pd.to_datetime(t[when], utc=True)
    per = t[when].dt.tz_localize(None).dt.to_period(freq)
    m = t.groupby([per, t[by]])[value].sum().unstack(by)
    full = pd.period_range(m.index.min(), m.index.max(), freq=freq)
    return m.reindex(full).fillna(0.0)


# ── PBO via CSCV ──────────────────────────────────────────────────────

def _sharpe_from_moments(n, s, q):
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = q / n - mean ** 2
        sr = mean / np.sqrt(var)
    sr[~np.isfinite(sr)] = np.nan
    return sr


def pbo(M: pd.DataFrame | np.ndarray, S: int = 16) -> dict:
    """
    M: T periods x N configurations of returns. S: number of blocks (even).
    Returns the PBO, the logit distribution, and how far the in-sample winner's
    Sharpe falls out of sample on average.
    """
    X = np.asarray(M, dtype=float)
    T, N = X.shape
    if S % 2 or S < 2:
        raise ValueError("S must be an even number >= 2")
    if T < 2 * S:
        raise ValueError(f"need at least {2*S} periods for S={S}, have {T}")
    if N < 2:
        raise ValueError("PBO needs at least two configurations")
    blocks = np.array_split(np.arange(T), S)
    cnt = np.array([len(b) for b in blocks], dtype=float)
    s1 = np.stack([X[b].sum(0) for b in blocks])            # (S, N)
    s2 = np.stack([(X[b] ** 2).sum(0) for b in blocks])
    tot_n, tot_s, tot_q = cnt.sum(), s1.sum(0), s2.sum(0)

    combos = np.array(list(combinations(range(S), S // 2)))  # (C, S/2)
    is_n = cnt[combos].sum(1)[:, None]
    is_s = s1[combos].sum(1)
    is_q = s2[combos].sum(1)
    r_is = _sharpe_from_moments(is_n, is_s, is_q)
    r_os = _sharpe_from_moments(tot_n - is_n, tot_s - is_s, tot_q - is_q)

    best = np.nanargmax(np.where(np.isnan(r_is), -np.inf, r_is), axis=1)
    rows = np.arange(len(combos))
    os_filled = np.where(np.isnan(r_os), -np.inf, r_os)
    # rank 1..N ascending (1 = worst); exact ties take column order, which
    # only matters for flat configs and is conservative for noise
    order = os_filled.argsort(axis=1).argsort(axis=1) + 1.0
    rank = order[rows, best]
    w = rank / (N + 1.0)
    lam = np.log(w / (1.0 - w))
    return {
        "pbo": float((lam <= 0).mean()),
        "logits": lam,
        "n_splits": int(len(combos)),
        "S": S, "T": T, "N": N,
        "is_sharpe_of_winner": float(np.nanmean(r_is[rows, best])),
        "oos_sharpe_of_winner": float(np.nanmean(r_os[rows, best])),
        "oos_median_sharpe": float(np.nanmean(np.nanmedian(r_os, axis=1))),
        # the paper's companion to PBO: how often whatever won in-sample LOSES
        # out of sample. PBO alone cannot tell a broad plateau (every config
        # good, so picking cannot help, PBO ~ 0.5) from noise; this can.
        "prob_oos_loss": float((np.nan_to_num(r_os[rows, best], nan=-1.0) < 0).mean()),
    }


def pbo_null(M: pd.DataFrame | np.ndarray, S: int = 16, reps: int = 200,
             mean_block: float = 8.0, seed: int = 0) -> np.ndarray:
    """
    PBO values this matrix would produce if NOTHING in it had an edge.

    Why it exists: a single PBO on pure noise ranges ~0.23-0.79 (5th-95th, 300
    trials, N=40), so "PBO < 0.5" is passed by noise half the time. The null
    here keeps everything about the real matrix except edge: each column is
    demeaned (no config has an edge), then rows are block-resampled together
    (the configs keep their real correlation with each other - a family of
    near-identical variants behaves like far fewer independent tries - and
    some of the time-clustering survives).
    """
    X = np.asarray(M, dtype=float)
    X0 = X - X.mean(axis=0, keepdims=True)
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for k in range(reps):
        out[k] = pbo(X0[stationary_bootstrap(len(X0), mean_block, rng)], S=S)["pbo"]
    return out


def pbo_test(M: pd.DataFrame | np.ndarray, S: int = 16, reps: int = 200, seed: int = 0) -> dict:
    """PBO plus where it sits against its own zero-edge null."""
    r = pbo(M, S=S)
    null = pbo_null(M, S=S, reps=reps, seed=seed)
    r["null_p"] = float((null <= r["pbo"]).mean())      # share of zero-edge worlds at least this good
    r["null_5th"] = float(np.percentile(null, 5))
    r["null_median"] = float(np.median(null))
    return r


# ── deflated Sharpe ───────────────────────────────────────────────────

def expected_max_sharpe(var_sr: float, n_trials: int) -> float:
    """The best Sharpe you would expect from n_trials of noise with this spread."""
    if n_trials <= 1 or not np.isfinite(var_sr) or var_sr <= 0:
        return 0.0
    g = EULER_GAMMA
    return math.sqrt(var_sr) * ((1 - g) * _N.inv_cdf(1 - 1 / n_trials)
                                + g * _N.inv_cdf(1 - 1 / (n_trials * math.e)))


def probabilistic_sharpe(sr: float, sr0: float, T: int, skew: float, kurt: float) -> float:
    """P(true Sharpe > sr0). kurt is plain kurtosis (normal = 3), not excess."""
    den = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if T < 2 or den <= 0 or not np.isfinite(den):
        return np.nan
    return _N.cdf((sr - sr0) * math.sqrt(T - 1) / math.sqrt(den))


def deflated_sharpe(returns, trial_sharpes, n_trials: int | None = None) -> dict:
    """
    returns         per-period returns of the configuration you picked
    trial_sharpes   the per-period Sharpe of EVERY configuration tried
    n_trials        defaults to len(trial_sharpes); pass more if you also
                    tried things that are not in the matrix
    """
    r = pd.Series(np.asarray(returns, dtype=float)).dropna()
    sd = r.std(ddof=1)
    sr = float(r.mean() / sd) if sd > 0 else np.nan
    trials = np.asarray(trial_sharpes, dtype=float)
    trials = trials[np.isfinite(trials)]
    n = int(n_trials if n_trials is not None else len(trials))
    var_sr = float(trials.var(ddof=1)) if len(trials) > 1 else 0.0
    sr0 = expected_max_sharpe(var_sr, n)
    skew = float(r.skew()) if len(r) > 2 else 0.0
    kurt = float(r.kurt()) + 3.0 if len(r) > 3 else 3.0
    return {"sharpe": sr, "sr_threshold": sr0, "n_trials": n, "T": int(len(r)),
            "skew": skew, "kurtosis": kurt,
            "psr_vs_zero": probabilistic_sharpe(sr, 0.0, len(r), skew, kurt),
            "dsr": probabilistic_sharpe(sr, sr0, len(r), skew, kurt)}


# ── Kaufman (Trading Systems and Methods, Ch 2) ───────────────────────

def shape(x) -> dict:
    """Skew and plain kurtosis (normal = 3) of a return series. Kaufman's
    warning: a system whose returns show kurtosis above ~7-8 has too many
    similar-sized wins to be believable. Read it WITH the skew: high kurtosis
    plus positive skew is a few huge winners (the trend-follower's profile,
    e.g. Storm Chaser), plus negative skew is many small wins and rare large
    losses - the overfit (or short-option) profile he is warning about."""
    r = pd.Series(np.asarray(x, dtype=float)).dropna()
    if len(r) < 4:
        return {"n": len(r), "skew": np.nan, "kurtosis": np.nan}
    return {"n": int(len(r)), "skew": float(r.skew()), "kurtosis": float(r.kurt()) + 3.0}


def prob_reach(returns, start: float, target: float, n_periods: int) -> float:
    """Kaufman's probability of ending at or above `target` after n periods,
    from the log of (1 + periodic return), assumed normal:
        z = (ln(T/B) - n * mean(ln(1+r))) / (sd(ln(1+r)) * sqrt(n));  P = 1 - N(z)
    It says nothing about the path - see path_odds for hitting a floor first."""
    lr = np.log1p(np.asarray(returns, dtype=float))
    lr = lr[np.isfinite(lr)]
    if len(lr) < 3 or n_periods <= 0:
        return np.nan
    mu, sd = lr.mean(), lr.std(ddof=1)
    if sd == 0:
        return float(n_periods * mu >= math.log(target / start))
    z = (math.log(target / start) - n_periods * mu) / (sd * math.sqrt(n_periods))
    return 1.0 - _N.cdf(z)


def path_odds(returns, start: float, target: float, floor: float, n_periods: int,
              reps: int = 5000, mean_block: float = 10.0, seed: int = 0) -> dict:
    """Resample the strategy's own per-trade returns (stationary block
    bootstrap, so losing streaks stay together) into `reps` paths of
    `n_periods` trades from `start`. Returns the share of paths that reach
    `target` before `floor`, that hit `floor` first, and that end at or above
    `target`, plus the median ending value. A return of -100% or worse ends
    the path at zero."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return {"p_target_first": np.nan, "p_floor_first": np.nan, "p_end_above": np.nan, "median_end": np.nan}
    rng = np.random.default_rng(seed)
    up = down = above = 0
    ends = np.empty(reps)
    for k in range(reps):
        idx = stationary_bootstrap(n, mean_block, rng)
        while len(idx) < n_periods:
            idx = np.concatenate([idx, stationary_bootstrap(n, mean_block, rng)])
        eq, hit = start, 0
        for j in idx[:n_periods]:
            eq *= max(1.0 + r[j], 0.0)
            if hit == 0 and eq >= target:
                hit = 1
            elif hit == 0 and eq <= floor:
                hit = -1
        up += hit == 1
        down += hit == -1
        above += eq >= target
        ends[k] = eq
    return {"p_target_first": up / reps, "p_floor_first": down / reps,
            "p_end_above": above / reps, "median_end": float(np.median(ends))}


# ── Kaufman Ch 2 p.58: return per unit of risk, and how much is just the market ──

def annualised_return(returns, periods_per_year: float, geometric: bool = True) -> float:
    """Compounded annual rate (geometric=True: prod(1+r)^(ppy/n) - 1; an account
    that hit zero returns -1) or the arithmetic mean x periods_per_year."""
    r = np.asarray(pd.to_numeric(pd.Series(returns), errors="coerce").dropna(), dtype=float)
    if not len(r):
        return np.nan
    if not geometric:
        return float(r.mean() * periods_per_year)
    growth = float(np.prod(1.0 + r))
    if growth <= 0:
        return -1.0
    return growth ** (periods_per_year / len(r)) - 1.0


def information_ratio(returns, periods_per_year: float, geometric: bool = True) -> float:
    """Kaufman's information ratio (p.58): annualised return / annualised
    volatility, volatility = SD of the periodic returns x sqrt(periods_per_year).
    It is a Sharpe ratio with no risk-free rate - NOT the fund-manager IR of
    active return over tracking error. A series with zero volatility returns
    +-inf (or NaN if it also earned nothing)."""
    r = np.asarray(pd.to_numeric(pd.Series(returns), errors="coerce").dropna(), dtype=float)
    if len(r) < 2:
        return np.nan
    ann = annualised_return(r, periods_per_year, geometric)
    vol = float(r.std(ddof=1)) * math.sqrt(periods_per_year)
    if vol <= 1e-15 * max(1.0, abs(float(r.mean()))):
        return math.copysign(math.inf, ann) if ann != 0 else np.nan
    return ann / vol


def _paired(a, b, rf=0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Strategy, benchmark and annual rf -> aligned float arrays, rows with any
    NaN dropped. Two pandas Series are aligned on their index (inner join; a
    Series rf is aligned the same way); anything else must be the same length."""
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        j = pd.concat([pd.to_numeric(a, errors="coerce").rename("a"),
                       pd.to_numeric(b, errors="coerce").rename("b")], axis=1, join="inner")
        if isinstance(rf, pd.Series):
            j["rf"] = pd.to_numeric(rf, errors="coerce").reindex(j.index)
        elif np.ndim(rf) == 0:
            j["rf"] = float(rf)
        elif len(rf) == len(j):
            j["rf"] = np.asarray(rf, dtype=float)
        else:
            raise ValueError("an array rf must have one value per aligned period")
        j = j.dropna()
        return j.a.to_numpy(float), j.b.to_numpy(float), j.rf.to_numpy(float)
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"strategy and benchmark lengths differ: {x.shape} vs {y.shape}")
    r = np.full(x.shape, float(rf)) if np.ndim(rf) == 0 else np.asarray(rf, dtype=float)
    if r.shape != x.shape:
        raise ValueError("an array rf must have one value per period")
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
    return x[ok], y[ok], r[ok]


def beta_alpha(strategy_returns, benchmark_returns, periods_per_year: float,
               rf=0.0, geometric: bool = True) -> dict:
    """
    How much of a strategy is just the market (Kaufman p.58).

      beta       cov(s - rf, b - rf) / var(b - rf): the strategy's sensitivity
                 to the benchmark. A long-only system that is in the market
                 half the time has a beta near 0.5 before it does anything clever.
      alpha      Jensen's alpha, annualised ARITHMETICALLY (it is a regression
                 intercept): (mean(s - rf) - beta x mean(b - rf)) x ppy.
      treynor    (annualised return - rf) / beta: return per unit of MARKET risk.
      correlation, r_squared (= share of the strategy's variance the benchmark
                 explains), information_ratio of each (Kaufman's definition).

    rf is an ANNUAL rate as a decimal (0.05 = 5%), either one number or one per
    period (e.g. the policy rate in force each day); it is divided by
    periods_per_year. "annualised return" for Treynor and the information
    ratio is compounded when geometric=True (the default), arithmetic otherwise.

    beta_return / alpha share: the arithmetic annual excess return of the
    strategy splits EXACTLY into beta x the benchmark's plus alpha, so
    beta_share = beta x bench / strategy says how much of the return the
    market exposure alone delivered.

    alpha_se is the ordinary least-squares standard error of the annualised
    alpha (assumes independent periods) - a size for the noise, not a test.
    """
    s, b, rf_a = _paired(strategy_returns, benchmark_returns, rf)
    n = len(s)
    out = {"n": n, "beta": np.nan, "alpha_ann": np.nan, "alpha_se_ann": np.nan,
           "correlation": np.nan, "r_squared": np.nan, "treynor": np.nan}
    if n < 3:
        return out
    rf_p = rf_a / periods_per_year
    xs, xb = s - rf_p, b - rf_p
    vb = float(np.var(xb, ddof=1))
    if vb <= 0:
        return out
    beta = float(np.cov(xs, xb, ddof=1)[0, 1] / vb)
    a_p = float(xs.mean() - beta * xb.mean())
    resid = xs - a_p - beta * xb
    s2 = float((resid ** 2).sum() / (n - 2)) if n > 2 else np.nan
    sxx = float(((xb - xb.mean()) ** 2).sum())
    se_a = math.sqrt(s2 * (1.0 / n + xb.mean() ** 2 / sxx)) if sxx > 0 and np.isfinite(s2) else np.nan
    sd_s = float(np.std(s, ddof=1))
    corr = float(np.corrcoef(s, b)[0, 1]) if sd_s > 0 else np.nan
    ann_s = annualised_return(s, periods_per_year, geometric)
    ann_b = annualised_return(b, periods_per_year, geometric)
    rf_ann = float(rf_a.mean())
    ex_s, ex_b = float(xs.mean() * periods_per_year), float(xb.mean() * periods_per_year)
    out.update({
        "beta": beta,
        "alpha_ann": a_p * periods_per_year,
        "alpha_se_ann": se_a * periods_per_year if np.isfinite(se_a) else np.nan,
        "correlation": corr,
        "r_squared": corr ** 2 if np.isfinite(corr) else np.nan,
        "treynor": (ann_s - rf_ann) / beta if beta != 0 else np.nan,
        "ann_return": ann_s, "bench_ann_return": ann_b, "rf_ann": rf_ann,
        "excess_arith_ann": ex_s, "bench_excess_arith_ann": ex_b,
        "beta_return_ann": beta * ex_b,
        "beta_share": beta * ex_b / ex_s if ex_s != 0 else np.nan,
        "information_ratio": information_ratio(s, periods_per_year, geometric),
        "bench_information_ratio": information_ratio(b, periods_per_year, geometric),
    })
    return out
