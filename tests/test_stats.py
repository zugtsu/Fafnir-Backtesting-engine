"""
Each statistic is checked on a case where the right answer is known in advance.
A PBO that has not been shown to say 0.5 on noise is not allowed to judge anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from saty_bt.stats import (deflated_sharpe, pbo, pbo_test, pf_interval,
                           returns_matrix, stationary_bootstrap)


def test_bootstrap_indices_and_run_length():
    rng = np.random.default_rng(1)
    idx = stationary_bootstrap(5000, 8.0, rng)
    assert idx.min() >= 0 and idx.max() < 5000
    # a run continues when the next index is the previous + 1 (mod n)
    breaks = (np.diff(idx) % 5000) != 1
    mean_run = len(idx) / (breaks.sum() + 1)
    assert 6.5 < mean_run < 9.5          # geometric with mean 8


def test_mean_block_one_is_iid():
    rng = np.random.default_rng(2)
    idx = stationary_bootstrap(2000, 1.0, rng)
    consecutive = ((np.diff(idx) % 2000) == 1).mean()
    assert consecutive < 0.01            # essentially never continues a run


def test_block_bootstrap_is_wider_when_trades_cluster():
    # winners and losers arrive in runs of 25, like a trend system's good and
    # bad months; overall PF ~1.2. An iid resample breaks the runs up.
    rng = np.random.default_rng(3)
    regime = np.repeat(rng.random(16) < 0.5, 25)
    pnl = np.where(regime, rng.normal(1.0, 1.0, 400), rng.normal(-0.75, 1.0, 400))
    iid = pf_interval(pnl, mean_block=1.0, seed=4)
    blk = pf_interval(pnl, mean_block=25.0, seed=4)
    assert (blk["hi"] - blk["lo"]) > 1.5 * (iid["hi"] - iid["lo"])


def test_pbo_is_half_on_pure_noise():
    # A single noise PBO swings ~0.23-0.79, so 6 seeds can average 0.41 by luck
    # (it did, and an earlier version of this test passed by 0.011). 80 seeds
    # puts the standard error near 0.017; the measured mean over 300 was 0.498.
    vals = [pbo(np.random.default_rng(1000 + s).normal(0, 1, (520, 40)), S=12)["pbo"]
            for s in range(80)]
    assert abs(np.mean(vals) - 0.5) < 0.06, np.mean(vals)


def test_pbo_null_separates_edge_from_noise():
    rng = np.random.default_rng(21)
    M = rng.normal(0, 1, (520, 30))
    M[:, 0] += 0.35
    assert pbo_test(M, S=12, reps=150, seed=1)["null_p"] < 0.05
    # noise should only rarely look that good against its own null
    flagged = sum(pbo_test(np.random.default_rng(300 + s).normal(0, 1, (520, 30)),
                           S=12, reps=150, seed=s)["null_p"] < 0.05 for s in range(10))
    assert flagged <= 2, flagged


def test_pbo_is_low_with_one_real_edge():
    rng = np.random.default_rng(7)
    M = rng.normal(0, 1, (520, 40))
    M[:, 0] += 0.35                                # one config with a real, persistent edge
    r = pbo(M, S=12)
    assert r["pbo"] < 0.05, r["pbo"]
    assert r["oos_sharpe_of_winner"] > r["oos_median_sharpe"]


def test_pbo_rejects_bad_shapes():
    with pytest.raises(ValueError):
        pbo(np.zeros((10, 5)), S=16)               # too few periods
    with pytest.raises(ValueError):
        pbo(np.zeros((200, 5)), S=15)              # odd S


def test_deflated_sharpe_shrinks_as_trials_grow():
    rng = np.random.default_rng(11)
    r = rng.normal(0.08, 1.0, 520)                 # a modest real edge
    trials = rng.normal(0, 0.05, 500)
    d1 = deflated_sharpe(r, trials, n_trials=1)["dsr"]
    d50 = deflated_sharpe(r, trials, n_trials=50)["dsr"]
    d500 = deflated_sharpe(r, trials, n_trials=500)["dsr"]
    assert d1 > d50 > d500
    assert d1 > 0.9                                # alone, it looks real


def test_deflated_sharpe_on_noise_is_unconvincing():
    rng = np.random.default_rng(12)
    M = rng.normal(0, 1, (520, 100))
    srs = M.mean(0) / M.std(0, ddof=1)
    best = int(np.argmax(srs))                     # pick the luckiest of 100 noise series
    d = deflated_sharpe(M[:, best], srs)
    assert d["psr_vs_zero"] > 0.9                  # naive test is fooled...
    assert d["dsr"] < 0.6                          # ...the deflated one is not


def test_returns_matrix_books_pnl_at_exit_on_a_shared_calendar():
    t = pd.DataFrame({
        "cfg": ["a", "a", "b"],
        "pnl_usd": [1.0, 2.0, -1.0],
        "exit_time": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-20"], utc=True),
    })
    M = returns_matrix(t)
    assert list(M.columns) == ["a", "b"]
    assert M["a"].sum() == 3.0 and M["b"].sum() == -1.0
    assert M.iloc[0]["a"] == 3.0                    # both of a's exits fell in week one
    assert (M.loc[M.index[1:-1]] == 0).all().all()  # empty weeks exist and are flat


@pytest.mark.parametrize("case,should_pass", [
    ("noise", False), ("one_edge", True), ("plateau", True), ("family_no_edge", False)])
def test_selection_rule_on_known_worlds(case, should_pass):
    """The gate's selection rule: PBO beats its null, OR the in-sample winner
    almost never loses out of sample. A plateau of good configs must pass even
    though picking among them cannot help (PBO ~ 0.5)."""
    rng = np.random.default_rng(5)
    T, N = 520, 40
    common = rng.normal(0, 1, (T, 1))
    X = {"noise": lambda: rng.normal(0, 1, (T, N)),
         "one_edge": lambda: np.c_[rng.normal(0.35, 1, (T, 1)), rng.normal(0, 1, (T, N - 1))],
         "plateau": lambda: 0.6 * common + 0.8 * rng.normal(0, 1, (T, N)) + 0.12,
         "family_no_edge": lambda: 0.6 * common + 0.8 * rng.normal(0, 1, (T, N))}[case]()
    r = pbo_test(X, S=12, reps=150, seed=1)
    assert (r["null_p"] <= 0.05 or r["prob_oos_loss"] <= 0.05) == should_pass, r
