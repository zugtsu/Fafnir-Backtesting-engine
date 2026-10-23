"""
Golden masters - see saty_bt/golden.py for what and why.

A failure here means one of two different things, and the message says which:
the data underneath changed (recapture deliberately), or the code changed a
fill (a regression, unless the strategy change was intended).

Skip these for a quick unit run with:  python -m pytest -m "not golden"
"""
from __future__ import annotations

import pytest

from saty_bt.golden import FIXTURES, SPECS, compare


@pytest.mark.golden
@pytest.mark.parametrize("name", list(SPECS))
def test_golden(name):
    if not (FIXTURES / f"{name}.meta.json").exists():
        pytest.fail(f"no fixture for {name}: python scripts/capture_golden.py {name} --why \"...\"")
    try:
        status, msg = compare(name)
    except FileNotFoundError:
        pytest.skip("market data not built here; the golden masters replay real bars")
    if status == "data_changed":
        pytest.fail("DATA CHANGED (not a code regression)\n" + msg)
    assert status == "ok", "CODE CHANGED A FILL\n" + msg
