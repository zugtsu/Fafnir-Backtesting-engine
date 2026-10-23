"""save_result writes the table and a registry row that identifies what made it."""
from __future__ import annotations

import pandas as pd

from saty_bt import provenance as pv


def test_save_result_logs_lineage(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "RESULTS", tmp_path)
    monkeypatch.setattr(pv, "REGISTRY", tmp_path / "registry.csv")
    monkeypatch.setattr(pv, "ROOT", tmp_path.parent)
    df = pd.DataFrame({"pf": [1.2, 0.9]})
    a = {"x": 1, "costs": {"spread": 0.014}, "_root": "ignored"}
    b = {"x": 1, "costs": {"spread": 0.028}, "_root": "also ignored"}

    p = pv.save_result(df, "demo", cfg=a, seed=7, note="first")
    pv.save_result(df, "demo", cfg=b, note="second")

    assert pd.read_csv(p).equals(df)
    h = pv.history("demo")
    assert len(h) == 2
    assert h.config_sha.iloc[0] != h.config_sha.iloc[1]        # a spread change is visible
    assert h.seed.iloc[0] == 7 and h.note.tolist() == ["first", "second"]
    assert set(h.dirty.astype(str)) <= {"True", "False"}


def test_config_sha_ignores_private_keys_and_order():
    assert pv.config_sha({"a": 1, "b": 2, "_root": "x"}) == pv.config_sha({"b": 2, "a": 1, "_root": "y"})
    assert pv.config_sha(None) == ""
