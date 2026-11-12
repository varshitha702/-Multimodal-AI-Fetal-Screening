"""Tests for the leakage-safe HC18 split (ml/split.py)."""
import pandas as pd
import pytest

from ml.split import CSV_PATH, HC_COL, group_id, load_split, make_split


def test_group_id_parses_hc18_names():
    assert group_id("014_HC.png") == "014"
    assert group_id("014_3HC.png") == "014"
    with pytest.raises(ValueError):
        group_id("bad.png")


def _fake_table(n_groups=200):
    rows = []
    for g in range(n_groups):
        for k in range(1 + g % 3):
            name = f"{g:03d}_HC.png" if k == 0 else f"{g:03d}_{k + 1}HC.png"
            rows.append({"filename": name, HC_COL: 50 + g, "pixel size(mm)": 0.1})
    return pd.DataFrame(rows)


def test_split_is_deterministic_and_group_disjoint():
    t = _fake_table()
    a, b = make_split(t, seed=1), make_split(t, seed=1)
    assert a["split"].tolist() == b["split"].tolist()
    assert (a.groupby("group")["split"].nunique() == 1).all()  # no group in two splits
    assert set(a["split"]) == {"train", "val", "test"}


def test_different_seed_changes_split():
    t = _fake_table()
    assert make_split(t, seed=1)["split"].tolist() != make_split(t, seed=2)["split"].tolist()


@pytest.mark.skipif(not CSV_PATH.is_file(), reason="HC18 dataset not downloaded")
def test_committed_split_matches_regeneration_and_has_no_leakage():
    committed = load_split()
    regenerated = make_split(pd.read_csv(CSV_PATH))
    assert len(committed) == 999
    assert committed["split"].tolist() == regenerated["split"].tolist()
    assert (committed.groupby("group")["split"].nunique() == 1).all()
    frac = committed["split"].value_counts(normalize=True)
    assert frac["train"] == pytest.approx(0.70, abs=0.03)
    assert frac["val"] == pytest.approx(0.15, abs=0.03)
    assert frac["test"] == pytest.approx(0.15, abs=0.03)
