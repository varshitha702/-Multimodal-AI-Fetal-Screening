"""Tests for the CTG branch (ml/ctg.py). Model tests use the committed outputs/ctg_xgb.json."""
import json

import numpy as np
import pandas as pd
import pytest

from ml.ctg import (
    CLASSES,
    EXAMPLES_PATH,
    FEATURE_NAMES,
    FEATURES,
    MODEL_PATH,
    SPLIT_PATH,
    XLS_PATH,
    CTGModel,
    load_ctg,
    make_split,
)

UCI_ORDER = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV", "MLTV", "Width", "Min", "Max",
             "Nmax", "Nzeros", "Mode", "Mean", "Median", "Variance", "Tendency"]
needs_xls = pytest.mark.skipif(not XLS_PATH.is_file(), reason="needs data/raw/CTG/CTG.xls (not committed)")


def test_feature_definitions_match_uci():
    assert FEATURE_NAMES == UCI_ORDER
    assert all(f.lo < f.hi and f.description for f in FEATURES)


def synthetic(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, 21)), columns=FEATURE_NAMES)
    df["NSP"] = rng.choice([1, 2, 3], size=n, p=[0.78, 0.14, 0.08])
    df["record"] = [f"R{i}" for i in range(n)]
    dup = df.iloc[:5].copy()
    dup["record"] = [f"D{i}" for i in range(5)]
    return pd.concat([df, dup], ignore_index=True)


def test_split_keeps_duplicates_together_and_is_stratified():
    df = make_split(synthetic())
    for g, rows in df.groupby("dup_group"):
        assert rows.split.nunique() == 1, g
    frac = df.split.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03 and abs(frac["test"] - 0.15) < 0.03
    for s in ("train", "val", "test"):
        share = (df[df.split == s].NSP == 3).mean()
        assert abs(share - (df.NSP == 3).mean()) < 0.04
    assert make_split(synthetic()).split.tolist() == df.split.tolist()      # deterministic


@pytest.fixture(scope="module")
def model():
    return CTGModel()


def test_committed_model_and_examples_exist():
    assert MODEL_PATH.is_file() and EXAMPLES_PATH.is_file()
    ex = json.loads(EXAMPLES_PATH.read_text())
    assert {e["label"] for e in ex} == set(CLASSES)
    assert all(e["split"] == "test" and set(e["features"]) == set(FEATURE_NAMES) for e in ex)


def test_examples_are_predicted_as_labelled(model):
    for e in json.loads(EXAMPLES_PATH.read_text()):
        r = model.predict(e["features"])
        assert r.predicted_class == e["label"], e["record"]
        assert sum(r.probabilities.values()) == pytest.approx(1, abs=1e-3)


def test_top_factors_sorted_and_explain_prediction(model):
    e = json.loads(EXAMPLES_PATH.read_text())[-1]
    r = model.predict(e["features"], top_k=5)
    assert len(r.top_factors) == 5
    mags = [abs(f["shap"]) for f in r.top_factors]
    assert mags == sorted(mags, reverse=True)
    assert all(f["feature"] in FEATURE_NAMES and f["direction"] in ("towards", "away from") for f in r.top_factors)


def test_prediction_is_deterministic(model):
    feats = json.loads(EXAMPLES_PATH.read_text())[0]["features"]
    assert model.predict(feats).probabilities == model.predict(feats).probabilities


@needs_xls
def test_data_load_matches_uci_counts():
    df = load_ctg()
    assert len(df) == 2126 and df.record.is_unique
    assert df.NSP.value_counts().to_dict() == {1: 1655, 2: 295, 3: 176}
    assert df.AC.max() < 0.05     # per-second rates, not counts


@needs_xls
def test_committed_split_matches_regenerated():
    committed = pd.read_csv(SPLIT_PATH)
    regen = make_split(load_ctg())
    assert committed.split.tolist() == regen.split.tolist()
