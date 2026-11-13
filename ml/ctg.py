"""CTG branch (Feature C): UCI Cardiotocography data, NSP classifier, SHAP explanations.

Data: Ayres-de-Campos et al., "SisPorto 2.0: a program for automated analysis of cardiotocograms",
J Matern Fetal Med 2000; UCI Machine Learning Repository, dataset 193 ("Cardiotocography").
2,126 CTG records, 21 features computed by SisPorto, NSP label from three expert obstetricians:
1 = Normal, 2 = Suspect, 3 = Pathological.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
XLS_PATH = ROOT / "data" / "raw" / "CTG" / "CTG.xls"
SPLIT_PATH = ROOT / "data" / "splits" / "ctg_split.csv"
MODEL_PATH = ROOT / "outputs" / "ctg_xgb.json"
META_PATH = ROOT / "outputs" / "ctg_model_meta.json"
EXAMPLES_PATH = ROOT / "outputs" / "ctg_examples.json"
CLASSES = ["Normal", "Suspect", "Pathological"]
SEED = 42


@dataclass(frozen=True)
class Feature:
    name: str
    description: str
    unit: str
    lo: float      # API validation range (inclusive): physiologically/technically plausible, wider than the data
    hi: float


# Ranges: dataset range (see docs/DATASET.md) widened to plausible limits; rates are events per second.
FEATURES: tuple[Feature, ...] = (
    Feature("LB", "FHR baseline", "bpm", 50, 240),
    Feature("AC", "Accelerations per second", "1/s", 0, 1),
    Feature("FM", "Fetal movements per second", "1/s", 0, 1),
    Feature("UC", "Uterine contractions per second", "1/s", 0, 1),
    Feature("DL", "Light decelerations per second", "1/s", 0, 1),
    Feature("DS", "Severe decelerations per second", "1/s", 0, 1),
    Feature("DP", "Prolonged decelerations per second", "1/s", 0, 1),
    Feature("ASTV", "Time with abnormal short-term variability", "%", 0, 100),
    Feature("MSTV", "Mean short-term variability", "bpm", 0, 20),
    Feature("ALTV", "Time with abnormal long-term variability", "%", 0, 100),
    Feature("MLTV", "Mean long-term variability", "bpm", 0, 100),
    Feature("Width", "FHR histogram width", "bpm", 0, 250),
    Feature("Min", "FHR histogram minimum", "bpm", 0, 250),
    Feature("Max", "FHR histogram maximum", "bpm", 0, 300),
    Feature("Nmax", "Histogram peaks", "count", 0, 50),
    Feature("Nzeros", "Histogram zeros", "count", 0, 50),
    Feature("Mode", "FHR histogram mode", "bpm", 0, 300),
    Feature("Mean", "FHR histogram mean", "bpm", 0, 300),
    Feature("Median", "FHR histogram median", "bpm", 0, 300),
    Feature("Variance", "FHR histogram variance", "bpm²", 0, 1000),
    Feature("Tendency", "Histogram tendency (-1 left, 0 symmetric, 1 right asymmetric)", "", -1, 1),
)
FEATURE_NAMES = [f.name for f in FEATURES]


def load_ctg(xls_path: Path = XLS_PATH) -> pd.DataFrame:
    """The 21 features (sheet 'Data', where AC..DP are per-second rates) + NSP + record id (SegFile)."""
    data = pd.read_excel(xls_path, "Data", header=1).iloc[:2126].reset_index(drop=True)
    raw = pd.read_excel(xls_path, "Raw Data").iloc[1:2127].reset_index(drop=True)
    aligned = all((data[c].to_numpy() == raw[c].to_numpy()).all() for c in ("b", "e", "NSP"))
    if not aligned:
        raise ValueError("'Data' and 'Raw Data' sheets are not row-aligned")
    x = data.iloc[:, 10:31].copy()
    x.columns = [str(c).split(".")[0] for c in x.columns]
    if list(x.columns) != FEATURE_NAMES:
        raise ValueError(f"Unexpected feature columns: {list(x.columns)}")
    x = x.astype(float)
    x.insert(0, "record", raw["SegFile"].str.replace(".txt", "", regex=False))
    x["NSP"] = data["NSP"].astype(int)
    if x[FEATURE_NAMES].isna().any().any() or len(x) != 2126:
        raise ValueError("Missing values or wrong number of records")
    return x


def make_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """70/15/15 stratified by NSP. Identical feature vectors (10 groups, 22 records, same label) form one group so
    a duplicate can never be in two splits."""
    from sklearn.model_selection import train_test_split

    group = df.groupby(FEATURE_NAMES, sort=False).ngroup()
    groups = pd.DataFrame({"group": group, "NSP": df.NSP}).drop_duplicates("group")
    g_train, g_rest = train_test_split(groups, test_size=0.30, stratify=groups.NSP, random_state=seed)
    g_val, g_test = train_test_split(g_rest, test_size=0.50, stratify=g_rest.NSP, random_state=seed)
    split = pd.Series("train", index=df.index)
    split[group.isin(g_val.group)] = "val"
    split[group.isin(g_test.group)] = "test"
    return df.assign(split=split.to_numpy(), dup_group=group.to_numpy())


def load_split() -> pd.DataFrame:
    df = load_ctg()
    if SPLIT_PATH.is_file():
        s = pd.read_csv(SPLIT_PATH)
        return df.merge(s[["record", "split"]], on="record", validate="one_to_one")
    df = make_split(df)
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["record", "NSP", "split"]].to_csv(SPLIT_PATH, index=False)
    return df


# --- inference -----------------------------------------------------------------------------------

@dataclass
class CTGResult:
    predicted_class: str
    probabilities: dict[str, float]
    top_factors: list[dict]           # SHAP contributions toward the predicted class, largest |value| first


class CTGModel:
    """XGBoost NSP classifier + SHAP TreeExplainer, loaded once."""

    def __init__(self, model_path: Path = MODEL_PATH, meta_path: Path = META_PATH):
        import shap
        import xgboost as xgb

        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)
        self.meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
        self.explainer = shap.TreeExplainer(self.model)

    def predict(self, features: dict[str, float], top_k: int = 5) -> CTGResult:
        x = pd.DataFrame([[float(features[n]) for n in FEATURE_NAMES]], columns=FEATURE_NAMES)
        proba = self.model.predict_proba(x)[0]
        k = int(proba.argmax())
        sv = np.asarray(self.explainer.shap_values(x))
        contrib = sv[0, :, k] if sv.ndim == 3 else sv[k][0]   # shap >= 0.45: (n, features, classes)
        order = np.argsort(-np.abs(contrib))[:top_k]
        desc = {f.name: f for f in FEATURES}
        factors = [{"feature": FEATURE_NAMES[i], "description": desc[FEATURE_NAMES[i]].description,
                    "unit": desc[FEATURE_NAMES[i]].unit, "value": float(x.iloc[0, i]),
                    "shap": round(float(contrib[i]), 4),
                    "direction": "towards" if contrib[i] > 0 else "away from"} for i in order]
        return CTGResult(CLASSES[k], {c: round(float(p), 4) for c, p in zip(CLASSES, proba, strict=True)}, factors)
