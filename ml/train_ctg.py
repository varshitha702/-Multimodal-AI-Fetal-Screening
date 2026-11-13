"""Train and evaluate the CTG NSP classifier (Feature C).

    python -m ml.train_ctg

* Split: data/splits/ctg_split.csv (70/15/15, stratified, duplicate records kept together).
* Class imbalance: "balanced" class weights computed on the TRAIN split only (no resampling, nothing
  touches validation or test).
* Models: logistic regression (standardised), random forest, XGBoost (early stopping on validation log-loss).
  All hyper-parameters are fixed below; the validation split is used only for XGBoost early stopping.
* Test metrics for all three: accuracy, macro-F1 (bootstrap 95 % CI), per-class precision/recall
  (Pathological recall highlighted), macro one-vs-rest ROC-AUC, confusion matrix.
* SHAP TreeExplainer on the XGBoost model: global mean |SHAP| per feature and class.
Writes outputs/ctg_xgb.json (deployed model), ctg_model_meta.json, ctg_metrics.json, ctg_test_predictions.csv,
ctg_examples.json (held-out test records for the UI) and docs/screenshots/ctg_*.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402

from ml.ctg import CLASSES, EXAMPLES_PATH, FEATURE_NAMES, META_PATH, MODEL_PATH, SEED, load_split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SHOTS = ROOT / "docs" / "screenshots"


def evaluate(y: np.ndarray, proba: np.ndarray, n_boot: int = 2000) -> dict:
    pred = proba.argmax(1)
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=[0, 1, 2], zero_division=0)
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        boots.append(f1_score(y[i], pred[i], average="macro", labels=[0, 1, 2], zero_division=0))
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "macro_f1_ci95": [round(float(v), 4) for v in np.percentile(boots, [2.5, 97.5])],
        "roc_auc_macro_ovr": round(float(roc_auc_score(y, proba, multi_class="ovr", average="macro")), 4),
        "per_class": {c: {"precision": round(float(p[k]), 4), "recall": round(float(r[k]), 4),
                          "f1": round(float(f[k]), 4), "support": int(s[k])} for k, c in enumerate(CLASSES)},
        "pathological_recall": round(float(r[2]), 4),
        "confusion_matrix": {"labels": CLASSES, "rows_true_cols_pred": confusion_matrix(y, pred).tolist()},
    }


def main() -> None:
    df = load_split()
    counts = pd.crosstab(df.split, df.NSP.map(dict(enumerate(CLASSES, 1))))
    print(counts.to_string())
    xs = {k: df[df.split == k][FEATURE_NAMES] for k in ("train", "val", "test")}
    ys = {k: df[df.split == k].NSP.to_numpy() - 1 for k in ("train", "val", "test")}
    w_train = compute_sample_weight("balanced", ys["train"])        # imbalance: train labels only

    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", C=1.0)),
        "random_forest": RandomForestClassifier(n_estimators=500, class_weight="balanced", min_samples_leaf=1,
                                                random_state=SEED, n_jobs=-1),
        "xgboost": xgb.XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=4, subsample=0.9,
                                     colsample_bytree=0.8, objective="multi:softprob", eval_metric="mlogloss",
                                     early_stopping_rounds=100, random_state=SEED, n_jobs=4),
    }
    models["logistic_regression"].fit(xs["train"], ys["train"])
    models["random_forest"].fit(xs["train"], ys["train"])
    models["xgboost"].fit(xs["train"], ys["train"], sample_weight=w_train,
                          eval_set=[(xs["val"], ys["val"])], verbose=False)

    metrics = {"split_counts": counts.to_dict(orient="index"), "imbalance": "balanced class weights from the train "
               "split only", "models": {}}
    for name, m in models.items():
        metrics["models"][name] = {"val": evaluate(ys["val"], m.predict_proba(xs["val"]), n_boot=200),
                                   "test": evaluate(ys["test"], m.predict_proba(xs["test"]))}
        t = metrics["models"][name]["test"]
        print(f"{name:20s} test macro-F1 {t['macro_f1']:.3f} {t['macro_f1_ci95']}  acc {t['accuracy']:.3f}  "
              f"AUC {t['roc_auc_macro_ovr']:.3f}  Pathological recall {t['pathological_recall']:.3f}")
    xgbm = models["xgboost"]
    metrics["models"]["xgboost"]["best_iteration"] = int(xgbm.best_iteration)

    # --- SHAP (global) on the test split ---
    explainer = shap.TreeExplainer(xgbm)
    sv = np.asarray(explainer.shap_values(xs["test"]))          # (n, features, classes)
    mean_abs = np.abs(sv).mean(0)                                # (features, classes)
    metrics["shap_mean_abs_test"] = {c: dict(sorted(((f, round(float(mean_abs[i, k]), 4))
                                                    for i, f in enumerate(FEATURE_NAMES)),
                                                   key=lambda kv: -kv[1])[:8]) for k, c in enumerate(CLASSES)}
    (OUT / "ctg_metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- deploy XGBoost: truncated to the best iteration, so the plain Booster, its predictions and SHAP all use
    # exactly the trees selected by early stopping ---
    xgbm.get_booster()[: xgbm.best_iteration + 1].save_model(MODEL_PATH)
    test_df = df[df.split == "test"].reset_index(drop=True)
    proba = xgbm.predict_proba(xs["test"])
    META_PATH.write_text(json.dumps({
        "model": "XGBoost (multi:softprob), balanced class weights from train", "classes": CLASSES,
        "features": FEATURE_NAMES, "best_iteration": int(xgbm.best_iteration),
        "params": {k: v for k, v in xgbm.get_params().items() if k in
                   ("n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree")},
        "test": {k: metrics["models"]["xgboost"]["test"][k] for k in
                 ("n", "accuracy", "macro_f1", "macro_f1_ci95", "roc_auc_macro_ovr", "pathological_recall")},
        "data": "UCI Cardiotocography (Ayres-de-Campos et al., 2000), 2,126 records, 70/15/15 stratified split",
    }, indent=2))
    pd.DataFrame({"record": test_df.record, "true": [CLASSES[i] for i in ys["test"]],
                  "pred": [CLASSES[i] for i in proba.argmax(1)],
                  **{f"p_{c}": proba[:, k].round(4) for k, c in enumerate(CLASSES)}}).to_csv(
        OUT / "ctg_test_predictions.csv", index=False)

    # Examples for the UI "load example" button: held-out TEST records, correctly classified, most confident
    examples = []
    for k, c in enumerate(CLASSES):
        idx = np.flatnonzero((ys["test"] == k) & (proba.argmax(1) == k))
        for i in idx[np.argsort(-proba[idx, k])][:2]:
            examples.append({"record": test_df.record[i], "label": c, "split": "test",
                             "features": {f: float(test_df.loc[i, f]) for f in FEATURE_NAMES}})
    EXAMPLES_PATH.write_text(json.dumps(examples, indent=2))

    # --- figures ---
    SHOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (name, m) in zip(axes, metrics["models"].items(), strict=True):
        cm = np.array(m["test"]["confusion_matrix"]["rows_true_cols_pred"])
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if norm[i, j] > 0.5 else "black")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{name}\nmacro-F1 {m['test']['macro_f1']:.3f}, Pathological recall "
                     f"{m['test']['pathological_recall']:.3f}", fontsize=10)
    fig.suptitle(f"CTG held-out test split (n={len(ys['test'])})")
    fig.tight_layout()
    fig.savefig(SHOTS / "ctg_confusion.png", dpi=120)
    plt.close(fig)

    plt.figure()
    shap.summary_plot(sv[:, :, 2], xs["test"], show=False, max_display=12)
    plt.title("SHAP values toward 'Pathological' (XGBoost, test split)", fontsize=10)
    plt.tight_layout()
    plt.savefig(SHOTS / "ctg_shap_pathological.png", dpi=120)
    plt.close("all")
    print(json.dumps(metrics["shap_mean_abs_test"]["Pathological"], indent=1))


if __name__ == "__main__":
    main()
