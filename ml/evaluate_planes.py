"""Evaluate the plane classifier on the OFFICIAL FETAL_PLANES_DB test set, and the gate on HC18.

    python -m ml.evaluate_planes

* plane head (6 classes) and brain sub-plane head (4 classes): accuracy, macro-F1, per-class precision/recall,
  confusion matrices;
* calibration: ECE and NLL before / after temperature scaling (temperatures fitted on validation, not test);
* gate: pass rate of every test category at the validation-chosen threshold;
* HC18 acceptance: share of HC18 images (training + official test set, all trans-thalamic by design)
  that the gate accepts. These images come from a different hospital and machines (domain shift).
Writes outputs/plane_metrics.json, outputs/plane_test_predictions.csv, outputs/hc18_plane_check.csv
and figures in docs/screenshots/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from ml.planes import (  # noqa: E402
    BRAIN,
    PLANES,
    SUBPLANES,
    TT,
    PlaneClassifier,
    expected_calibration_error,
    letterbox,
    tt_score,
)
from ml.train_planes import (  # noqa: E402
    CHECKPOINT_PATH,
    confusion,
    load_images,
    load_split,
    per_class,
    predict_logits,
    read_gray,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SHOTS = ROOT / "docs" / "screenshots"
HC18_DIRS = [ROOT / "data" / "raw" / "HC18" / "training_set", ROOT / "data" / "raw" / "HC18" / "test_set"]
SHORT = {"Fetal brain": "Brain", "Fetal abdomen": "Abdomen", "Fetal femur": "Femur", "Fetal thorax": "Thorax",
         "Maternal cervix": "Cervix", "Other": "Other", "Trans-thalamic": "TT", "Trans-cerebellum": "TC",
         "Trans-ventricular": "TV"}


def head_report(y: np.ndarray, logits: torch.Tensor, temperature: float, names: list[str]) -> tuple[dict, np.ndarray]:
    raw = F.softmax(logits, 1).cpu().numpy()
    cal = F.softmax(logits / temperature, 1).cpu().numpy()
    pred = raw.argmax(1)
    cm = confusion(y, pred, len(names))
    pc = per_class(cm, names)
    yt = torch.from_numpy(y)
    report = {
        "n": int(len(y)),
        "accuracy": round(float((pred == y).mean()), 4),
        "macro_f1": round(float(np.mean([v["f1"] for v in pc.values()])), 4),
        "per_class": pc,
        "confusion_matrix": {"labels": names, "rows_true_cols_pred": cm.tolist()},
        "calibration": {
            "temperature": round(temperature, 4),
            "ece_before": round(expected_calibration_error(raw, y), 4),
            "ece_after": round(expected_calibration_error(cal, y), 4),
            "nll_before": round(float(F.cross_entropy(logits.cpu(), yt)), 4),
            "nll_after": round(float(F.cross_entropy(logits.cpu() / temperature, yt)), 4),
        },
    }
    return report, cal


def reliability(ax, probs: np.ndarray, y: np.ndarray, title: str, n_bins: int = 10) -> None:
    conf, correct = probs.max(1), probs.argmax(1) == y
    edges = np.linspace(0, 1, n_bins + 1)
    xs, accs, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            xs.append(conf[m].mean())
            accs.append(correct[m].mean())
            ns.append(m.sum())
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.plot(xs, accs, "o-")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{title}\nECE {expected_calibration_error(probs, y):.3f}", fontsize=10)


def plot_cm(ax, cm: np.ndarray, names: list[str], title: str) -> None:
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    labels = [SHORT.get(n, n) for n in names]
    ax.set_xticks(range(len(names)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), labels)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=10)


def acceptance_by_hc(hc: pd.DataFrame) -> dict:
    """HC18 training images (reference HC known): acceptance per HC band (small HC = early gestation)."""
    ref = pd.read_csv(HC18_DIRS[0].parent / "training_set_pixel_size_and_HC.csv")
    m = hc[hc.set == "training_set"].merge(ref, left_on="image", right_on="filename")
    bands = pd.cut(m["head circumference (mm)"], [0, 100, 150, 200, 250, 400])
    return {f"{int(b.left)}-{int(b.right)}": {"n": int(len(g)), "acceptance_rate": round(float(g.passed.mean()), 4),
                                              "predicted_brain_rate": round(float((g.pred_plane == "Fetal brain")
                                                                                  .mean()), 4)}
            for b, g in m.groupby(bands, observed=True)}


def main() -> None:
    clf = PlaneClassifier(CHECKPOINT_PATH)
    device = clf.device
    df = load_split()
    images = load_images(df, clf.input_size)
    test = df.split.to_numpy() == "test"
    tdf = df[test].reset_index(drop=True)
    lp, ls = predict_logits(clf.model, torch.from_numpy(images[test]).to(device))
    y, ys = tdf.label.to_numpy(), tdf.sub_label.to_numpy()

    plane_rep, pp = head_report(y, lp, clf.t_plane, PLANES)
    m = ys >= 0
    sub_rep, _ = head_report(ys[m], ls[torch.from_numpy(m).to(device)], clf.t_sub, SUBPLANES)
    sp = F.softmax(ls / clf.t_sub, 1).cpu().numpy()
    p_tt = tt_score(pp, sp)
    passed = p_tt >= clf.threshold

    category = np.where(y == BRAIN, "Brain: " + tdf.brain_plane, tdf.plane)
    gate = {c: {"n": int((category == c).sum()), "pass_rate": round(float(passed[category == c].mean()), 4)}
            for c in sorted(set(category))}
    is_tt = (y == BRAIN) & (ys == TT)
    gate_summary = {
        "threshold": round(clf.threshold, 4),
        "threshold_rule": clf.meta.get("threshold_rule"),
        "tt_acceptance_rate": round(float(passed[is_tt].mean()), 4),
        "non_tt_rejection_rate": round(float((~passed[~is_tt]).mean()), 4),
        "other_brain_planes_rejection_rate": round(float((~passed[(y == BRAIN) & ~is_tt]).mean()), 4),
        "non_brain_rejection_rate": round(float((~passed[y != BRAIN]).mean()), 4),
        "per_category": gate,
    }
    pd.DataFrame({"image": tdf.image, "plane": tdf.plane, "brain_plane": tdf.brain_plane,
                  "pred_plane": [PLANES[i] for i in pp.argmax(1)], "plane_conf": pp.max(1).round(4),
                  "pred_subplane": [SUBPLANES[i] for i in sp.argmax(1)], "p_tt": p_tt.round(4),
                  "passed": passed}).to_csv(OUT / "plane_test_predictions.csv", index=False)

    # --- HC18 acceptance (domain shift check) ---
    rows = []
    for d in HC18_DIRS:
        for path in sorted(d.glob("*.png")):
            if "Annotation" in path.name:
                continue
            r = clf.check(read_gray(path))
            rows.append({"set": d.name, "image": path.name, "pred_plane": r.plane,
                         "plane_conf": round(r.plane_confidence, 4),
                         "pred_subplane": r.brain_subplane, "p_tt": round(r.tt_probability, 4), "passed": r.passed})
    hc = pd.DataFrame(rows)
    hc.to_csv(OUT / "hc18_plane_check.csv", index=False)
    hc18 = {
        "n": int(len(hc)),
        "acceptance_rate": round(float(hc.passed.mean()), 4),
        "acceptance_by_set": {k: {"n": int(len(g)), "acceptance_rate": round(float(g.passed.mean()), 4)}
                              for k, g in hc.groupby("set")},
        "predicted_plane_counts": hc.pred_plane.value_counts().to_dict(),
        "predicted_subplane_counts_if_brain": hc.pred_subplane.value_counts().to_dict(),
        "median_p_tt": round(float(hc.p_tt.median()), 4),
        "training_set_by_reference_hc_mm": acceptance_by_hc(hc),
    }

    metrics = {
        "dataset": "FETAL_PLANES_DB official test split (Burgos-Artizzu et al., 2020)",
        "checkpoint_epoch": clf.meta.get("epoch"),
        "architecture": clf.meta.get("architecture"),
        "plane": plane_rep,
        "brain_subplane": sub_rep,
        "gate": gate_summary,
        "hc18_acceptance": hc18,
    }
    (OUT / "plane_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("plane", "brain_subplane")}, indent=2))
    for name, rep in (("plane", plane_rep), ("brain_subplane", sub_rep)):
        print(name, "acc", rep["accuracy"], "macro-F1", rep["macro_f1"], rep["calibration"])

    # --- figures ---
    SHOTS.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_cm(axes[0], np.array(plane_rep["confusion_matrix"]["rows_true_cols_pred"]), PLANES,
            f"Plane (official test, n={plane_rep['n']})\nacc {plane_rep['accuracy']:.3f}, "
            f"macro-F1 {plane_rep['macro_f1']:.3f}")
    plot_cm(axes[1], np.array(sub_rep["confusion_matrix"]["rows_true_cols_pred"]), SUBPLANES,
            f"Brain sub-plane (test brain images, n={sub_rep['n']})\nacc {sub_rep['accuracy']:.3f}, "
            f"macro-F1 {sub_rep['macro_f1']:.3f}")
    fig.tight_layout()
    fig.savefig(SHOTS / "plane_confusion.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    reliability(axes[0], F.softmax(lp, 1).cpu().numpy(), y, "Plane head, before scaling")
    reliability(axes[1], pp, y, f"Plane head, after (T={clf.t_plane:.2f})")
    fig.tight_layout()
    fig.savefig(SHOTS / "plane_reliability.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.8))
    bins = np.linspace(0, 1, 41)
    ax.hist(p_tt[is_tt], bins=bins, alpha=0.6, label="FETAL_PLANES test: trans-thalamic", density=True)
    ax.hist(p_tt[~is_tt], bins=bins, alpha=0.6, label="FETAL_PLANES test: other planes", density=True)
    ax.hist(hc.p_tt, bins=bins, histtype="step", lw=2, color="black", label="HC18 (all images)", density=True)
    ax.axvline(clf.threshold, color="red", ls="--", label=f"threshold {clf.threshold:.2f}")
    ax.set_yscale("log")
    ax.set_xlabel("calibrated P(trans-thalamic)")
    ax.set_ylabel("density (log)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(SHOTS / "plane_gate_scores.png", dpi=130)
    plt.close(fig)

    rejected = hc[~hc.passed].sort_values("p_tt").head(8)
    if len(rejected):
        fig, axes = plt.subplots(2, 4, figsize=(14, 6))
        for ax in axes.flat:
            ax.axis("off")
        for ax, (_, r) in zip(axes.flat, rejected.iterrows(), strict=False):
            d = HC18_DIRS[0] if r.set == "training_set" else HC18_DIRS[1]
            ax.imshow(letterbox(read_gray(d / r.image), 320), cmap="gray")
            ax.set_title(f"{r.image}\n{SHORT.get(r.pred_plane, r.pred_plane)} "
                         f"{r.plane_conf:.2f}, P(TT)={r.p_tt:.2f}", fontsize=8)
        fig.suptitle("HC18 images rejected by the plane gate (lowest P(TT) first)")
        fig.tight_layout()
        fig.savefig(SHOTS / "plane_hc18_rejected.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    main()
