"""Train the standard-plane classifier (Feature A) on FETAL_PLANES_DB.

    python -m ml.train_planes [--size 224] [--epochs 25] [--batch-size 64] [--lr 3e-4]

Data: data/raw/FETAL_PLANES_DB/FETAL_PLANES_DB_data.csv + Images/ (Burgos-Artizzu et al., 2020).
  * test   = the official ``Train == 0`` rows (never used for training, model selection or calibration)
  * train / validation = the official ``Train == 1`` rows split by Patient_num (no patient in both),
    committed at data/splits/fetal_planes_split.csv
Model: ImageNet EfficientNet-B0 + two heads (ml/planes.py), AdamW, cosine LR, AMP, GPU augmentation,
early stopping on validation score = mean(plane macro-F1, brain sub-plane macro-F1).
After training, on the VALIDATION set only:
  * one temperature per head (temperature scaling),
  * the gate threshold: the largest t with validation trans-thalamic recall >= 0.95
    (rule fixed before looking at test results).
Writes outputs/plane_classifier_best.pth (gitignored), plane_training_history.csv, plane_training_summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from ml.planes import (
    BRAIN,
    INPUT_SIZE,
    PLANES,
    SUBPLANES,
    TT,
    PlaneNet,
    fit_temperature,
    letterbox,
    normalize,
    tt_score,
)
from ml.train import set_seed

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "FETAL_PLANES_DB"
CSV_PATH = DATA_DIR / "FETAL_PLANES_DB_data.csv"
IMAGE_DIR = DATA_DIR / "Images"
SPLIT_PATH = ROOT / "data" / "splits" / "fetal_planes_split.csv"
CACHE_DIR = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
CHECKPOINT_PATH = OUT / "plane_classifier_best.pth"
VAL_PATIENT_FRACTION = 0.15
TARGET_TT_RECALL = 0.95
SEED = 42


# --- data --------------------------------------------------------------------------------------

def load_index(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """Official CSV -> one row per image with integer labels. sub_label = -1 for non-brain images."""
    df = pd.read_csv(csv_path, sep=";")
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Image_name": "image", "Patient_num": "patient", "Plane": "plane",
                            "Brain_plane": "brain_plane", "Train": "official_train"})
    for c in ("image", "plane", "brain_plane"):
        df[c] = df[c].astype(str).str.strip()
    unknown = set(df.plane) - set(PLANES)
    if unknown:
        raise ValueError(f"Unknown plane labels: {unknown}")
    df["label"] = df.plane.map(PLANES.index)
    df["sub_label"] = df.brain_plane.map(lambda b: SUBPLANES.index(b) if b in SUBPLANES else -1)
    df.loc[df.label != BRAIN, "sub_label"] = -1
    return df


def make_split(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Official test stays test; official train is split into train/val by patient (no leakage)."""
    rng = np.random.default_rng(seed)
    train_patients = np.array(sorted(df.loc[df.official_train == 1, "patient"].unique()))
    rng.shuffle(train_patients)
    n_val = round(len(train_patients) * VAL_PATIENT_FRACTION)
    val_patients = set(train_patients[:n_val].tolist())
    split = np.where(df.official_train == 0, "test", np.where(df.patient.isin(val_patients), "val", "train"))
    return df.assign(split=split)


def load_split() -> pd.DataFrame:
    if SPLIT_PATH.is_file():
        split = pd.read_csv(SPLIT_PATH)
        return load_index().merge(split[["image", "split"]], on="image", validate="one_to_one")
    df = make_split(load_index())
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df[["image", "patient", "plane", "brain_plane", "official_train", "split"]].to_csv(SPLIT_PATH, index=False)
    return df


def read_gray(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.uint8)


def load_images(df: pd.DataFrame, size: int = INPUT_SIZE) -> np.ndarray:
    """All images letterboxed to ``size``, cached as one uint8 array (row order = CSV order)."""
    cache = CACHE_DIR / f"fetal_planes_{size}.npy"
    if cache.is_file():
        arr = np.load(cache)
        if len(arr) == len(df):
            return arr
    arr = np.stack([letterbox(read_gray(IMAGE_DIR / f"{name}.png"), size) for name in df.image])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


# --- augmentation (on the GPU, whole batch at once) ----------------------------------------------

def gpu_augment(x: torch.Tensor) -> torch.Tensor:
    """(N, H, W) uint8 -> (N, H, W) float in [0, 1]: flip, rotation, scale, shift, brightness/contrast/gamma."""
    n = x.shape[0]
    dev = x.device
    x = x.float().div(255.0).unsqueeze(1)
    flip = torch.rand(n, device=dev) < 0.5
    x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    ang = torch.empty(n, device=dev).uniform_(-12, 12) * math.pi / 180
    scale = torch.empty(n, device=dev).uniform_(0.85, 1.15)
    tx, ty = (torch.empty(n, device=dev).uniform_(-0.06, 0.06) for _ in range(2))
    cos, sin = torch.cos(ang) / scale, torch.sin(ang) / scale
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    contrast = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
    bright = torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.08, 0.08)
    gamma = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.25)
    x = ((x * contrast + bright).clamp(0, 1) ** gamma).clamp(0, 1)
    return x[:, 0]


# --- metrics -------------------------------------------------------------------------------------

def confusion(y: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, p), 1)
    return cm


def per_class(cm: np.ndarray, names: list[str]) -> dict:
    out = {}
    for i, n in enumerate(names):
        tp = cm[i, i]
        prec = tp / cm[:, i].sum() if cm[:, i].sum() else 0.0
        rec = tp / cm[i].sum() if cm[i].sum() else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[n] = {"precision": round(float(prec), 4), "recall": round(float(rec), 4),
                  "f1": round(float(f1), 4), "support": int(cm[i].sum())}
    return out


def macro_f1(y: np.ndarray, p: np.ndarray, k: int) -> float:
    return float(np.mean([v["f1"] for v in per_class(confusion(y, p, k), [str(i) for i in range(k)]).values()]))


@torch.no_grad()
def predict_logits(model, images: torch.Tensor, batch_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits for uint8 images (N, H, W) already on the model's device."""
    model.eval()
    lps, lss = [], []
    for i in range(0, len(images), batch_size):
        with torch.autocast(images.device.type, enabled=images.device.type == "cuda"):
            lp, ls = model(normalize(images[i:i + batch_size]))
        lps.append(lp.float())
        lss.append(ls.float())
    return torch.cat(lps), torch.cat(lss)


def score(lp: torch.Tensor, ls: torch.Tensor, y: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
    """(plane macro-F1, brain sub-plane macro-F1 on brain images with a sub-plane label)."""
    plane_f1 = macro_f1(y, lp.argmax(1).cpu().numpy(), len(PLANES))
    m = ys >= 0
    sub_f1 = macro_f1(ys[m], ls[torch.from_numpy(m).to(ls.device)].argmax(1).cpu().numpy(), len(SUBPLANES))
    return plane_f1, sub_f1


def choose_threshold(p_tt: np.ndarray, is_tt: np.ndarray, target_recall: float = TARGET_TT_RECALL) -> float:
    """Largest threshold t such that recall of true trans-thalamic images (p_tt >= t) is >= target_recall."""
    s = np.sort(p_tt[is_tt])
    k = int(np.floor((1 - target_recall) * len(s)))  # we may reject at most k true TT images
    return float(s[k])


# --- training ------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=INPUT_SIZE, help="square input size in pixels")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    args = ap.parse_args()

    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_split()
    images = load_images(df, args.size)
    counts = df.groupby("split").agg(images=("image", "size"), patients=("patient", "nunique"))
    print(counts.to_string(), flush=True)
    assert not (set(df[df.split == "train"].patient) & set(df[df.split == "val"].patient)), "patient leakage"

    def subset(name):
        idx = np.flatnonzero(df.split.to_numpy() == name)
        return (torch.from_numpy(images[idx]).to(device), df.label.to_numpy()[idx], df.sub_label.to_numpy()[idx])

    xtr, ytr, ystr = subset("train")
    xva, yva, ysva = subset("val")
    ytr_t, ystr_t = torch.from_numpy(ytr).to(device), torch.from_numpy(ystr).to(device)

    model = PlaneNet(pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = math.ceil(len(xtr) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    history, best, bad = [], -1.0, 0
    OUT.mkdir(exist_ok=True)
    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr), device=device)
        total, n = 0.0, 0
        t0 = time.time()
        for i in range(0, len(perm), args.batch_size):
            b = perm[i:i + args.batch_size]
            x = normalize(gpu_augment(xtr[b]))
            y, ys = ytr_t[b], ystr_t[b]
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                lp, ls = model(x)
                loss = F.cross_entropy(lp.float(), y)
                m = ys >= 0
                if m.any():
                    loss = loss + F.cross_entropy(ls.float()[m], ys[m])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * len(b)
            n += len(b)
        lp, ls = predict_logits(model, xva)
        plane_f1, sub_f1 = score(lp, ls, yva, ysva)
        val_loss = F.cross_entropy(lp, torch.from_numpy(yva).to(device)).item()
        val_score = (plane_f1 + sub_f1) / 2
        history.append({"epoch": epoch, "train_loss": total / n, "val_plane_loss": val_loss,
                        "val_plane_macro_f1": plane_f1, "val_subplane_macro_f1": sub_f1, "val_score": val_score,
                        "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0})
        print(f"epoch {epoch:2d} loss {total / n:.4f} val plane F1 {plane_f1:.4f} sub F1 {sub_f1:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if val_score > best:
            best, bad = val_score, 0
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch}, CHECKPOINT_PATH)
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping", flush=True)
                break
    pd.DataFrame(history).to_csv(OUT / "plane_training_history.csv", index=False)

    # --- calibration + gate threshold, validation set only ---
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    lp, ls = predict_logits(model, xva)
    m = ysva >= 0
    t_plane = fit_temperature(lp, torch.from_numpy(yva))
    t_sub = fit_temperature(ls[torch.from_numpy(m).to(device)], torch.from_numpy(ysva[m]))
    pp = F.softmax(lp / t_plane, 1).cpu().numpy()
    sp = F.softmax(ls / t_sub, 1).cpu().numpy()
    is_tt = (yva == BRAIN) & (ysva == TT)
    threshold = choose_threshold(tt_score(pp, sp), is_tt)
    plane_f1, sub_f1 = score(lp, ls, yva, ysva)
    ckpt.update({
        "architecture": "EfficientNet-B0 (ImageNet) + plane head (6) + brain sub-plane head (4)",
        "input_size": args.size, "planes": PLANES, "subplanes": SUBPLANES,
        "temperature_plane": t_plane, "temperature_subplane": t_sub, "tt_threshold": threshold,
        "val_plane_macro_f1": plane_f1, "val_subplane_macro_f1": sub_f1,
        "threshold_rule": f"largest t with validation trans-thalamic recall >= {TARGET_TT_RECALL}",
    })
    torch.save(ckpt, CHECKPOINT_PATH)
    summary = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    summary.update({"split_counts": counts.reset_index().to_dict(orient="records"),
                    "epochs_run": len(history), "minutes": round((time.time() - t_start) / 60, 1),
                    "args": vars(args), "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"})
    (OUT / "plane_training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
