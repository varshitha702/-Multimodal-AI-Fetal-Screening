"""Tests for the standard-plane quality gate (ml/planes.py, ml/train_planes.py).

No dataset or trained weights needed.
"""
import numpy as np
import pandas as pd
import pytest
import torch

from ml.planes import (
    BRAIN,
    INPUT_SIZE,
    PLANES,
    SUBPLANES,
    TT,
    PlaneClassifier,
    PlaneNet,
    expected_calibration_error,
    fit_temperature,
    letterbox,
    normalize,
    tt_score,
)
from ml.train_planes import choose_threshold, gpu_augment, load_index, make_split


def test_letterbox_keeps_aspect_and_pads_black():
    img = np.full((100, 200), 255, np.uint8)
    out = letterbox(img, 64)
    assert out.shape == (64, 64) and out.dtype == np.uint8
    assert out[0].max() == 0 and out[-1].max() == 0      # padded rows top/bottom
    assert out[32].min() == 255                          # content spans the full width
    assert (out[:, 0] > 0).sum() == 32                   # 100/200 * 64 rows of content


def test_normalize_uint8_and_float_agree():
    u8 = torch.randint(0, 256, (2, 8, 8), dtype=torch.uint8)
    a, b = normalize(u8), normalize(u8.float() / 255)
    assert a.shape == (2, 3, 8, 8)
    assert torch.allclose(a, b)


def test_fit_temperature_recovers_overconfidence():
    rng = np.random.default_rng(0)
    true_logits = torch.from_numpy(rng.normal(0, 2, (4000, 5))).float()
    labels = torch.distributions.Categorical(logits=true_logits).sample()
    t = fit_temperature(true_logits * 3, labels)        # model 3x overconfident
    assert t == pytest.approx(3, rel=0.15)


def test_ece_zero_for_calibrated_and_positive_for_overconfident():
    y = np.array([0, 0, 0, 1])
    calibrated = np.array([[0.75, 0.25]] * 4)          # 75% confident, 75% correct
    assert expected_calibration_error(calibrated, y) == pytest.approx(0, abs=1e-9)
    over = np.array([[0.99, 0.01]] * 4)
    assert expected_calibration_error(over, y) == pytest.approx(0.24, abs=1e-6)


def test_tt_score_is_product():
    pp = np.zeros(len(PLANES))
    pp[BRAIN] = 0.8
    sp = np.zeros(len(SUBPLANES))
    sp[TT] = 0.5
    assert tt_score(pp, sp) == pytest.approx(0.4)


def test_choose_threshold_meets_target_recall():
    rng = np.random.default_rng(1)
    p = rng.random(1000)
    is_tt = rng.random(1000) < 0.3
    t = choose_threshold(p, is_tt, target_recall=0.95)
    assert (p[is_tt] >= t).mean() >= 0.95
    assert (p[is_tt] >= np.nextafter(t, 1)).mean() < 0.95   # and it is the largest such threshold


def test_gpu_augment_shape_and_range():
    x = torch.randint(0, 256, (4, 32, 32), dtype=torch.uint8)
    y = gpu_augment(x)
    assert y.shape == (4, 32, 32) and y.dtype == torch.float32
    assert 0 <= y.min() and y.max() <= 1


@pytest.fixture
def fetal_planes_csv(tmp_path):
    rows = []
    planes = ["Fetal brain", "Fetal abdomen", "Other", "Maternal cervix"]
    for p in range(20):
        for k, plane in enumerate(planes):
            brain = "Trans-thalamic" if plane == "Fetal brain" else "Not A Brain"
            rows.append(f"Patient{p:05d}_Plane{k}_1_of_1;{p};{plane};{brain};Op1;Voluson E6;{int(p < 14)}")
    path = tmp_path / "data.csv"
    path.write_text("Image_name;Patient_num;Plane;Brain_plane;Operator;US_Machine;Train \n" + "\n".join(rows))
    return path


def test_load_index_labels(fetal_planes_csv):
    df = load_index(fetal_planes_csv)
    assert len(df) == 80 and "official_train" in df.columns
    brain = df[df.plane == "Fetal brain"]
    assert (brain.label == BRAIN).all() and (brain.sub_label == TT).all()
    assert (df[df.plane != "Fetal brain"].sub_label == -1).all()


def test_make_split_by_patient_keeps_official_test(fetal_planes_csv):
    df = make_split(load_index(fetal_planes_csv))
    assert (df[df.official_train == 0].split == "test").all()
    assert not (set(df[df.split == "train"].patient) & set(df[df.split == "val"].patient))
    assert df[df.split == "val"].patient.nunique() == 2      # 15% of 14 training patients
    assert make_split(load_index(fetal_planes_csv)).split.tolist() == df.split.tolist()   # deterministic


@pytest.fixture(scope="module")
def random_plane_classifier(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("planes") / "planes.pth"
    torch.save({"model_state_dict": PlaneNet().state_dict(), "epoch": 1, "input_size": INPUT_SIZE,
                "temperature_plane": 1.5, "temperature_subplane": 1.2, "tt_threshold": 0.3}, path)
    return PlaneClassifier(path, device="cpu")


def test_plane_classifier_check(random_plane_classifier):
    gray = np.random.default_rng(0).integers(0, 255, (300, 450), dtype=np.uint8)
    r = random_plane_classifier.check(gray)
    assert r.plane in PLANES
    assert sum(r.plane_probabilities.values()) == pytest.approx(1, abs=1e-4)
    assert sum(r.subplane_probabilities.values()) == pytest.approx(1, abs=1e-4)
    assert r.passed == (r.tt_probability >= 0.3)
    assert r.tt_probability == pytest.approx(r.plane_probabilities["Fetal brain"]
                                             * r.subplane_probabilities["Trans-thalamic"], rel=1e-5)
    if r.plane != "Fetal brain":
        assert r.brain_subplane is None


def test_plane_classifier_rejects_colour(random_plane_classifier):
    with pytest.raises(ValueError):
        random_plane_classifier.check(np.zeros((64, 64, 3), np.uint8))


def test_load_index_rejects_unknown_plane(tmp_path):
    path = tmp_path / "bad.csv"
    header = "Image_name;Patient_num;Plane;Brain_plane;Operator;US_Machine;Train \n"
    path.write_text(header + "x;1;Fetal heart;Not A Brain;a;b;1")
    with pytest.raises(ValueError):
        load_index(path)


def test_split_file_columns_roundtrip(fetal_planes_csv, tmp_path):
    df = make_split(load_index(fetal_planes_csv))
    p = tmp_path / "s.csv"
    df[["image", "patient", "plane", "brain_plane", "official_train", "split"]].to_csv(p, index=False)
    assert pd.read_csv(p).split.value_counts().to_dict() == df.split.value_counts().to_dict()
