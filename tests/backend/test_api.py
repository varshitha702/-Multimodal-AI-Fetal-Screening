"""Backend API tests (numbered T1..T20, referenced in docs/TESTING.md).

A randomly initialised U-Net checkpoint is written to a temp dir so these tests do not need
the trained weights (they test the API contract, not segmentation accuracy).
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.core.config import Settings
from backend.app.main import create_app
from ml.model import UNet


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("ckpt") / "test_unet.pth"
    torch.save({"model_state_dict": UNet().state_dict(), "image_size": 256, "epoch": 1,
                "best_val_dice": 0.5, "architecture": "UNet(base=32, in=1, out=1)"}, path)
    return path


def make_client(checkpoint_path, **overrides) -> TestClient:
    settings = Settings(checkpoint_path=checkpoint_path, device="cpu", **overrides)
    return TestClient(create_app(settings))


@pytest.fixture(scope="module")
def client(checkpoint):
    with make_client(checkpoint) as c:
        yield c


def png_bytes(w=320, h=240, mode="L", fmt="PNG") -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, (h, w) if mode == "L" else (h, w, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def post(client, data: bytes, content_type="image/png", name="scan.png", **form):
    return client.post("/api/v1/predict", files={"file": (name, data, content_type)},
                       data={k: str(v) for k, v in form.items()})


# --- system endpoints ---------------------------------------------------------------------------

def test_T01_health_ok_when_model_loaded(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True, "version": "1.0.0"}


def test_T02_model_info(client):
    r = client.get("/api/v1/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["input_size"] == 256 and body["checkpoint_epoch"] == 1
    assert "not a medical device" in body["disclaimer"]


def test_T03_openapi_docs_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/api/v1/health", "/api/v1/model-info", "/api/v1/predict"} <= set(paths)


# --- /predict: normal cases -----------------------------------------------------------------------

def test_T04_predict_valid_png_with_pixel_size(client):
    r = post(client, png_bytes(), pixel_size_mm=0.1)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["image_width"] == 320 and body["image_height"] == 240
    assert body["pixel_size_mm"] == 0.1
    for key in ("mask_png_base64", "overlay_png_base64"):
        img = Image.open(io.BytesIO(base64.b64decode(body[key])))
        assert img.size == (320, 240)
    assert set(body["quality"]) >= {"mean_confidence", "uncertain_fraction", "flags", "touches_border"}
    assert "not a medical device" in body["disclaimer"]


def test_T05_predict_valid_jpeg_rgb(client):
    r = post(client, png_bytes(mode="RGB", fmt="JPEG"), content_type="image/jpeg", name="scan.jpg")
    assert r.status_code == 200, r.text


def test_T06_predict_without_pixel_size_reports_no_hc(client):
    body = post(client, png_bytes()).json()
    assert body["head_circumference_mm"] is None and body["head_area_mm2"] is None
    assert any("pixel spacing" in f for f in body["quality"]["flags"])


def test_T07_hc_is_consistent_with_ellipse_and_pixel_size(client):
    body = post(client, png_bytes(), pixel_size_mm=0.2).json()
    if body["ellipse"] is None:
        pytest.skip("random weights produced no ellipse")
    e = body["ellipse"]
    assert e["semi_major_mm"] == pytest.approx(e["semi_major_px"] * 0.2, abs=0.01)
    assert body["head_circumference_mm"] > 0


# --- /predict: boundary cases -------------------------------------------------------------------

def test_T08_minimum_size_image_accepted(client):
    assert post(client, png_bytes(64, 64)).status_code == 200


def test_T09_tiny_image_rejected_400(client):
    r = post(client, png_bytes(32, 32))
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_image"


def test_T10_file_just_under_limit_accepted(checkpoint):
    data = png_bytes()
    with make_client(checkpoint, max_upload_mb=len(data) / (1024 * 1024) + 0.001) as c:
        assert post(c, data).status_code == 200


def test_T11_file_over_limit_rejected_413(checkpoint):
    data = png_bytes()
    with make_client(checkpoint, max_upload_mb=(len(data) - 1) / (1024 * 1024)) as c:
        r = post(c, data)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "file_too_large"


def test_T12_huge_11mb_file_rejected_413(client):
    r = post(client, b"\x89PNG\r\n\x1a\n" + b"\0" * (11 * 1024 * 1024))
    assert r.status_code == 413


# --- /predict: negative cases -------------------------------------------------------------------

def test_T13_corrupted_png_rejected_400(client):
    data = png_bytes()
    r = post(client, data[: len(data) // 2])  # truncated file
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_image"


def test_T14_random_bytes_with_png_type_rejected_400(client):
    assert post(client, b"this is not an image at all").status_code == 400


def test_T15_empty_file_rejected_400(client):
    assert post(client, b"").status_code == 400


def test_T16_wrong_content_type_rejected_415(client):
    r = post(client, b"%PDF-1.4 ...", content_type="application/pdf", name="report.pdf")
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "unsupported_media_type"


def test_T17_real_gif_disguised_as_png_rejected_415(client):
    buf = io.BytesIO()
    Image.new("L", (100, 100)).save(buf, format="GIF")
    assert post(client, buf.getvalue()).status_code == 415


@pytest.mark.parametrize("value", ["-0.1", "0", "5", "abc"])
def test_T18_invalid_pixel_size_rejected_422(client, value):
    r = client.post("/api/v1/predict", files={"file": ("a.png", png_bytes(), "image/png")},
                    data={"pixel_size_mm": value})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_error"


def test_T19_missing_file_rejected_422(client):
    assert client.post("/api/v1/predict", data={"pixel_size_mm": "0.1"}).status_code == 422


# --- missing model ------------------------------------------------------------------------------

def test_T20_missing_checkpoint_gives_503_and_degraded_health(tmp_path):
    with make_client(tmp_path / "does_not_exist.pth") as c:
        health = c.get("/api/v1/health").json()
        assert health["status"] == "degraded" and health["model_loaded"] is False
        r = post(c, png_bytes())
        assert r.status_code == 503
        assert r.json()["error"]["code"] == "model_not_loaded"
        assert c.get("/api/v1/model-info").status_code == 503


# --- gestational age / INTERGROWTH-21st centile ----------------------------------------------------

def test_T21_ga_with_pixel_size_returns_growth_assessment(client):
    body = post(client, png_bytes(), pixel_size_mm=0.2, gestational_age_weeks=22.0).json()
    if body["head_circumference_mm"] is None:
        pytest.skip("random weights produced no ellipse")
    g = body["growth"]
    assert g["gestational_age_weeks"] == 22.0
    assert g["median_mm"] == 196.3  # published INTERGROWTH-21st median at 22 weeks
    assert 0 <= g["centile"] <= 100 and "INTERGROWTH-21st" in g["reference"]


def test_T22_no_ga_means_no_growth_block(client):
    assert post(client, png_bytes(), pixel_size_mm=0.2).json()["growth"] is None


def test_T23_ga_without_pixel_size_gives_flag_not_centile(client):
    body = post(client, png_bytes(), gestational_age_weeks=30).json()
    assert body["growth"] is None
    assert any("centile needs HC" in f for f in body["quality"]["flags"])


@pytest.mark.parametrize("value", ["13.9", "40.5", "-1", "abc"])
def test_T24_ga_outside_standard_range_rejected_422(client, value):
    r = client.post("/api/v1/predict", files={"file": ("a.png", png_bytes(), "image/png")},
                    data={"gestational_age_weeks": value})
    assert r.status_code == 422


def test_T25_centile_curves_endpoint(client):
    body = client.get("/api/v1/reference/hc-centiles").json()
    pts = body["points"]
    assert pts[0]["ga_weeks"] == 14 and pts[-1]["ga_weeks"] == 40 and len(pts) == 53
    assert round(pts[0]["p50"], 1) == 97.9  # published median at 14 weeks
