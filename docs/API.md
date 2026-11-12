# FetalAI REST API (v1)

Base URL (local): `http://localhost:8000`. Interactive docs: `/docs` (Swagger UI), `/redoc`, schema at `/openapi.json`.

> Academic research prototype, not a medical device. Every prediction response includes this disclaimer.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness and readiness (`ok`, or `degraded` if the model is not loaded) |
| GET | `/api/v1/model-info` | Architecture, checkpoint, validation Dice, held-out test metrics |
| POST | `/api/v1/predict` | **Plane check**, then segment the fetal head, measure head circumference, optional GA centile |
| POST | `/api/v1/plane-check` | Only the standard-plane quality gate (which plane is this; is it trans-thalamic?) |
| GET | `/api/v1/reference/hc-centiles` | INTERGROWTH-21st HC centile curves (3rd/10th/50th/90th/97th), 14–40 weeks |

### GET /api/v1/health

```json
{ "status": "ok", "model_loaded": true, "plane_model_loaded": true, "version": "1.0.0" }
```

Always returns 200 while the process is up; `status` is `"degraded"` when the segmentation checkpoint failed to
load. `plane_model_loaded` is `false` when the plane classifier is missing; the API then still measures and
every response reports the plane check as `unavailable`.

### GET /api/v1/model-info

Returns `name`, `architecture`, `input_size`, `device`, `checkpoint_epoch`, `validation_dice`,
`test_metrics` (contents of `outputs/test_metrics.json`), `training_data`, `intended_use`, `disclaimer`, and
`plane_classifier` (architecture, epoch, temperatures, `tt_threshold`, threshold rule and official-test metrics
from `outputs/plane_metrics.json`; `null` if the plane model is not loaded).
503 if the model is not loaded.

### POST /api/v1/predict

`multipart/form-data`:

| Field | Type | Required | Rules |
|---|---|---|---|
| `file` | file | yes | PNG or JPEG, ≤ 10 MB, each side ≥ 64 px, ≤ 20 MP |
| `pixel_size_mm` | number | no | 0 < value ≤ 1.0 (mm per pixel). Without it, HC and area in mm are `null` |
| `gestational_age_weeks` | number | no | 14 ≤ value ≤ 40, exact weeks (22+3 → 22.43). With HC available, adds a `growth` block |
| `override_plane_check` | boolean | no | Default `false`. If the plane check warns, measure anyway (the warning is kept) |

**Order of operations:** validate → decode → **plane check** → U-Net → ellipse → HC → centile. If
`plane_check.status` is `"warn"` and `override_plane_check` is not set, the response has
`measurement_withheld: true` and `head_circumference_mm`, `head_area_mm2`, the ellipse sizes in mm and `growth`
are `null`; the mask and overlay are still returned so the user can see the image.

Example:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@data/raw/HC18/training_set/006_HC.png;type=image/png" \
  -F "pixel_size_mm=0.0656056482541" \
  -F "override_plane_check=true"
```

Response (200; base64 strings shortened). This is a real response for a held-out test image whose
reference HC is 67.84 mm, sent with `-F override_plane_check=true`. This early-gestation image (HC18 has many)
is **not** accepted by the plane gate (see `MODEL_CARD.md`, "HC18 acceptance"); without the override the same
request returns `measurement_withheld: true` and `head_circumference_mm: null`:

```json
{
  "image_width": 800, "image_height": 540, "pixel_size_mm": 0.0656056482541,
  "plane_check": { "status": "warn", "plane": "Fetal brain", "plane_confidence": 0.7539,
                   "brain_subplane": "Trans-cerebellum", "brain_subplane_confidence": 0.8283,
                   "tt_probability": 0.007, "threshold": 0.1379, "overridden": true,
                   "message": "This looks like a trans-cerebellum brain plane, not trans-thalamic. HC is defined on the trans-thalamic plane; please re-acquire. Measured anyway because override_plane_check was set.",
                   "plane_probabilities": { "...": 0 }, "subplane_probabilities": { "...": 0 },
                   "model_version": "planes-effnetb0-epoch16" },
  "measurement_withheld": false,
  "head_detected": true,
  "head_circumference_mm": 67.34,
  "head_area_mm2": 358.9, "head_area_px": 83387,
  "ellipse": { "center_x_px": 433.73, "center_y_px": 240.49, "semi_major_px": 175.92,
               "semi_minor_px": 150.3, "angle_deg": 71.12, "semi_major_mm": 11.54, "semi_minor_mm": 9.86 },
  "quality": { "mean_confidence": 0.9791, "uncertain_fraction": 0.0307, "mask_area_fraction": 0.193,
               "touches_border": false, "ellipse_fit_ratio": 1.004, "flags": [] },
  "mask_png_base64": "iVBORw0KGgo...", "overlay_png_base64": "iVBORw0KGgo...",
  "inference_ms": 169.2, "model_version": "hc18-unet-epoch50",
  "disclaimer": "Academic research prototype, not a medical device. ..."
}
```

* `plane_check`: `status` is `pass` (calibrated P(trans-thalamic) = P(brain) × P(TT | brain) ≥ `threshold`),
  `warn`, or `unavailable` (no plane model). `message` is plain language; `overridden` is true when the
  measurement was forced. Threshold and calibration come from the validation set (`MODEL_CARD.md`).
* `mask_png_base64`: binary mask (0/255) at the original resolution.
* `overlay_png_base64`: image with the mask (green) and fitted ellipse (yellow).
* `growth` (only with `gestational_age_weeks` and a measured HC): the HC compared with the INTERGROWTH-21st
  fetal standard (Papageorghiou et al., *Lancet* 2014;384:869-79), for example for `544_HC.png` with an
  example GA of 20.0 weeks:
  `{"gestational_age_weeks": 20.0, "z_score": -0.21, "centile": 41.8, "median_mm": 172.5, "p3_mm": 158.5,
  "p97_mm": 186.5, "within_p3_p97": true, "reference": "INTERGROWTH-21st fetal growth standard (...)"}`.
  A comparison with a population standard, not a diagnosis.
* `quality.flags`: plain-language warnings (low confidence, uncertain boundary, head touching the
  border, non-elliptical shape, no pixel spacing). These are heuristics, not calibrated uncertainty
  (see `docs/MODEL_CARD.md`).

### POST /api/v1/plane-check

`multipart/form-data` with `file` only (same rules and errors as `/predict`). Returns the `plane_check` block.
Real response for a FETAL_PLANES_DB test femur image (`Patient00647_Plane5_1_of_1.png`):

```json
{ "status": "warn", "plane": "Fetal femur", "plane_confidence": 0.9986,
  "brain_subplane": null, "brain_subplane_confidence": null, "tt_probability": 0.0, "threshold": 0.1379,
  "plane_probabilities": { "Fetal brain": 0.0003, "Fetal abdomen": 0.0, "Fetal femur": 0.9986,
                           "Fetal thorax": 0.0001, "Maternal cervix": 0.0001, "Other": 0.0009 },
  "subplane_probabilities": { "...": 0 }, "overridden": false,
  "message": "This looks like a fetal femur image (100%), not a fetal brain. HC is defined on the trans-thalamic plane; please re-acquire.",
  "model_version": "planes-effnetb0-epoch16" }
```

Latency (laptop, one 800 × 540 image): plane check 31 ms on CPU / 61 ms on the RTX 4050; a full `/predict`
round trip is about 0.3 s on the GPU (curl to `127.0.0.1`).

## Errors

All errors share one format:

```json
{ "error": { "code": "file_too_large", "message": "File exceeds the 10 MB limit" } }
```

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_image` | Empty, truncated or corrupted file (`PIL.Image.verify`), undecodable, smaller than 64 px, larger than 20 MP |
| 413 | `file_too_large` | Upload larger than `FETALAI_MAX_UPLOAD_MB` (default 10). Read in 1 MB chunks and rejected as soon as the limit is passed |
| 415 | `unsupported_media_type` | Declared type is not `image/png` / `image/jpeg`, or the decoded format is not PNG/JPEG (e.g. a GIF renamed `.png`) |
| 422 | `validation_error` | Missing `file`, `pixel_size_mm` not a number in (0, 1], `gestational_age_weeks` not in [14, 40], or `override_plane_check` not a boolean. `details` lists the field and reason |
| 503 | `model_not_loaded` | Checkpoint missing or failed to load at startup |

## Configuration

Environment variables (or `.env`, see `.env.example`):

| Variable | Default |
|---|---|
| `FETALAI_CHECKPOINT_PATH` | `outputs/hc18_unet_best.pth` |
| `FETALAI_PLANE_CHECKPOINT_PATH` | `outputs/plane_classifier_best.pth` (optional; gate `unavailable` without it) |
| `FETALAI_DEVICE` | automatic (`cuda` if available) |
| `FETALAI_MAX_UPLOAD_MB` | `10` |
| `FETALAI_CORS_ORIGINS` | `["http://localhost:3000","http://127.0.0.1:3000"]` |
| `FETALAI_LOG_LEVEL` | `INFO` |

Logs are JSON lines (one per request with `request_id`, method, path, status, ms; one per prediction).
Clients may send `X-Request-ID`; it is echoed in the response.
