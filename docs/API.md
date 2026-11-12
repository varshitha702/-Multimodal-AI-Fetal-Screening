# FetalAI REST API (v1)

Base URL (local): `http://localhost:8000`. Interactive docs: `/docs` (Swagger UI), `/redoc`, schema at `/openapi.json`.

> Academic research prototype, not a medical device. Every prediction response includes this disclaimer.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness and readiness (`ok`, or `degraded` if the model is not loaded) |
| GET | `/api/v1/model-info` | Architecture, checkpoint, validation Dice, held-out test metrics |
| POST | `/api/v1/predict` | Segment the fetal head and measure head circumference |

### GET /api/v1/health

```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0" }
```

Always returns 200 while the process is up; `status` is `"degraded"` when the checkpoint failed to load.

### GET /api/v1/model-info

Returns `name`, `architecture`, `input_size`, `device`, `checkpoint_epoch`, `validation_dice`,
`test_metrics` (contents of `outputs/test_metrics.json`), `training_data`, `intended_use`, `disclaimer`.
503 if the model is not loaded.

### POST /api/v1/predict

`multipart/form-data`:

| Field | Type | Required | Rules |
|---|---|---|---|
| `file` | file | yes | PNG or JPEG, ≤ 10 MB, each side ≥ 64 px, ≤ 20 MP |
| `pixel_size_mm` | number | no | 0 < value ≤ 1.0 (mm per pixel). Without it, HC and area in mm are `null` |

Example:

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -F "file=@data/raw/HC18/training_set/006_HC.png;type=image/png" \
  -F "pixel_size_mm=0.0656056482541"
```

Response (200; base64 strings shortened). This is a real response for a held-out test image whose
reference HC is 67.84 mm:

```json
{
  "image_width": 800, "image_height": 540, "pixel_size_mm": 0.0656056482541,
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

* `mask_png_base64`: binary mask (0/255) at the original resolution.
* `overlay_png_base64`: image with the mask (green) and fitted ellipse (yellow).
* `quality.flags`: plain-language warnings (low confidence, uncertain boundary, head touching the
  border, non-elliptical shape, no pixel spacing). These are heuristics, not calibrated uncertainty
  (see `docs/MODEL_CARD.md`).

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
| 422 | `validation_error` | Missing `file`, or `pixel_size_mm` not a number in (0, 1]. `details` lists the field and reason |
| 503 | `model_not_loaded` | Checkpoint missing or failed to load at startup |

## Configuration

Environment variables (or `.env`, see `.env.example`):

| Variable | Default |
|---|---|
| `FETALAI_CHECKPOINT_PATH` | `outputs/hc18_unet_best.pth` |
| `FETALAI_DEVICE` | automatic (`cuda` if available) |
| `FETALAI_MAX_UPLOAD_MB` | `10` |
| `FETALAI_CORS_ORIGINS` | `["http://localhost:3000","http://127.0.0.1:3000"]` |
| `FETALAI_LOG_LEVEL` | `INFO` |

Logs are JSON lines (one per request with `request_id`, method, path, status, ms; one per prediction).
Clients may send `X-Request-ID`; it is echoed in the response.
