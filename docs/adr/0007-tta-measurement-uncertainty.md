# ADR-0007: Measurement uncertainty by test-time augmentation

* Status: accepted (12 Nov 2026)
* Module: ML (Nachiketha K G), API (Vishwas K), UI (Rakshitha V)

## Context

Phase 1 reported one HC value with heuristic quality flags that correlated with the HC error at only
Spearman ≈ 0.1. A clinician needs to know when a measurement is unreliable and should be re-scanned. The method
had to be cheap (no retraining) and its value had to be shown on held-out data, even if weak.

## Decision

* **Test-time augmentation** with the existing U-Net: 8 passes (identity, flip, zoom 0.9/1.1, gamma 0.8/1.25,
  flip + zoom), each geometric transform undone on the probability map before combining. Chosen over MC dropout
  because the U-Net has no dropout layers and retraining was out of the time box.
* Outputs: HC ± SD (SD over the 8 per-pass HCs), CV, per-pixel SD map, mask disagreement fraction.
* **Flag rule fixed before running:** "low reliability, re-scan" if CV or disagreement is above its
  **validation** 90th percentile (1.05 %, 0.100). **Primary HC** = single pass or TTA mean, whichever had the
  lower validation MAE (single pass, 1.843 vs 1.861 mm).
* API: `uncertainty` on by default, `uncertainty=false` for a fast single pass; the flag is also the first
  quality warning. UI: HC ± SD, a reliability card, and a segmentation / uncertainty-map toggle.
* CPU: passes run one at a time (a batch of 8 was ~5× slower per image on this CPU); GPU: one batch.

## Consequences

* Held-out test: Spearman ρ(SD, |error|) = 0.27 (95 % CI 0.11–0.41); flagged images (11 %) have MAE 3.68 mm vs
  1.99 mm; 6 of the 16 largest errors are flagged. A real but modest signal, documented as such.
* TTA does not make HC more accurate (MAE 2.19 vs 2.18 mm), so it is used only for uncertainty.
* About +0.8 s per request on the Docker CPU backend (0.23 → 1.0 s round trip). Encoding PNGs without
  `optimize` (about 7× faster) was done at the same time.
