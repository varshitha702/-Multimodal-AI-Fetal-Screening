# ADR-0006: Standard-plane quality gate before HC measurement

* Status: accepted (12 Nov 2026)
* Module: ML (Nachiketha K G), data (Pujari Varshitha), API (Vishwas K), UI (Rakshitha V)

## Context

HC is clinically defined on the trans-thalamic (TT) plane. The HC18 U-Net measures whatever it is given: a
femur or abdomen image still produces an "HC". Commercial machines already automate head biometry, so a check
of *whether the image is the right plane* is the part of FetalAI that differs from them, and it is the core of
the project's patent disclosure.

## Decision

* **Data:** FETAL_PLANES_DB (Burgos-Artizzu et al., *Sci Rep* 2020). Keep the official test set untouched;
  split the official training set into train/val **by `Patient_num`** (committed CSV).
* **Model:** one ImageNet EfficientNet-B0 backbone with **two heads**: plane (6 classes) and brain sub-plane
  (TT / TC / TV / Other). The sub-plane loss is applied to brain images only. We added "Other" to the three
  standard brain planes so that non-standard brain views have a class of their own rather than being forced
  into TT/TC/TV.
* **Calibration:** temperature scaling per head, fitted on validation.
* **Gate score:** calibrated P(brain) × P(TT | brain). **Threshold rule fixed in advance:** the largest
  threshold that keeps validation TT recall ≥ 95 % (we prefer to rarely block a good image). Result: 0.138.
* **Model selection:** 224 px and 320 px inputs were trained; 320 px was kept because its *validation* score
  was higher (0.869 vs 0.860). Test numbers for both are reported.
* **API behaviour:** `/predict` runs the gate first. On `warn`, HC, area, mm sizes and centile are withheld
  (`measurement_withheld: true`) with a plain-language message ("HC is defined on the trans-thalamic plane;
  please re-acquire"). `override_plane_check=true` measures anyway and keeps the warning, so HC18 demos
  still work. If the plane model is missing, the gate reports `unavailable` and does not block.
  `/plane-check` exposes the gate alone.
* **UI:** a PASS / WARN / N/A badge with plane, calibrated confidence and P(TT) vs threshold, shown above the
  HC result, with a "Measure anyway (override)" button; the plane check is printed on the report.

## Consequences

* Official test: plane accuracy 94.0 % (macro-F1 0.934), sub-plane accuracy 82.7 % (macro-F1 0.721); ECE
  0.035 → 0.014 (plane) and 0.120 → 0.033 (sub-plane). The gate accepts 95.7 % of TT images and rejects 99.8 %
  of non-brain images, but only 42–78 % of the neighbouring brain planes (Other 42 %, TV 48 %, TC 78 %).
* **Only 60.3 % of HC18 images pass** (31.5 % below HC 100 mm), because of the 18–40 week range of the
  training data, scanner/site shift and the TT/TV ambiguity. We report this rather than lowering the
  threshold on HC18. The override keeps demos usable; external validation on local data is Phase 2.
* Adds about 31 ms (CPU) / 61 ms (GPU) per request and a second checkpoint to deploy (Docker mounts both).
