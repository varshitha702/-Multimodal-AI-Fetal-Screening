# ADR-0003: Measure HC as the perimeter of an ellipse fitted at original resolution

* Status: accepted (12 Nov 2026)
* Module: model evaluation / backend

## Context

The network works on 256 × 256 inputs, but HC18 images are about 800 × 540, so the resize scales x and y
differently. Pixel spacing (mm/px) is only isotropic at the original resolution. A raw pixel-contour length
is jagged and overestimates the perimeter.

## Decision

Resize the predicted mask back to the original image size, keep the largest connected component, fit an
ellipse to its contour, and compute the perimeter with Ramanujan's second approximation. HC (mm) = perimeter
(px) × pixel spacing. If no spacing is supplied, HC is not reported.

## Consequences

* Applying the same geometry to the reference outlines reproduces the CSV HC with an MAE of 0.009 mm, so the
  measured error reflects the segmentation, not the measurement code.
* The same code (`ml/geometry.py` via `ml/inference.py`) is used by evaluation, the API and the dashboard.
