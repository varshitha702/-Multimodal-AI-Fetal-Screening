# ADR-0001: Fill HC18 annotations at original resolution, then resize

* Status: accepted (12 Nov 2026)
* Module: dataset and preprocessing

## Context

HC18 annotations are 1–2 px ellipse outlines. The original trainer resized them to 128 × 128 with nearest
neighbour and then filled the largest contour. Every mask came out empty (999/999); the model learned to
predict nothing and an epsilon-smoothed Dice reported 1.0.

## Decision

Fit an ellipse (`cv2.fitEllipse`) to all outline pixels at the original resolution, draw it filled, and only
then resize the filled mask (area interpolation, threshold 0.5). Hard Dice/IoU return NaN, not 1.0, for empty
targets, and the dataset raises an error if a mask is empty.

## Consequences

* 0/999 empty masks; mean foreground 30 %.
* Ellipse fitting also handles outlines that touch the image border.
* Tests U01–U06 guard against regressions, including one that reproduces the old bug.
