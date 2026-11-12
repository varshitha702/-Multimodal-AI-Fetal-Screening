# Evidence: HC18 empty-mask bug (before fix)

The two `*_INVALID_before_fix.*` files are the output of the original `train_hc18.py`
(2 CPU epochs, 128x128). They report **validation Dice 1.0 with loss ~1.0. This number is
invalid and must not be reported.**

**Root cause:** the trainer resized the 1-pixel ellipse *outline* annotation to 128x128
with NEAREST interpolation *before* filling it. The outline broke into fragments,
`findContours` + `FILLED` produced an empty mask for **999/999** images
(`outputs/mask_pipeline_check.json`), the model learned to predict nothing, and the Dice
metric `(2I + eps) / (P + T + eps)` scored empty-vs-empty as 1.0.

**Fix:** `ml/masks.py` fits an ellipse to the outline at the original resolution, draws
it filled, and only then resizes (0/999 empty masks after the fix). `ml/metrics.py`
returns NaN for empty targets instead of 1.0. Tests: `tests/test_masks.py`,
`tests/test_metrics.py`. Figures: `docs/screenshots/mask_bug_before_after.png`,
`docs/screenshots/mask_sanity_grid.png`.
