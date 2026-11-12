# ADR-0002: Split HC18 by examination group, not by image

* Status: accepted (12 Nov 2026)
* Module: dataset and preprocessing / evaluation

## Context

HC18 has no patient IDs. Filenames that share a numeric prefix (`014_HC`, `014_2HC`, `014_3HC`) are repeat
scans of the same fetus with nearly identical reference HC; one pair is even byte-identical. A random image-level
split would put near-duplicates in both training and test and inflate the test metrics.

## Decision

Split the 806 prefix groups 70/15/15 with seed 42, stratified by mean HC quintile. Commit the resulting CSV
(`data/splits/hc18_split.csv`). Use validation for early stopping and checkpoint selection only, and the
test split only once, for the final metrics.

## Consequences

* No fetus (as far as filenames reveal) appears in two splits; a test checks this.
* The split is documented as group-level, not verified patient-level.
* The official HC18 test set (no labels) is used only for qualitative demos.
