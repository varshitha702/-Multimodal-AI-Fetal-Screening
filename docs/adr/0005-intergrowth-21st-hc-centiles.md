# ADR-0005: HC-for-gestational-age centiles from INTERGROWTH-21st, verified against the published table

* Status: accepted (12 Nov 2026)
* Module: backend and integration (clinical-context input)

## Context

The project brief allows an image + clinical-input feature only if the formulas come from a cited source and
are unit-tested against that source's published values. Inventing or misremembering a medical formula is not
acceptable.

## Decision

* Use the INTERGROWTH-21st international fetal growth standard for HC (Papageorghiou et al., *Lancet*
  2014;384:869-79). The mean and SD equations are copied from **Table 2 (p. 876)** of the paper; centiles are
  mean + Z × SD with Z = ±1.88 for the 3rd/97th centiles as stated in the paper.
* Verify the implementation against **all 189 cells** of the official INTERGROWTH-21st "Head Circumference
  (mm)" z-score table (University of Oxford), stored as a test fixture. Result: every cell within 0.06 mm.
* Only HC-for-GA (measurement → centile) is implemented. Estimating GA from HC is a different equation and is
  left for Phase 2.
* Accept GA only in the standard's range (14–40 weeks); report centile, z-score, median and 3rd–97th range,
  worded as "comparison with a population standard, not a diagnosis".

## Consequences

* The centile is only as good as the measured HC (test MAE 2.18 mm) and the user-entered GA. HC18 contains no
  GA, so demos use example GA values.
* The curves served to the frontend come from the same Python module, so there is one source of truth.
