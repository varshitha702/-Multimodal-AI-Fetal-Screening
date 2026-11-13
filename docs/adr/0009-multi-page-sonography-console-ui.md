# ADR-0009: Multi-page "Sonography Console" UI

* Status: accepted (13 Nov 2026)
* Module: frontend (Rakshitha V), tests (Yashwanth Gowda M N)

## Context

After features A–C, everything lived on one long page: upload, results, CTG form and summary stacked in
generic cards. It worked but did not explain the project, and the extra outputs (plane gate, uncertainty, CTG,
fusion) had no room to be understood. The review audience is technical and judges the project partly on how
clearly it is presented.

## Decision

* **Separate pages by job:** `/` explains the project (landing); `/scan`, `/ctg`, `/report` are the workspace;
  `/evidence` holds the metrics and figures. `/model` redirects to `/evidence`.
* **One shared session** (React context, in memory only) carries the scan, CTG result and combined summary
  across pages; a header tray shows what has been analysed. Nothing is stored in the browser or on the server.
* **A visual language taken from the domain:** an ultrasound console (ink-black, teal phosphor, sector fans,
  calliper overlays, monospace readouts) by default, and a light "report paper" theme. Each page has its own
  idea: a monitor viewport with a stage rail (scan), physiological channels with a probability gauge and SHAP
  chart (CTG), a lit rule ladder (report), figure "printouts" (evidence).
* **No new npm dependencies** (installs are very slow on the team's network): effects are canvas, SVG and CSS
  keyframes; `prefers-reduced-motion` disables them.
* **Truth in the showcase:** the landing hero replays this system's real output for a named held-out image
  and says so; every number on the landing page comes from the evaluation outputs; the limits have their own
  section.

## Consequences

* The end-to-end check navigates through the header links, so it now also tests the cross-page session.
* More frontend code to maintain (five pages), but the API client, upload checks, pixel-size lookup, centile
  chart and print report are reused.
* Test IDs used by `scripts/ui_e2e.py` (`plane-check`, `metric-*`, `screening-summary`, `ctg-result`, …) are
  part of the UI contract.
