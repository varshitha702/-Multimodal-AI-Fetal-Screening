# frontend/: Next.js web client

Owner: **Rakshitha V** (frontend).

Next.js 15 (App Router) + TypeScript + Tailwind CSS + shadcn/ui components (`src/components/ui`),
talking to the FastAPI backend.

## Run

```powershell
# terminal 1, from the repository root: start the backend on :8000
scripts\run_backend.bat

# terminal 2
cd frontend
npm install
npm run dev            # http://localhost:3000
```

The API URL defaults to `http://localhost:8000`; override it with `NEXT_PUBLIC_API_URL` (see `.env.example`).

## Design: "Sonography Console"

The UI borrows the visual language of an ultrasound machine: an ink-black console (default) with teal phosphor
echoes, sector-fan motifs, calliper overlays and monospace readouts; a light "report paper" theme is one click
away. Fonts: Inter, Space Grotesk (display), JetBrains Mono (readouts), loaded with `next/font`. All effects are
hand-written (canvas, SVG, CSS keyframes in `globals.css`): no animation library, and `prefers-reduced-motion`
turns them off.

## Pages

| Route | Page | What is special |
|---|---|---|
| `/` | Overview (landing) | Live-scan hero: canvas speckle in a sector, sweeping beam, traced calliper ellipse and HUD readouts replaying this system's real output for `544_HC.png`; auto-advancing five-stage pipeline; count-up held-out metrics; "what we do not claim"; team |
| `/scan` | Ultrasound workstation | Probe input · monitor viewport (corner brackets, HUD, tabs: segmentation / uncertainty map / original / mask) · findings rail (plane badge with override, quality checks, export); a stage rail lights up plane → segment → measure → trust → centile |
| `/ctg` | CTG lab | 21 features grouped into heart rate, activity, decelerations, variability; held-out examples; probability gauge; diverging SHAP chart |
| `/report` | Screening report | Verdict band, input tiles (what was / was not assessed), decision trace with the fired rules (R1–R6) lit, caveats, print |
| `/evidence` | Evidence locker | Tabs per model with live metrics from `/model-info`, evaluation figures (`public/evidence/`, click to enlarge), source files and limitations |

`/model` redirects to `/evidence` (`next.config.ts`). The scan, CTG result and combined summary live in a shared
in-memory session (`src/lib/session.tsx`) so they carry across pages; a session tray in the header shows what has
been analysed and the current level. Nothing is stored on the server or in browser storage.

Other features kept from Phase 1: client-side upload checks that mirror the API; HC18 pixel spacing **auto-filled**
by SHA-256 lookup (`src/lib/hc18-pixel-sizes.json`, from `python scripts/export_pixel_sizes.py`); INTERGROWTH-21st
centile chart; print report (plane check, HC ± SD, CTG and summary); JSON and overlay downloads; loading, empty and
error states; API status; responsive down to 390 px; "not a medical device" on every page and in the report.

## Checks

```powershell
npm run lint
npm run build
npm run test:smoke     # starts `next start` and checks the pages (run after build)
```

## Layout

```text
src/app/                 layout (header, disclaimer, theme), / (analyser), /model
src/components/          analyzer, upload-dropzone, results-panel, print-report, model-info, ...
src/components/ui/       shadcn/ui primitives (button, card, alert, badge, input, label, skeleton)
src/lib/api.ts           typed API client
src/lib/pixel-size.ts    HC18 pixel-spacing auto-fill
```
