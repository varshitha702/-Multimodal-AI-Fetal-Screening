"use client";

import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  Crosshair,
  Download,
  FileJson,
  Loader2,
  Printer,
  RotateCcw,
  ScanSearch,
  Sparkles,
  Stethoscope,
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { CornerBrackets, PanelTitle, StageRail, type StageState } from "@/components/console";
import { GrowthPanel } from "@/components/growth-panel";
import { MetricCard } from "@/components/metric-card";
import { PlaneCheckCard } from "@/components/plane-check";
import { PrintReport } from "@/components/print-report";
import { UploadDropzone } from "@/components/upload-dropzone";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError, dataUrl, predict, type Prediction } from "@/lib/api";
import { lookupPixelSize } from "@/lib/pixel-size";
import { useSession } from "@/lib/session";
import { cn } from "@/lib/utils";

type Status = "idle" | "loading" | "done" | "error";
type View = "overlay" | "uncertainty" | "original" | "mask";

const PX_MIN = 0.001;
const PX_MAX = 1;
const GA_MIN = 14;
const GA_MAX = 40;
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function download(href: string, name: string) {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  a.click();
}

function stages(status: Status, hasFile: boolean, r: Prediction | null): { label: string; state: StageState; note?: string }[] {
  const base = (s: StageState) =>
    ["Plane", "Segment", "Measure", "Trust", "Centile"].map((label) => ({ label, state: s }));
  if (status === "loading") return base("running");
  if (!r) return base(hasFile ? "ready" : "idle");
  const pc = r.plane_check.status;
  return [
    { label: "Plane", state: pc === "pass" ? "done" : pc === "warn" ? "warn" : "skip", note: pc },
    { label: "Segment", state: r.head_detected ? "done" : "warn", note: r.head_detected ? "head found" : "no head" },
    {
      label: "Measure",
      state: r.head_circumference_mm !== null ? "done" : r.measurement_withheld ? "warn" : "skip",
      note: r.measurement_withheld ? "withheld" : r.head_circumference_mm !== null ? "HC mm" : "no spacing",
    },
    {
      label: "Trust",
      state: !r.uncertainty ? "skip" : r.uncertainty.low_reliability ? "warn" : "done",
      note: !r.uncertainty ? "off" : r.uncertainty.low_reliability ? "re-scan" : "reliable",
    },
    { label: "Centile", state: r.growth ? "done" : "skip", note: r.growth ? `${Math.round(r.growth.centile)}th` : "no GA" },
  ];
}

export function ScanWorkstation() {
  const { scan, setScan, ctg, summary } = useSession();
  const { file, previewUrl, pixelSize, match, gaWeeks, gaDays, result } = scan;
  const [status, setStatus] = useState<Status>(result ? "done" : "idle");
  const [error, setError] = useState<string | null>(null);
  const [overriding, setOverriding] = useState(false);
  const [view, setView] = useState<View>("overlay");
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const selectFile = useCallback(
    async (f: File | null) => {
      abortRef.current?.abort();
      setError(null);
      setStatus("idle");
      setView("overlay");
      if (previewUrl) URL.revokeObjectURL(previewUrl);
      setScan({ file: f, previewUrl: f ? URL.createObjectURL(f) : null, result: null, match: null, pixelSize: "" });
      if (f) {
        const hit = await lookupPixelSize(f);
        setScan({ match: hit, pixelSize: hit ? String(hit.pixelSizeMm) : "" });
      }
    },
    [previewUrl, setScan],
  );

  const pxValue = pixelSize.trim() === "" ? null : Number(pixelSize);
  const pxInvalid = pxValue !== null && (!Number.isFinite(pxValue) || pxValue < PX_MIN || pxValue > PX_MAX);
  const gaGiven = gaWeeks.trim() !== "" || gaDays.trim() !== "";
  const gaW = Number(gaWeeks.trim() || "NaN");
  const gaD = gaDays.trim() === "" ? 0 : Number(gaDays);
  const gaValue = gaGiven ? gaW + gaD / 7 : null;
  const gaInvalid =
    gaGiven && (!Number.isInteger(gaW) || !Number.isInteger(gaD) || gaD < 0 || gaD > 6 || gaValue! < GA_MIN || gaValue! > GA_MAX);

  async function analyze(overridePlaneCheck = false) {
    if (!file || pxInvalid || gaInvalid) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    if (overridePlaneCheck) setOverriding(true);
    else setStatus("loading");
    setError(null);
    try {
      const r = await predict(file, pxValue, gaInvalid ? null : gaValue, controller.signal, overridePlaneCheck);
      setScan({ result: r });
      setStatus("done");
    } catch (e) {
      if (controller.signal.aborted) return;
      setError(e instanceof ApiError ? e.message : "Unexpected error while analysing the image.");
      setStatus("error");
    } finally {
      setOverriding(false);
    }
  }

  const r = status === "done" ? result : null;
  const u = r?.uncertainty ?? null;
  const e = r?.ellipse ?? null;
  const viewSrc =
    r && view === "overlay"
      ? dataUrl(r.overlay_png_base64)
      : r && view === "uncertainty" && r.uncertainty_png_base64
        ? dataUrl(r.uncertainty_png_base64)
        : r && view === "mask"
          ? dataUrl(r.mask_png_base64)
          : previewUrl;
  const viewLabel =
    r && view === "uncertainty"
      ? `Uncertainty map (SD of ${u?.n_passes ?? 8} TTA passes)`
      : r && view === "mask"
        ? "Predicted mask"
        : r && view === "overlay"
          ? "Overlay + fitted ellipse"
          : "Original";
  const hcValue =
    !r || r.head_circumference_mm === null
      ? "—"
      : u?.hc_sd_mm != null
        ? `${r.head_circumference_mm.toFixed(1)} ± ${u.hc_sd_mm.toFixed(1)}`
        : r.head_circumference_mm.toFixed(1);
  const stem = file?.name.replace(/\.[^.]+$/, "") ?? "scan";

  return (
    <>
      <div className="mx-auto max-w-7xl px-4 py-6 print:hidden">
        <div className="panel mb-6 px-4 py-4">
          <StageRail stages={stages(status, !!file, r)} />
        </div>

        <div className="grid gap-6 xl:grid-cols-[300px_minmax(0,1fr)_340px]">
          {/* ---------------- input ---------------- */}
          <section className="panel h-fit xl:sticky xl:top-20" aria-label="Image input">
            <PanelTitle icon={<Stethoscope className="size-4 text-primary" />}>Probe input</PanelTitle>
            <div className="space-y-5 p-4">
              <UploadDropzone
                file={file}
                previewUrl={previewUrl}
                onFile={selectFile}
                onReject={(m) => {
                  setError(m);
                  setStatus("error");
                }}
                disabled={status === "loading"}
              />
              <div className="space-y-2">
                <div className="flex items-center justify-between gap-2">
                  <Label htmlFor="pixel-size" className="readout normal-case tracking-normal">
                    Pixel spacing (mm / pixel)
                  </Label>
                  {match && (
                    <span className="inline-flex items-center gap-1 rounded-full border border-primary/40 bg-primary/10 px-2 py-0.5 text-[10px] text-primary" title={match.source}>
                      <Sparkles className="size-3" /> HC18 auto-filled
                    </span>
                  )}
                </div>
                <Input
                  id="pixel-size"
                  inputMode="decimal"
                  placeholder="e.g. 0.1386 (optional)"
                  value={pixelSize}
                  onChange={(ev) => setScan({ pixelSize: ev.target.value, match: null })}
                  aria-invalid={pxInvalid}
                  className={cn("font-mono", pxInvalid && "border-destructive")}
                />
                <p className={cn("text-xs", pxInvalid ? "text-destructive" : "text-muted-foreground")}>
                  {pxInvalid
                    ? `Enter a number between ${PX_MIN} and ${PX_MAX}.`
                    : match
                      ? `Matched ${match.source} from the HC18 dataset.`
                      : "Needed to report HC in millimetres (scanner DICOM metadata)."}
                </p>
              </div>
              <div className="space-y-2">
                <Label htmlFor="ga-weeks" className="readout normal-case tracking-normal">
                  Gestational age (optional)
                </Label>
                <div className="flex items-center gap-2">
                  <Input id="ga-weeks" inputMode="numeric" placeholder="weeks" value={gaWeeks}
                    onChange={(ev) => setScan({ gaWeeks: ev.target.value })} aria-invalid={gaInvalid}
                    className={cn("font-mono", gaInvalid && "border-destructive")} />
                  <span className="text-muted-foreground">+</span>
                  <Input id="ga-days" aria-label="Gestational age days" inputMode="numeric" placeholder="days" value={gaDays}
                    onChange={(ev) => setScan({ gaDays: ev.target.value })} aria-invalid={gaInvalid}
                    className={cn("font-mono", gaInvalid && "border-destructive")} />
                </div>
                <p className={cn("text-xs", gaInvalid ? "text-destructive" : "text-muted-foreground")}>
                  {gaInvalid
                    ? `Enter whole weeks and 0–6 days, between ${GA_MIN}+0 and ${GA_MAX}+0.`
                    : "Places HC on the INTERGROWTH-21st standard (14–40 weeks)."}
                </p>
              </div>
              <button
                type="button"
                onClick={() => analyze()}
                disabled={!file || pxInvalid || gaInvalid || status === "loading"}
                className="group relative flex h-11 w-full items-center justify-center gap-2 overflow-hidden rounded-lg bg-primary font-medium text-primary-foreground shadow-[0_0_24px_-6px_hsl(var(--primary))] transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
              >
                {status === "loading" ? (
                  <>
                    <Loader2 className="size-4 animate-spin" /> Analysing…
                  </>
                ) : (
                  <>
                    <ScanSearch className="size-4" /> Analyse image
                  </>
                )}
              </button>
            </div>
          </section>

          {/* ---------------- viewport ---------------- */}
          <section className="min-w-0 space-y-4" aria-live="polite">
            {status === "error" && error && (
              <div role="alert" className="flex gap-3 rounded-xl border border-destructive/50 bg-destructive/10 p-4 text-sm">
                <AlertCircle className="mt-0.5 size-4 shrink-0 text-destructive" />
                <div>
                  <p className="font-medium">Could not analyse the image</p>
                  <p className="text-muted-foreground">{error}</p>
                </div>
              </div>
            )}

            <div className="panel overflow-hidden">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b px-3 py-2">
                <div className="flex gap-1" role="group" aria-label="Overlay view">
                  {(
                    [
                      ["overlay", "Segmentation"],
                      ["uncertainty", "Uncertainty map"],
                      ["original", "Original"],
                      ["mask", "Mask"],
                    ] as [View, string][]
                  ).map(([k, l]) => (
                    <button
                      key={k}
                      type="button"
                      onClick={() => setView(k)}
                      disabled={!r || (k === "uncertainty" && !r.uncertainty_png_base64)}
                      aria-pressed={view === k}
                      className={cn(
                        "rounded-md px-2.5 py-1 text-xs transition-colors disabled:opacity-40",
                        view === k && r ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted",
                      )}
                    >
                      {l}
                    </button>
                  ))}
                </div>
                <span className="readout">{viewLabel}</span>
              </div>

              <div className="scanlines relative aspect-[4/3] bg-black" data-testid={!file ? "empty-state" : status === "loading" ? "loading" : undefined}>
                <CornerBrackets />
                {viewSrc ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={viewSrc} alt={viewLabel} className="absolute inset-0 h-full w-full object-contain" />
                ) : (
                  <NoSignal />
                )}
                {status === "loading" && (
                  <div className="absolute inset-0 overflow-hidden">
                    <div className="absolute inset-x-0 h-24 animate-[scan_1.6s_ease-in-out_infinite] bg-gradient-to-b from-transparent via-primary/25 to-transparent" />
                    <p className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full bg-black/60 px-3 py-1 font-mono text-[11px] text-primary">
                      plane check · segmentation · 8 TTA passes…
                    </p>
                  </div>
                )}
                {/* HUD */}
                {file && (
                  <div className="pointer-events-none absolute left-4 top-4 z-10 space-y-0.5 font-mono text-[10px] text-white/70">
                    <p className="text-white/90">{file.name}</p>
                    {r && <p>{r.image_width} × {r.image_height} px</p>}
                    {pxValue && <p>{pxValue.toFixed(4)} mm/px</p>}
                  </div>
                )}
                {r && (
                  <div className="pointer-events-none absolute bottom-4 left-4 z-10 rounded-md border border-white/10 bg-black/60 px-3 py-2 font-mono text-white backdrop-blur">
                    <p className="text-[10px] text-white/50">HC</p>
                    <p className="text-lg text-echo">
                      {hcValue}
                      {r.head_circumference_mm !== null && <span className="text-xs text-white/60"> mm</span>}
                    </p>
                  </div>
                )}
                {!file && <div className="absolute inset-x-0 bottom-6 text-center text-sm text-white/60">Drop an image in the probe input to begin.</div>}
              </div>
            </div>

            {r && (
              <div className="grid grid-cols-2 gap-3" data-testid="results">
                <MetricCard
                  label="Head circumference"
                  value={hcValue}
                  unit={r.head_circumference_mm !== null ? "mm" : undefined}
                  hint={
                    r.measurement_withheld
                      ? "withheld: plane check"
                      : r.pixel_size_mm
                        ? `at ${r.pixel_size_mm.toFixed(4)} mm/px${r.plane_check.overridden ? " · override" : ""}`
                        : "needs pixel spacing"
                  }
                  highlight
                />
                <MetricCard
                  label="Ellipse axes"
                  value={e?.semi_major_mm ? `${(2 * e.semi_major_mm).toFixed(1)} × ${(2 * e.semi_minor_mm!).toFixed(1)}` : e ? `${(2 * e.semi_major_px).toFixed(0)} × ${(2 * e.semi_minor_px).toFixed(0)}` : "—"}
                  unit={e?.semi_major_mm ? "mm" : e ? "px" : undefined}
                  hint="major × minor diameter"
                />
                <MetricCard
                  label="Head area"
                  value={r.head_area_mm2 !== null ? (r.head_area_mm2 / 100).toFixed(1) : r.head_area_px.toLocaleString()}
                  unit={r.head_area_mm2 !== null ? "cm²" : "px"}
                  hint={`${pct(r.quality.mask_area_fraction)} of image`}
                />
                {u ? (
                  <MetricCard
                    label="Reliability"
                    value={u.low_reliability ? "Low: re-scan" : "OK"}
                    tone={u.low_reliability ? "warn" : "ok"}
                    hint={`CV ${u.hc_cv_percent !== null ? `${u.hc_cv_percent.toFixed(2)}%` : "—"} · disagreement ${pct(u.disagreement_fraction)} · ${r.inference_ms.toFixed(0)} ms`}
                  />
                ) : (
                  <MetricCard label="Model confidence" value={pct(r.quality.mean_confidence)} hint={`${r.inference_ms.toFixed(0)} ms`} />
                )}
              </div>
            )}

            {r?.growth && r.head_circumference_mm !== null && <GrowthPanel growth={r.growth} hcMm={r.head_circumference_mm} />}

            {r && e && (
              <div className="panel">
                <PanelTitle icon={<Crosshair className="size-4 text-primary" />}>Measurement details</PanelTitle>
                <dl className="grid grid-cols-2 gap-x-6 gap-y-3 p-4 text-sm sm:grid-cols-4">
                  {[
                    ["Image size", `${r.image_width} × ${r.image_height} px`],
                    ["Ellipse centre", `(${e.center_x_px.toFixed(0)}, ${e.center_y_px.toFixed(0)}) px`],
                    ["Rotation", `${e.angle_deg.toFixed(1)}°`],
                    ["Semi-axes", `${e.semi_major_px.toFixed(1)} / ${e.semi_minor_px.toFixed(1)} px`],
                    ["Ellipse fit ratio", r.quality.ellipse_fit_ratio?.toFixed(3) ?? "—"],
                    ["Model confidence", pct(r.quality.mean_confidence)],
                    ...(u?.hc_mean_mm != null ? [["TTA mean HC", `${u.hc_mean_mm.toFixed(1)} mm (${u.n_passes} passes)`]] : []),
                    ["Model", r.model_version],
                  ].map(([t, v]) => (
                    <div key={t}>
                      <dt className="readout normal-case tracking-normal">{t}</dt>
                      <dd className="font-mono text-[13px]">{v}</dd>
                    </div>
                  ))}
                </dl>
                <p className="border-t px-4 py-3 text-xs text-muted-foreground">
                  HC = perimeter of the ellipse fitted to the segmented head (Ramanujan) × pixel spacing.
                  {u && " ± is the SD of HC over test-time-augmented passes (flip, zoom, gamma), not a confidence interval."}
                </p>
              </div>
            )}
          </section>

          {/* ---------------- findings ---------------- */}
          <aside className="space-y-4" aria-label="Findings">
            {r ? (
              <>
                <PlaneCheckCard check={r.plane_check} withheld={r.measurement_withheld} onOverride={() => analyze(true)} overriding={overriding} />
                <div className="panel">
                  <PanelTitle
                    icon={r.quality.flags.length ? <AlertTriangle className="size-4 text-warn" /> : <CheckCircle2 className="size-4 text-ok" />}
                  >
                    {r.measurement_withheld ? "Segmentation preview (measurement withheld)" : r.head_detected ? "Fetal head segmented" : "No fetal head detected"}
                  </PanelTitle>
                  <div className="p-4 text-sm">
                    {r.quality.flags.length ? (
                      <>
                        <p className="readout mb-2 text-warn">Quality checks</p>
                        <ul className="space-y-2">
                          {r.quality.flags.map((f) => (
                            <li key={f} className="flex gap-2">
                              <span className="mt-1.5 size-1.5 shrink-0 rounded-full bg-warn" />
                              <span>{f}</span>
                            </li>
                          ))}
                        </ul>
                      </>
                    ) : (
                      <p className="text-muted-foreground">
                        <span className="font-medium text-foreground">Quality checks passed.</span> High confidence, clear
                        boundary, head fully in view, elliptical shape.
                      </p>
                    )}
                  </div>
                </div>
                <div className="panel p-4">
                  <p className="readout mb-3">Export</p>
                  <div className="grid grid-cols-3 gap-2">
                    <button type="button" onClick={() => window.print()} className="flex flex-col items-center gap-1 rounded-lg border p-2 text-xs hover:border-primary/50">
                      <Printer className="size-4" /> Print report
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        const { mask_png_base64: _m, overlay_png_base64: _o, uncertainty_png_base64: _u, ...rest } = r;
                        void _m;
                        void _o;
                        void _u;
                        const url = URL.createObjectURL(new Blob([JSON.stringify({ file: file?.name, ...rest }, null, 2)], { type: "application/json" }));
                        download(url, `${stem}_fetalai.json`);
                        setTimeout(() => URL.revokeObjectURL(url), 1000);
                      }}
                      className="flex flex-col items-center gap-1 rounded-lg border p-2 text-xs hover:border-primary/50"
                    >
                      <FileJson className="size-4" /> JSON
                    </button>
                    <button type="button" onClick={() => download(dataUrl(r.overlay_png_base64), `${stem}_overlay.png`)} className="flex flex-col items-center gap-1 rounded-lg border p-2 text-xs hover:border-primary/50">
                      <Download className="size-4" /> Overlay
                    </button>
                  </div>
                </div>
                <Link href={ctg.result ? "/report" : "/ctg"} className="panel group flex items-center justify-between p-4 transition-colors hover:border-primary/50">
                  <span>
                    <span className="readout block">Next</span>
                    <span className="font-display font-semibold">{ctg.result ? "Open the combined report" : "Add a CTG record"}</span>
                    {summary && <span className="block text-xs text-muted-foreground">Current level: {summary.level}</span>}
                  </span>
                  <ArrowRight className="size-4 text-primary transition-transform group-hover:translate-x-1" />
                </Link>
              </>
            ) : (
              <div className="panel p-5">
                <p className="readout">Findings</p>
                <ol className="mt-4 space-y-4 text-sm">
                  {[
                    ["Plane gate", "Is it the trans-thalamic plane? If not, HC is withheld unless you override."],
                    ["Segment + measure", "U-Net outline, ellipse fit, HC in mm."],
                    ["Trust", "8 augmented passes give HC ± SD and an uncertainty map."],
                    ["Centile", "With gestational age: INTERGROWTH-21st position."],
                  ].map(([t, d], i) => (
                    <li key={t} className="flex gap-3">
                      <span className="grid size-6 shrink-0 place-items-center rounded-full border font-mono text-[10px] text-primary">{i + 1}</span>
                      <span>
                        <span className="font-medium">{t}</span>
                        <span className="block text-muted-foreground">{d}</span>
                      </span>
                    </li>
                  ))}
                </ol>
                {(file || result) && (
                  <button type="button" onClick={() => selectFile(null)} className="mt-6 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
                    <RotateCcw className="size-3" /> Clear image
                  </button>
                )}
              </div>
            )}
          </aside>
        </div>
      </div>

      {r && previewUrl && file && <PrintReport result={r} originalUrl={previewUrl} fileName={file.name} ctg={ctg.result} summary={summary} />}
    </>
  );
}

/** Empty viewport: an idle probe sweep. */
function NoSignal() {
  return (
    <div className="absolute inset-0 grid place-items-center">
      <svg viewBox="0 0 200 160" className="h-2/3 w-2/3 opacity-60" aria-hidden>
        <path d="M100 8 L180 140 A92 92 0 0 1 20 140 Z" fill="none" stroke="hsl(var(--echo) / 0.35)" strokeDasharray="3 4" />
        <g className="animate-sweep" style={{ transformOrigin: "100px 8px" }}>
          <path d="M100 8 L94 150 L106 150 Z" fill="hsl(var(--echo) / 0.25)" />
        </g>
        <text x="100" y="100" textAnchor="middle" className="fill-[hsl(var(--echo)/0.8)] font-mono text-[9px] tracking-[0.3em]">
          NO SIGNAL
        </text>
      </svg>
    </div>
  );
}
