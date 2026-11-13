"use client";

import { AlertCircle, AlertTriangle, FileText, Maximize2, RefreshCw, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { MetricCard } from "@/components/metric-card";
import { ApiError, getModelInfo, type ModelInfo } from "@/lib/api";
import { cn } from "@/lib/utils";

/* eslint-disable @typescript-eslint/no-explicit-any */
type Any = any;

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function Figure({ src, caption, onOpen }: { src: string; caption: string; onOpen: (s: { src: string; caption: string }) => void }) {
  return (
    <figure className="group">
      <button
        type="button"
        onClick={() => onOpen({ src, caption })}
        className="relative block w-full overflow-hidden rounded-xl border bg-white p-2 shadow-[0_12px_40px_-18px_rgba(0,0,0,0.6)] transition-transform hover:-translate-y-0.5"
        aria-label={`Enlarge: ${caption}`}
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={caption} className="w-full" loading="lazy" />
        <span className="absolute right-3 top-3 rounded-md bg-black/60 p-1.5 text-white opacity-0 transition-opacity group-hover:opacity-100">
          <Maximize2 className="size-3.5" />
        </span>
      </button>
      <figcaption className="mt-2 text-xs text-muted-foreground">{caption}</figcaption>
    </figure>
  );
}

function Sources({ files }: { files: string[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {files.map((f) => (
        <span key={f} className="inline-flex items-center gap-1 rounded-md border bg-muted/40 px-2 py-1 font-mono text-[11px] text-muted-foreground">
          <FileText className="size-3" /> {f}
        </span>
      ))}
    </div>
  );
}

function Limitation({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex gap-3 rounded-xl border border-warn/35 bg-warn/[0.06] p-4 text-sm">
      <AlertTriangle className="mt-0.5 size-4 shrink-0 text-warn" />
      <div>{children}</div>
    </div>
  );
}

const TABS = ["Segmentation", "Plane gate", "Uncertainty", "CTG", "Fusion"] as const;
type Tab = (typeof TABS)[number];

export function EvidenceLocker() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("Segmentation");
  const [zoom, setZoom] = useState<{ src: string; caption: string } | null>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);

  const load = useCallback(() => {
    setError(null);
    setInfo(null);
    getModelInfo()
      .then(setInfo)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Could not load model information."));
  }, []);
  useEffect(load, [load]);

  useEffect(() => {
    const d = dialogRef.current;
    if (!d) return;
    if (zoom && !d.open) d.showModal();
    if (!zoom && d.open) d.close();
  }, [zoom]);

  const t = info?.test_metrics as Any;
  const p = info?.plane_classifier as Any;
  const pm = p?.test_metrics;
  const u = info?.uncertainty as Any;
  const um = u?.test_metrics;
  const c = info?.ctg_classifier as Any;

  return (
    <div className="mx-auto max-w-7xl px-4 py-6" data-testid="model-info">
      {error && (
        <div role="alert" className="mb-6 flex flex-wrap items-center gap-3 rounded-xl border border-destructive/50 bg-destructive/10 p-4 text-sm">
          <AlertCircle className="size-4 text-destructive" />
          <span className="font-medium">Model information unavailable.</span> <span className="text-muted-foreground">{error}</span>
          <button type="button" onClick={load} className="ml-auto inline-flex items-center gap-1 rounded-md border px-2 py-1 text-xs">
            <RefreshCw className="size-3" /> Retry
          </button>
        </div>
      )}

      <div className="sticky top-14 z-20 -mx-4 mb-6 border-b bg-background/85 px-4 py-2 backdrop-blur" role="tablist" aria-label="Evidence">
        <div className="flex gap-1 overflow-x-auto">
          {TABS.map((x, i) => (
            <button
              key={x}
              role="tab"
              type="button"
              aria-selected={tab === x}
              onClick={() => setTab(x)}
              className={cn(
                "flex shrink-0 items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-colors",
                tab === x ? "bg-primary/10 text-primary" : "text-muted-foreground hover:text-foreground",
              )}
            >
              <span className="font-mono text-[10px]">E{i + 1}</span> {x}
            </button>
          ))}
        </div>
      </div>

      {tab === "Segmentation" && (
        <section className="space-y-6 animate-in fade-in duration-300">
          <Claim title="The HC measurement is accurate to about 2 mm on unseen fetuses.">
            U-Net trained on HC18 with a group-level split (806 examination groups), evaluated on 151 held-out images at
            original resolution.
          </Claim>
          {t ? (
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard label="Test Dice" value={t.dice.mean.toFixed(3)} hint={`± ${t.dice.sd.toFixed(3)} SD`} highlight />
              <MetricCard label="Test IoU" value={t.iou.mean.toFixed(3)} hint={`± ${t.iou.sd.toFixed(3)} SD`} />
              <MetricCard label="HC mean abs. error" value={t.hc_mae_mm.toFixed(2)} unit="mm" hint={`${t.hc_abs_error_percent_mean.toFixed(2)}% of HC`} />
              <MetricCard label="HC bias" value={`${t.hc_error_mm.mean >= 0 ? "+" : ""}${t.hc_error_mm.mean.toFixed(2)}`} unit="mm"
                hint={`95% LoA ${t.bland_altman_limits_of_agreement_mm[0].toFixed(1)} to +${t.bland_altman_limits_of_agreement_mm[1].toFixed(1)} mm`} />
            </div>
          ) : (
            <Pending error={!!error} />
          )}
          <div className="grid gap-6 md:grid-cols-2">
            <Figure src="/evidence/test_hc_agreement.png" caption="Predicted vs reference HC and Bland–Altman plot (test split, n = 151)" onOpen={setZoom} />
            <Figure src="/evidence/training_curves.png" caption="Training curves: loss, Dice and IoU per epoch" onOpen={setZoom} />
          </div>
          <Figure src="/evidence/test_examples.png" caption="Worst, 2nd worst, median and best test cases (green prediction, red reference)" onOpen={setZoom} />
          {info && (
            <dl className="panel grid gap-x-6 gap-y-3 p-5 text-sm sm:grid-cols-3">
              {[
                ["Architecture", info.architecture],
                ["Input", `${info.input_size} × ${info.input_size} grayscale`],
                ["Inference device", info.device.toUpperCase()],
                ["Checkpoint epoch", String(info.checkpoint_epoch ?? "—")],
                ["Validation Dice", info.validation_dice?.toFixed(4) ?? "—"],
                ["Training data", info.training_data],
              ].map(([k, v]) => (
                <div key={k}>
                  <dt className="readout normal-case tracking-normal">{k}</dt>
                  <dd className="font-medium">{v}</dd>
                </div>
              ))}
            </dl>
          )}
          <Sources files={["outputs/test_metrics.json", "outputs/test_predictions.csv", "docs/MODEL_CARD.md"]} />
          <Limitation>
            Small, early-gestation heads are the weakest case (worst test image: HC error 11.6 mm). One public dataset, no
            external validation.
          </Limitation>
        </section>
      )}

      {tab === "Plane gate" && (
        <section className="space-y-6 animate-in fade-in duration-300" data-testid="plane-gate-info">
          <Claim title="The plane is checked, and the confidence is calibrated.">
            EfficientNet-B0 with a plane head and a brain sub-plane head, trained on FETAL_PLANES_DB split by patient,
            evaluated on the official test set (5,271 images, unseen patients).
          </Claim>
          {pm ? (
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard label="Plane accuracy" value={pct(pm.plane_accuracy)} hint={`macro-F1 ${pm.plane_macro_f1.toFixed(3)} · 6 classes`} highlight />
              <MetricCard label="Brain sub-plane" value={pct(pm.subplane_accuracy)} hint={`macro-F1 ${pm.subplane_macro_f1.toFixed(3)}`} />
              <MetricCard label="Calibration (ECE)" value={pm.plane_ece_after.toFixed(3)} hint={`before temperature scaling ${pm.plane_ece_before.toFixed(3)}`} />
              <MetricCard label="HC18 accepted" value={pct(pm.hc18_acceptance.acceptance_rate)} tone="warn"
                hint={`of ${pm.hc18_acceptance.n} images · TT test ${pct(pm.gate.tt_acceptance_rate)}`} />
            </div>
          ) : (
            <Pending error={!!error} />
          )}
          <div className="grid gap-6 md:grid-cols-2">
            <Figure src="/evidence/plane_confusion.png" caption="Confusion matrices: plane (6 classes) and brain sub-plane" onOpen={setZoom} />
            <Figure src="/evidence/plane_reliability.png" caption="Reliability diagram before and after temperature scaling" onOpen={setZoom} />
            <Figure src="/evidence/plane_gate_scores.png" caption="Calibrated P(trans-thalamic): TT vs other planes vs HC18, with the threshold" onOpen={setZoom} />
            <Figure src="/evidence/plane_hc18_rejected.png" caption="HC18 images the gate rejects: mostly small, early-gestation heads" onOpen={setZoom} />
          </div>
          {p && <p className="text-sm text-muted-foreground">Threshold P(TT) ≥ {p.tt_threshold.toFixed(3)} ({p.threshold_rule}).</p>}
          <Sources files={["outputs/plane_metrics.json", "outputs/plane_ablation_224px/", "docs/adr/0006-standard-plane-quality-gate.md"]} />
          <Limitation>
            Only 60.3 % of HC18 images pass (31.5 % below HC 100 mm): FETAL_PLANES_DB covers 18–40 weeks from two hospitals,
            and trans-thalamic vs trans-ventricular is a fine distinction. The threshold was not tuned on HC18.
          </Limitation>
        </section>
      )}

      {tab === "Uncertainty" && (
        <section className="space-y-6 animate-in fade-in duration-300" data-testid="uncertainty-info">
          <Claim title="The spread of eight re-measurements tracks the real error, modestly.">
            Test-time augmentation (flip, zoom, gamma) with each transform undone; thresholds of the re-scan flag fixed
            on the validation split before testing.
          </Claim>
          {um ? (
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard label="SD vs actual error" value={`ρ ${um.spearman.sd_mm_vs_abs_error_mm.rho.toFixed(2)}`} highlight
                hint={`Spearman, 95% CI ${um.spearman.sd_mm_vs_abs_error_mm.ci95[0].toFixed(2)}–${um.spearman.sd_mm_vs_abs_error_mm.ci95[1].toFixed(2)}`} />
              <MetricCard label="Flagged on test" value={pct(um.flag.flagged_rate)} hint={`${um.flag.flagged} of ${um.n} images`} />
              <MetricCard label="MAE, flagged" value={um.flag.mae_flagged_mm.toFixed(2)} unit="mm" tone="warn" />
              <MetricCard label="MAE, not flagged" value={um.flag.mae_unflagged_mm.toFixed(2)} unit="mm" tone="ok" />
            </div>
          ) : (
            <Pending error={!!error} />
          )}
          <Figure src="/evidence/uncertainty_vs_error.png" caption="TTA HC SD and CV vs actual HC error (flagged images in red)" onOpen={setZoom} />
          <Figure src="/evidence/uncertainty_examples.png" caption="Per-pixel uncertainty maps, lowest to highest SD" onOpen={setZoom} />
          <Sources files={["outputs/uncertainty_metrics.json", "outputs/uncertainty_calibration.json", "docs/adr/0007-tta-measurement-uncertainty.md"]} />
          <Limitation>ρ 0.27 is a real but modest signal: 10 of the 16 largest errors are not flagged. Not a confidence interval.</Limitation>
        </section>
      )}

      {tab === "CTG" && (
        <section className="space-y-6 animate-in fade-in duration-300" data-testid="ctg-info">
          <Claim title="CTG records are classified with their reasons.">
            XGBoost on the 21 SisPorto features (UCI Cardiotocography, 2,126 records), compared with logistic regression
            and random forest; class weights from the training split only; SHAP explanations.
          </Claim>
          {c ? (
            <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
              <MetricCard label="CTG macro-F1" value={c.test.macro_f1.toFixed(3)} highlight
                hint={`95% CI ${c.test.macro_f1_ci95[0].toFixed(3)}–${c.test.macro_f1_ci95[1].toFixed(3)} · n=${c.test.n}`} />
              <MetricCard label="CTG accuracy" value={pct(c.test.accuracy)} hint="held-out test split" />
              <MetricCard label="ROC-AUC" value={c.test.roc_auc_macro_ovr.toFixed(3)} hint="macro, one-vs-rest" />
              <MetricCard label="Pathological recall" value={pct(c.test.pathological_recall)} hint="27 test cases (Wilson CI 0.875–1)" />
            </div>
          ) : (
            <Pending error={!!error} />
          )}
          {c?.comparison_test_macro_f1 && (
            <div className="panel p-5">
              <p className="readout mb-4">Model comparison (test macro-F1)</p>
              <div className="space-y-3">
                {Object.entries(c.comparison_test_macro_f1 as Record<string, number>)
                  .sort((a, b) => b[1] - a[1])
                  .map(([k, v]) => (
                    <div key={k} className="grid grid-cols-[150px_1fr_52px] items-center gap-3 text-sm">
                      <span className="capitalize">{k.replace("_", " ")}</span>
                      <div className="h-2 rounded-full bg-muted">
                        <div className={cn("h-2 rounded-full", k === "xgboost" ? "bg-primary" : "bg-muted-foreground/50")} style={{ width: `${v * 100}%` }} />
                      </div>
                      <span className="font-mono">{v.toFixed(3)}</span>
                    </div>
                  ))}
              </div>
            </div>
          )}
          <div className="grid gap-6 md:grid-cols-2">
            <Figure src="/evidence/ctg_confusion.png" caption="Confusion matrices of the three CTG models (test split)" onOpen={setZoom} />
            <Figure src="/evidence/ctg_shap_pathological.png" caption="SHAP values toward 'Pathological' (XGBoost, test split)" onOpen={setZoom} />
          </div>
          <Sources files={["outputs/ctg_metrics.json", "outputs/ctg_xgb.json", "docs/adr/0008-ctg-branch-and-rule-based-fusion.md"]} />
          <Limitation>
            Only 27 pathological records in the test split; the validation macro-F1 was 0.918, so the test figure may be
            optimistic. One hospital, 1990s SisPorto features.
          </Limitation>
        </section>
      )}

      {tab === "Fusion" && (
        <section className="space-y-6 animate-in fade-in duration-300">
          <Claim title="The combined summary is transparent by design.">
            Rule-based, decision-level fusion: a pathological CTG always escalates; any other abnormal input gives Review
            recommended; missing inputs are listed, never assumed normal. Every rule has a unit test.
          </Claim>
          <div className="grid gap-3 md:grid-cols-2">
            {[
              ["R1", "CTG Pathological", "Urgent review"],
              ["R2", "CTG Suspect", "Review recommended"],
              ["R3", "HC below the 3rd or above the 97th centile (|z| > 1.88)", "Review recommended"],
              ["R4", "Plane check did not confirm a trans-thalamic plane", "Review recommended"],
              ["R5", "TTA flags the HC as low reliability", "Review recommended"],
              ["R6", "Everything assessed is unremarkable", "Routine"],
            ].map(([id, cond, out]) => (
              <div key={id} className="panel flex items-start gap-3 p-4">
                <span className="grid size-8 shrink-0 place-items-center rounded-full border border-primary/50 font-mono text-[10px] text-primary">{id}</span>
                <div className="text-sm">
                  <p>{cond}</p>
                  <p className={cn("text-xs", out === "Urgent review" ? "text-alarm" : out === "Routine" ? "text-ok" : "text-warn")}>→ {out}</p>
                </div>
              </div>
            ))}
          </div>
          <Sources files={["ml/fusion.py", "tests/test_fusion.py", "docs/MODEL_CARD.md"]} />
          <Limitation>
            The ultrasound (HC18, FETAL_PLANES_DB) and CTG (UCI) data come from different patients: the fusion cannot be
            learned or calibrated, and the combined level has no measured accuracy. Learned fusion is Phase 2.
          </Limitation>
        </section>
      )}

      <dialog
        ref={dialogRef}
        onClose={() => setZoom(null)}
        onClick={(e) => e.target === e.currentTarget && setZoom(null)}
        className="max-h-[92vh] w-[min(1200px,94vw)] rounded-2xl border bg-card p-0 text-foreground backdrop:bg-black/70"
      >
        {zoom && (
          <div>
            <div className="flex items-center justify-between border-b px-4 py-2">
              <p className="text-sm">{zoom.caption}</p>
              <button type="button" onClick={() => setZoom(null)} aria-label="Close" className="rounded-md p-1 hover:bg-muted">
                <X className="size-4" />
              </button>
            </div>
            <div className="bg-white p-3">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={zoom.src} alt={zoom.caption} className="mx-auto max-h-[80vh]" />
            </div>
          </div>
        )}
      </dialog>
    </div>
  );
}

function Claim({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="max-w-3xl">
      <h2 className="font-display text-2xl font-semibold tracking-tight">{title}</h2>
      <p className="mt-2 text-muted-foreground">{children}</p>
    </div>
  );
}

function Pending({ error }: { error: boolean }) {
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {[0, 1, 2, 3].map((i) => (
        <div key={i} className={cn("h-24 rounded-xl border bg-card/50", !error && "animate-pulse")} />
      ))}
    </div>
  );
}
