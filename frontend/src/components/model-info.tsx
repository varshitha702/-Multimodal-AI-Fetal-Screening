"use client";

import { AlertCircle, RefreshCw } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { MetricCard } from "@/components/metric-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, getModelInfo, type ModelInfo } from "@/lib/api";

interface Summary {
  mean: number;
  sd: number;
  median: number;
}

interface TestMetrics {
  test_images: number;
  dice: Summary;
  iou: Summary;
  hc_error_mm: Summary;
  hc_mae_mm: number;
  hc_abs_error_percent_mean: number;
  bland_altman_limits_of_agreement_mm: [number, number];
  ellipse_fit_failures: number;
}

interface PlaneMetrics {
  plane_accuracy: number;
  plane_macro_f1: number;
  subplane_accuracy: number;
  subplane_macro_f1: number;
  plane_ece_before: number;
  plane_ece_after: number;
  gate: { threshold: number; tt_acceptance_rate: number; non_tt_rejection_rate: number };
  hc18_acceptance: { n: number; acceptance_rate: number };
}

interface PlaneInfo {
  architecture: string;
  checkpoint_epoch: number;
  tt_threshold: number;
  threshold_rule: string;
  training_data: string;
  test_metrics: PlaneMetrics | null;
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

function PlaneGateSection({ p }: { p: PlaneInfo | null }) {
  if (!p)
    return (
      <Alert>
        <AlertTitle>Plane gate not loaded</AlertTitle>
        <AlertDescription>
          Train it with <code>python -m ml.train_planes</code>; until then every result says the plane check was
          unavailable.
        </AlertDescription>
      </Alert>
    );
  const m = p.test_metrics;
  return (
    <Card data-testid="plane-gate-info">
      <CardHeader>
        <CardTitle className="text-base">Standard-plane quality gate</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {m && (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <MetricCard
              label="Plane accuracy"
              value={pct(m.plane_accuracy)}
              hint={`macro-F1 ${m.plane_macro_f1.toFixed(3)} · 6 classes`}
              highlight
            />
            <MetricCard
              label="Brain sub-plane"
              value={pct(m.subplane_accuracy)}
              hint={`macro-F1 ${m.subplane_macro_f1.toFixed(3)} · TT/TC/TV/other`}
            />
            <MetricCard
              label="Calibration (ECE)"
              value={m.plane_ece_after.toFixed(3)}
              hint={`before temperature scaling ${m.plane_ece_before.toFixed(3)}`}
            />
            <MetricCard
              label="HC18 accepted"
              value={pct(m.hc18_acceptance.acceptance_rate)}
              hint={`of ${m.hc18_acceptance.n} images · TT test ${pct(m.gate.tt_acceptance_rate)}`}
            />
          </div>
        )}
        <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
          <Row term="Architecture" value={p.architecture} />
          <Row term="Checkpoint epoch" value={String(p.checkpoint_epoch)} />
          <Row term="Pass threshold, P(trans-thalamic)" value={`${p.tt_threshold.toFixed(3)} (${p.threshold_rule})`} />
          <Row term="Training data" value={p.training_data} />
          {m && <Row term="Non-TT images rejected (official test)" value={pct(m.gate.non_tt_rejection_rate)} />}
        </dl>
      </CardContent>
    </Card>
  );
}

interface Corr {
  rho: number;
  ci95: [number, number];
  p_value: number;
  n: number;
}

interface UncertaintyInfo {
  n_passes: number;
  thresholds: { hc_cv_percent: number; disagreement_fraction: number };
  rule: string;
  primary_hc: string;
  test_metrics: {
    n: number;
    spearman: { sd_mm_vs_abs_error_mm: Corr; cv_percent_vs_abs_error_percent: Corr };
    flag: { flagged: number; flagged_rate: number; mae_flagged_mm: number; mae_unflagged_mm: number };
  } | null;
}

function UncertaintySection({ u }: { u: UncertaintyInfo | null }) {
  if (!u) return null;
  const m = u.test_metrics;
  const s = m?.spearman.sd_mm_vs_abs_error_mm;
  return (
    <Card data-testid="uncertainty-info">
      <CardHeader>
        <CardTitle className="text-base">Measurement uncertainty (test-time augmentation)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {m && s && (
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <MetricCard
              label="SD vs actual error"
              value={`ρ ${s.rho.toFixed(2)}`}
              hint={`Spearman, 95% CI ${s.ci95[0].toFixed(2)}–${s.ci95[1].toFixed(2)} · n=${s.n}`}
              highlight
            />
            <MetricCard label="Flagged on test" value={pct(m.flag.flagged_rate)} hint={`${m.flag.flagged} of ${m.n} images`} />
            <MetricCard label="MAE, flagged" value={m.flag.mae_flagged_mm.toFixed(2)} unit="mm" hint="images flagged low reliability" />
            <MetricCard label="MAE, not flagged" value={m.flag.mae_unflagged_mm.toFixed(2)} unit="mm" hint="images passing the check" />
          </div>
        )}
        <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
          <Row term="Passes" value={`${u.n_passes} (identity, flip, zoom 0.9/1.1, gamma 0.8/1.25, flip + zoom)`} />
          <Row
            term="Low-reliability thresholds"
            value={`HC CV > ${u.thresholds.hc_cv_percent.toFixed(2)}% or disagreement > ${pct(u.thresholds.disagreement_fraction)} (${u.rule})`}
          />
          <Row term="Reported HC" value={u.primary_hc === "single" ? "single pass (lower validation MAE than the TTA mean); ± is the TTA SD" : "TTA mean"} />
        </dl>
        <p className="text-xs text-muted-foreground">
          The SD correlates only modestly with the real error: a low SD does not guarantee an accurate HC.
        </p>
      </CardContent>
    </Card>
  );
}

interface CtgInfo {
  model: string;
  best_iteration: number;
  data: string;
  test: {
    n: number;
    accuracy: number;
    macro_f1: number;
    macro_f1_ci95: [number, number];
    roc_auc_macro_ovr: number;
    pathological_recall: number;
  };
  comparison_test_macro_f1?: Record<string, number>;
}

function CtgSection({ c }: { c: CtgInfo | null }) {
  if (!c) return null;
  const t = c.test;
  return (
    <Card data-testid="ctg-info">
      <CardHeader>
        <CardTitle className="text-base">CTG classifier (NSP: Normal / Suspect / Pathological)</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <MetricCard
            label="CTG macro-F1"
            value={t.macro_f1.toFixed(3)}
            hint={`95% CI ${t.macro_f1_ci95[0].toFixed(3)}–${t.macro_f1_ci95[1].toFixed(3)} · n=${t.n}`}
            highlight
          />
          <MetricCard label="CTG accuracy" value={pct(t.accuracy)} hint="held-out test split" />
          <MetricCard label="ROC-AUC" value={t.roc_auc_macro_ovr.toFixed(3)} hint="macro, one-vs-rest" />
          <MetricCard label="Pathological recall" value={pct(t.pathological_recall)} hint="few cases: see model card" />
        </div>
        <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
          <Row term="Model" value={`${c.model} (best iteration ${c.best_iteration})`} />
          <Row term="Training data" value={c.data} />
          {c.comparison_test_macro_f1 && (
            <Row
              term="Test macro-F1 comparison"
              value={Object.entries(c.comparison_test_macro_f1)
                .map(([k, v]) => `${k.replace("_", " ")} ${v.toFixed(3)}`)
                .join(" · ")}
            />
          )}
          <Row
            term="Fusion"
            value="Rule-based screening summary; the CTG and ultrasound data come from different patients, so it is illustrative and uncalibrated."
          />
        </dl>
      </CardContent>
    </Card>
  );
}

export function ModelInfoView() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    setError(null);
    setInfo(null);
    getModelInfo()
      .then(setInfo)
      .catch((e) => setError(e instanceof ApiError ? e.message : "Could not load model information."));
  }, []);

  useEffect(load, [load]);

  if (error)
    return (
      <Alert variant="destructive">
        <AlertCircle className="size-4" />
        <AlertTitle>Model information unavailable</AlertTitle>
        <AlertDescription className="flex flex-wrap items-center gap-3">
          {error}
          <Button size="sm" variant="outline" onClick={load}>
            <RefreshCw className="size-4" /> Retry
          </Button>
        </AlertDescription>
      </Alert>
    );

  if (!info)
    return (
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
    );

  const t = info.test_metrics as TestMetrics | null;

  return (
    <div className="space-y-6" data-testid="model-info">
      {t ? (
        <>
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
            <MetricCard label="Test Dice" value={t.dice.mean.toFixed(3)} hint={`± ${t.dice.sd.toFixed(3)} SD`} highlight />
            <MetricCard label="Test IoU" value={t.iou.mean.toFixed(3)} hint={`± ${t.iou.sd.toFixed(3)} SD`} />
            <MetricCard
              label="HC mean abs. error"
              value={t.hc_mae_mm.toFixed(2)}
              unit="mm"
              hint={`${t.hc_abs_error_percent_mean.toFixed(2)}% of HC`}
            />
            <MetricCard
              label="HC bias"
              value={`${t.hc_error_mm.mean >= 0 ? "+" : ""}${t.hc_error_mm.mean.toFixed(2)}`}
              unit="mm"
              hint={`95% LoA ${t.bland_altman_limits_of_agreement_mm[0].toFixed(1)} to +${t.bland_altman_limits_of_agreement_mm[1].toFixed(1)} mm`}
            />
          </div>
          <p className="text-sm text-muted-foreground">
            Held-out test split: {t.test_images} images, split by examination so no fetus appears in both training and
            test. Ellipse-fit failures: {t.ellipse_fit_failures}.
          </p>
        </>
      ) : (
        <Alert>
          <AlertTitle>No test metrics</AlertTitle>
          <AlertDescription>Run <code>python -m ml.evaluate</code> to produce held-out test metrics.</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Checkpoint</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-x-6 gap-y-3 text-sm sm:grid-cols-2">
            <Row term="Name" value={info.name} />
            <Row term="Architecture" value={info.architecture} />
            <Row term="Input" value={`${info.input_size} × ${info.input_size} grayscale`} />
            <Row term="Inference device" value={info.device.toUpperCase()} />
            <Row term="Checkpoint epoch" value={String(info.checkpoint_epoch ?? "—")} />
            <Row term="Validation Dice" value={info.validation_dice?.toFixed(4) ?? "—"} />
            <Row term="Training data" value={info.training_data} />
            <Row term="Intended use" value={info.intended_use} />
          </dl>
        </CardContent>
      </Card>

      <PlaneGateSection p={info.plane_classifier as PlaneInfo | null} />

      <UncertaintySection u={info.uncertainty as UncertaintyInfo | null} />

      <CtgSection c={info.ctg_classifier as CtgInfo | null} />

      <Alert variant="warning">
        <AlertCircle className="size-4" />
        <AlertTitle>Limitations</AlertTitle>
        <AlertDescription>
          The segmentation model was trained and tested on one public dataset only, with no external validation.
          The plane gate was trained on a different public dataset (FETAL_PLANES_DB, two Barcelona hospitals), so it
          can be less reliable on images from other machines; it accepts only 60% of HC18 images and about a third of
          early-gestation ones (HC under 100 mm), so use the override when needed. Small, early-gestation heads are
          also the weakest case for segmentation. The
          single-image quality indicators are heuristics; the TTA uncertainty is a measured but only modest predictor of
          the real error. {info.disclaimer}
        </AlertDescription>
      </Alert>
    </div>
  );
}

function Row({ term, value }: { term: string; value: string }) {
  return (
    <div>
      <dt className="text-muted-foreground">{term}</dt>
      <dd className="font-medium">{value}</dd>
    </div>
  );
}
