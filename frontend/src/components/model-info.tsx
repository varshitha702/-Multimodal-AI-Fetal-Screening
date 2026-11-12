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

      <Alert variant="warning">
        <AlertCircle className="size-4" />
        <AlertTitle>Limitations</AlertTitle>
        <AlertDescription>
          Trained and tested on one public dataset only, with no external validation. It assumes a correct
          trans-thalamic plane and does not check it. Small, early-gestation heads are the weakest case. The quality
          indicators are heuristics, not calibrated uncertainty. {info.disclaimer}
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
