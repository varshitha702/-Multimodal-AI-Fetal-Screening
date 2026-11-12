"use client";

import { AlertTriangle, CheckCircle2, Download, FileJson, Printer } from "lucide-react";

import { MetricCard } from "@/components/metric-card";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { dataUrl, type Prediction } from "@/lib/api";

interface Props {
  result: Prediction;
  originalUrl: string;
  fileName: string;
}

function download(href: string, name: string) {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  a.click();
}

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

export function ResultsPanel({ result, originalUrl, fileName }: Props) {
  const stem = fileName.replace(/\.[^.]+$/, "");
  const e = result.ellipse;
  const flags = result.quality.flags;

  const downloadJson = () => {
    const { mask_png_base64: _m, overlay_png_base64: _o, ...rest } = result;
    void _m;
    void _o;
    const blob = new Blob([JSON.stringify({ file: fileName, ...rest }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    download(url, `${stem}_fetalai.json`);
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };

  return (
    <div className="space-y-6" data-testid="results">
      <Card>
        <CardHeader className="flex flex-row flex-wrap items-center justify-between gap-3 space-y-0">
          <CardTitle className="flex items-center gap-2 text-lg">
            {result.head_detected ? (
              <CheckCircle2 className="size-5 text-emerald-500" />
            ) : (
              <AlertTriangle className="size-5 text-amber-500" />
            )}
            {result.head_detected ? "Fetal head segmented" : "No fetal head detected"}
          </CardTitle>
          <div className="flex flex-wrap gap-2 print:hidden">
            <Button variant="outline" size="sm" onClick={() => window.print()}>
              <Printer className="size-4" /> Print report
            </Button>
            <Button variant="outline" size="sm" onClick={downloadJson}>
              <FileJson className="size-4" /> JSON
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => download(dataUrl(result.overlay_png_base64), `${stem}_overlay.png`)}
            >
              <Download className="size-4" /> Overlay
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 sm:grid-cols-3">
            {[
              { label: "Original", src: originalUrl },
              { label: "Overlay + fitted ellipse", src: dataUrl(result.overlay_png_base64) },
              { label: "Predicted mask", src: dataUrl(result.mask_png_base64) },
            ].map((img) => (
              <figure key={img.label} className="overflow-hidden rounded-lg border bg-black">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={img.src} alt={img.label} className="aspect-[4/3] w-full object-contain" />
                <figcaption className="border-t bg-card px-3 py-1.5 text-xs text-muted-foreground">{img.label}</figcaption>
              </figure>
            ))}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <MetricCard
          label="Head circumference"
          value={result.head_circumference_mm !== null ? result.head_circumference_mm.toFixed(1) : "—"}
          unit={result.head_circumference_mm !== null ? "mm" : undefined}
          hint={result.pixel_size_mm ? `at ${result.pixel_size_mm.toFixed(4)} mm/px` : "needs pixel spacing"}
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
          value={result.head_area_mm2 !== null ? (result.head_area_mm2 / 100).toFixed(1) : result.head_area_px.toLocaleString()}
          unit={result.head_area_mm2 !== null ? "cm²" : "px"}
          hint={`${pct(result.quality.mask_area_fraction)} of image`}
        />
        <MetricCard
          label="Model confidence"
          value={pct(result.quality.mean_confidence)}
          hint={`${pct(result.quality.uncertain_fraction)} uncertain boundary · ${result.inference_ms.toFixed(0)} ms`}
        />
      </div>

      {flags.length > 0 ? (
        <Alert variant="warning">
          <AlertTriangle className="size-4" />
          <AlertTitle>Quality checks</AlertTitle>
          <AlertDescription>
            <ul className="mt-1 list-disc space-y-1 pl-4">
              {flags.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          </AlertDescription>
        </Alert>
      ) : (
        <Alert>
          <CheckCircle2 className="size-4 text-emerald-500" />
          <AlertTitle>Quality checks passed</AlertTitle>
          <AlertDescription>
            High confidence, clear boundary, head fully in view and an elliptical shape.
          </AlertDescription>
        </Alert>
      )}

      {e && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Measurement details</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-2 gap-x-6 gap-y-2 text-sm sm:grid-cols-3">
              <Detail term="Image size" value={`${result.image_width} × ${result.image_height} px`} />
              <Detail term="Ellipse centre" value={`(${e.center_x_px.toFixed(0)}, ${e.center_y_px.toFixed(0)}) px`} />
              <Detail term="Rotation" value={`${e.angle_deg.toFixed(1)}°`} />
              <Detail term="Semi-axes" value={`${e.semi_major_px.toFixed(1)} / ${e.semi_minor_px.toFixed(1)} px`} />
              <Detail
                term="Ellipse fit ratio"
                value={result.quality.ellipse_fit_ratio?.toFixed(3) ?? "—"}
              />
              <Detail term="Model" value={<Badge variant="secondary">{result.model_version}</Badge>} />
            </dl>
            <p className="mt-4 text-xs text-muted-foreground">
              HC = perimeter of the ellipse fitted to the segmented head (Ramanujan approximation) × pixel spacing.
            </p>
          </CardContent>
        </Card>
      )}
    </div>
  );
}

function Detail({ term, value }: { term: string; value: React.ReactNode }) {
  return (
    <div>
      <dt className="text-muted-foreground">{term}</dt>
      <dd className="font-medium tabular-nums">{value}</dd>
    </div>
  );
}
