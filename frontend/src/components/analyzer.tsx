"use client";

import { AlertCircle, Loader2, ScanSearch, Sparkles } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

import { PrintReport } from "@/components/print-report";
import { ResultsPanel } from "@/components/results-panel";
import { UploadDropzone } from "@/components/upload-dropzone";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, predict, type Prediction } from "@/lib/api";
import { lookupPixelSize, type PixelSizeMatch } from "@/lib/pixel-size";

type Status = "idle" | "loading" | "done" | "error";

const PX_MIN = 0.001;
const PX_MAX = 1;
const GA_MIN = 14;
const GA_MAX = 40;

export function Analyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [pixelSize, setPixelSize] = useState("");
  const [match, setMatch] = useState<PixelSizeMatch | null>(null);
  const [gaWeeks, setGaWeeks] = useState("");
  const [gaDays, setGaDays] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<Prediction | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => () => abortRef.current?.abort(), []);

  const selectFile = useCallback(async (f: File | null) => {
    abortRef.current?.abort();
    setResult(null);
    setError(null);
    setStatus("idle");
    setMatch(null);
    setPreviewUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return f ? URL.createObjectURL(f) : null;
    });
    setFile(f);
    if (f) {
      const hit = await lookupPixelSize(f);
      setMatch(hit);
      setPixelSize(hit ? String(hit.pixelSizeMm) : "");
    } else {
      setPixelSize("");
    }
  }, []);

  const pxValue = pixelSize.trim() === "" ? null : Number(pixelSize);
  const pxInvalid = pxValue !== null && (!Number.isFinite(pxValue) || pxValue < PX_MIN || pxValue > PX_MAX);

  // Gestational age as weeks + days -> exact weeks (INTERGROWTH-21st standard covers 14+0 to 40+0)
  const gaGiven = gaWeeks.trim() !== "" || gaDays.trim() !== "";
  const gaW = Number(gaWeeks.trim() || "NaN");
  const gaD = gaDays.trim() === "" ? 0 : Number(gaDays);
  const gaValue = gaGiven ? gaW + gaD / 7 : null;
  const gaInvalid =
    gaGiven &&
    (!Number.isInteger(gaW) || !Number.isInteger(gaD) || gaD < 0 || gaD > 6 || gaValue! < GA_MIN || gaValue! > GA_MAX);

  async function analyze() {
    if (!file || pxInvalid || gaInvalid) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setStatus("loading");
    setError(null);
    try {
      const r = await predict(file, pxValue, gaInvalid ? null : gaValue, controller.signal);
      setResult(r);
      setStatus("done");
    } catch (e) {
      if (controller.signal.aborted) return;
      setError(e instanceof ApiError ? e.message : "Unexpected error while analysing the image.");
      setStatus("error");
    }
  }

  return (
    <>
      <div className="grid gap-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)] print:hidden">
        <Card className="h-fit lg:sticky lg:top-20">
          <CardHeader>
            <CardTitle className="text-lg">Image and calibration</CardTitle>
            <CardDescription>Upload a 2-D ultrasound image of the fetal head.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-5">
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
                <Label htmlFor="pixel-size">Pixel spacing (mm / pixel)</Label>
                {match && (
                  <Badge variant="secondary" className="gap-1" title={match.source}>
                    <Sparkles className="size-3" /> HC18 auto-filled
                  </Badge>
                )}
              </div>
              <Input
                id="pixel-size"
                inputMode="decimal"
                placeholder="e.g. 0.1386 (optional)"
                value={pixelSize}
                onChange={(e) => {
                  setPixelSize(e.target.value);
                  setMatch(null);
                }}
                aria-invalid={pxInvalid}
                className={pxInvalid ? "border-destructive focus-visible:ring-destructive" : undefined}
              />
              <p className={pxInvalid ? "text-xs text-destructive" : "text-xs text-muted-foreground"}>
                {pxInvalid
                  ? `Enter a number between ${PX_MIN} and ${PX_MAX}.`
                  : match
                    ? `Matched ${match.source} from the HC18 dataset.`
                    : "Needed to report head circumference in millimetres. Found in the scanner's DICOM metadata."}
              </p>
            </div>

            <div className="space-y-2">
              <Label htmlFor="ga-weeks">Gestational age (optional)</Label>
              <div className="flex items-center gap-2">
                <Input
                  id="ga-weeks"
                  inputMode="numeric"
                  placeholder="weeks"
                  value={gaWeeks}
                  onChange={(e) => setGaWeeks(e.target.value)}
                  aria-invalid={gaInvalid}
                  className={gaInvalid ? "border-destructive" : undefined}
                />
                <span className="text-muted-foreground">+</span>
                <Input
                  id="ga-days"
                  aria-label="Gestational age days"
                  inputMode="numeric"
                  placeholder="days"
                  value={gaDays}
                  onChange={(e) => setGaDays(e.target.value)}
                  aria-invalid={gaInvalid}
                  className={gaInvalid ? "border-destructive" : undefined}
                />
              </div>
              <p className={gaInvalid ? "text-xs text-destructive" : "text-xs text-muted-foreground"}>
                {gaInvalid
                  ? `Enter whole weeks and 0–6 days, between ${GA_MIN}+0 and ${GA_MAX}+0.`
                  : "From the dating scan. Compares HC with the INTERGROWTH-21st standard (14–40 weeks)."}
              </p>
            </div>

            <Button
              className="w-full"
              size="lg"
              onClick={analyze}
              disabled={!file || pxInvalid || gaInvalid || status === "loading"}
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
            </Button>
          </CardContent>
        </Card>

        <section aria-live="polite" className="min-w-0">
          {status === "error" && error && (
            <Alert variant="destructive" className="mb-6">
              <AlertCircle className="size-4" />
              <AlertTitle>Could not analyse the image</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
          {status === "loading" && <LoadingState />}
          {status === "done" && result && previewUrl && file && (
            <ResultsPanel result={result} originalUrl={previewUrl} fileName={file.name} />
          )}
          {(status === "idle" || (status === "error" && !result)) && <EmptyState />}
        </section>
      </div>

      {status === "done" && result && previewUrl && file && (
        <PrintReport result={result} originalUrl={previewUrl} fileName={file.name} />
      )}
    </>
  );
}

function LoadingState() {
  return (
    <div className="space-y-6" data-testid="loading">
      <Card>
        <CardContent className="grid gap-3 p-6 sm:grid-cols-3">
          {[0, 1, 2].map((i) => (
            <Skeleton key={i} className="aspect-[4/3] w-full" />
          ))}
        </CardContent>
      </Card>
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-24" />
        ))}
      </div>
    </div>
  );
}

function EmptyState() {
  const steps = [
    ["Upload", "A PNG or JPEG trans-thalamic ultrasound image."],
    ["Segment", "A U-Net outlines the fetal skull."],
    ["Measure", "An ellipse is fitted and its perimeter gives the head circumference."],
  ];
  return (
    <Card className="border-dashed" data-testid="empty-state">
      <CardContent className="flex flex-col items-center px-6 py-14 text-center">
        <span className="rounded-full bg-primary/10 p-4 text-primary">
          <ScanSearch className="size-8" />
        </span>
        <h2 className="mt-4 text-lg font-semibold">No analysis yet</h2>
        <p className="mt-1 max-w-md text-sm text-muted-foreground">
          Results appear here: the original image, the segmentation overlay, the mask and the measurements.
        </p>
        <ol className="mt-8 grid w-full max-w-2xl gap-4 text-left sm:grid-cols-3">
          {steps.map(([title, text], i) => (
            <li key={title} className="rounded-lg border bg-muted/40 p-4">
              <span className="text-xs font-semibold text-primary">STEP {i + 1}</span>
              <p className="mt-1 font-medium">{title}</p>
              <p className="mt-1 text-sm text-muted-foreground">{text}</p>
            </li>
          ))}
        </ol>
      </CardContent>
    </Card>
  );
}
