"use client";

import { Activity, AlertCircle, FlaskConical, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError, getCtgExamples, predictCtg, type CtgClass, type CtgExample, type CtgPrediction } from "@/lib/api";
import { cn } from "@/lib/utils";

// Mirrors ml/ctg.py FEATURES (the backend validates again and returns 422 on bad input).
const FIELDS: { name: string; label: string; unit: string; min: number; max: number }[] = [
  { name: "LB", label: "FHR baseline", unit: "bpm", min: 50, max: 240 },
  { name: "AC", label: "Accelerations", unit: "/s", min: 0, max: 1 },
  { name: "FM", label: "Fetal movements", unit: "/s", min: 0, max: 1 },
  { name: "UC", label: "Uterine contractions", unit: "/s", min: 0, max: 1 },
  { name: "DL", label: "Light decelerations", unit: "/s", min: 0, max: 1 },
  { name: "DS", label: "Severe decelerations", unit: "/s", min: 0, max: 1 },
  { name: "DP", label: "Prolonged decelerations", unit: "/s", min: 0, max: 1 },
  { name: "ASTV", label: "Abnormal STV time", unit: "%", min: 0, max: 100 },
  { name: "MSTV", label: "Mean STV", unit: "bpm", min: 0, max: 20 },
  { name: "ALTV", label: "Abnormal LTV time", unit: "%", min: 0, max: 100 },
  { name: "MLTV", label: "Mean LTV", unit: "bpm", min: 0, max: 100 },
  { name: "Width", label: "Histogram width", unit: "bpm", min: 0, max: 250 },
  { name: "Min", label: "Histogram min", unit: "bpm", min: 0, max: 250 },
  { name: "Max", label: "Histogram max", unit: "bpm", min: 0, max: 300 },
  { name: "Nmax", label: "Histogram peaks", unit: "", min: 0, max: 50 },
  { name: "Nzeros", label: "Histogram zeros", unit: "", min: 0, max: 50 },
  { name: "Mode", label: "Histogram mode", unit: "bpm", min: 0, max: 300 },
  { name: "Mean", label: "Histogram mean", unit: "bpm", min: 0, max: 300 },
  { name: "Median", label: "Histogram median", unit: "bpm", min: 0, max: 300 },
  { name: "Variance", label: "Histogram variance", unit: "", min: 0, max: 1000 },
  { name: "Tendency", label: "Histogram tendency", unit: "−1/0/1", min: -1, max: 1 },
];

export const CTG_STYLE: Record<CtgClass, string> = {
  Normal: "bg-emerald-600 text-white",
  Suspect: "bg-amber-500 text-black",
  Pathological: "bg-red-600 text-white",
};

interface Props {
  result: CtgPrediction | null;
  onResult: (r: CtgPrediction | null) => void;
}

export function CtgPanel({ result, onResult }: Props) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [examples, setExamples] = useState<CtgExample[]>([]);
  const [loaded, setLoaded] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCtgExamples()
      .then(setExamples)
      .catch(() => setExamples([]));
  }, []);

  const invalid = FIELDS.filter((f) => {
    const v = values[f.name];
    if (v === undefined || v.trim() === "") return true;
    const n = Number(v);
    return !Number.isFinite(n) || n < f.min || n > f.max;
  }).map((f) => f.name);
  const touched = Object.keys(values).length > 0;

  function loadExample(label: CtgClass) {
    const ex = examples.find((e) => e.label === label);
    if (!ex) return;
    setValues(Object.fromEntries(FIELDS.map((f) => [f.name, String(ex.features[f.name])])));
    setLoaded(`${ex.record} (expert label: ${ex.label}, held-out test record)`);
    onResult(null);
    setError(null);
  }

  async function classify() {
    if (invalid.length) return;
    setBusy(true);
    setError(null);
    try {
      onResult(await predictCtg(Object.fromEntries(FIELDS.map((f) => [f.name, Number(values[f.name])]))));
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not classify the CTG record.");
      onResult(null);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card data-testid="ctg-panel">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <Activity className="size-5 text-primary" /> CTG (cardiotocography), optional
        </CardTitle>
        <CardDescription>
          21 SisPorto features of one CTG record → Normal / Suspect / Pathological, with the features that drove the
          result (SHAP). The CTG model was trained on a different dataset and different patients from the ultrasound
          models.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <FlaskConical className="size-4 text-muted-foreground" />
          <span className="text-sm text-muted-foreground">Load example from the held-out test set:</span>
          {(["Normal", "Suspect", "Pathological"] as CtgClass[]).map((c) => (
            <Button
              key={c}
              size="sm"
              variant="outline"
              onClick={() => loadExample(c)}
              disabled={!examples.some((e) => e.label === c)}
            >
              {c}
            </Button>
          ))}
        </div>
        {loaded && <p className="text-xs text-muted-foreground">Loaded record {loaded}.</p>}

        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-7">
          {FIELDS.map((f) => {
            const bad = touched && invalid.includes(f.name);
            return (
              <div key={f.name} className="space-y-1">
                <Label htmlFor={`ctg-${f.name}`} className="text-xs" title={`${f.label}, ${f.min} to ${f.max}`}>
                  {f.name} <span className="text-muted-foreground">{f.unit}</span>
                </Label>
                <Input
                  id={`ctg-${f.name}`}
                  inputMode="decimal"
                  className={cn("h-8 text-sm", bad && "border-destructive")}
                  aria-invalid={bad}
                  placeholder={f.label}
                  value={values[f.name] ?? ""}
                  onChange={(e) => {
                    setValues((v) => ({ ...v, [f.name]: e.target.value }));
                    setLoaded(null);
                  }}
                />
              </div>
            );
          })}
        </div>
        {touched && invalid.length > 0 && (
          <p className="text-xs text-destructive">
            Missing or out-of-range: {invalid.join(", ")}. Hover a label for its allowed range.
          </p>
        )}

        <Button onClick={classify} disabled={busy || invalid.length > 0}>
          {busy && <Loader2 className="size-4 animate-spin" />} Classify CTG
        </Button>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="size-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && (
          <div className="space-y-3 rounded-lg border p-4" data-testid="ctg-result">
            <div className="flex flex-wrap items-center gap-3">
              <span className={cn("rounded px-2 py-0.5 text-sm font-bold", CTG_STYLE[result.predicted_class])}>
                {result.predicted_class.toUpperCase()}
              </span>
              <span className="text-sm text-muted-foreground">{result.model_version}</span>
            </div>
            <div className="space-y-1.5">
              {(Object.entries(result.probabilities) as [CtgClass, number][]).map(([c, p]) => (
                <div key={c} className="flex items-center gap-2 text-sm">
                  <span className="w-28">{c}</span>
                  <div className="h-2 flex-1 rounded-full bg-muted">
                    <div className={cn("h-2 rounded-full", CTG_STYLE[c])} style={{ width: `${p * 100}%` }} />
                  </div>
                  <span className="w-14 text-right tabular-nums">{(p * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
            <div>
              <p className="text-sm font-medium">Top factors (SHAP, toward {result.predicted_class})</p>
              <ul className="mt-1 space-y-1 text-sm">
                {result.top_factors.map((f) => (
                  <li key={f.feature} className="flex flex-wrap justify-between gap-2">
                    <span>
                      {f.description} ({f.feature} = {f.value}
                      {f.unit ? ` ${f.unit}` : ""})
                    </span>
                    <span className={cn("tabular-nums", f.shap > 0 ? "text-red-600" : "text-emerald-600")}>
                      {f.shap > 0 ? "+" : ""}
                      {f.shap.toFixed(2)} {f.direction}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
            <p className="text-xs text-muted-foreground">{result.note}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
