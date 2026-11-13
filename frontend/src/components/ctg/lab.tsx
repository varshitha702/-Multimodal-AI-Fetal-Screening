"use client";

import { Activity, AlertCircle, ArrowRight, FlaskConical, HeartPulse, Loader2, TrendingDown, Waves } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { PanelTitle } from "@/components/console";
import { Input } from "@/components/ui/input";
import { ApiError, getCtgExamples, predictCtg, type CtgClass, type CtgExample, type CtgPrediction } from "@/lib/api";
import { useSession } from "@/lib/session";
import { cn } from "@/lib/utils";

type Field = { name: string; label: string; unit: string; min: number; max: number };

// Mirrors ml/ctg.py FEATURES (the backend validates again and returns 422 on bad input), grouped by physiology.
const GROUPS: { title: string; icon: React.ElementType; blurb: string; fields: Field[] }[] = [
  {
    title: "Heart rate",
    icon: HeartPulse,
    blurb: "Baseline and the FHR histogram over the record",
    fields: [
      { name: "LB", label: "FHR baseline", unit: "bpm", min: 50, max: 240 },
      { name: "Min", label: "Histogram min", unit: "bpm", min: 0, max: 250 },
      { name: "Max", label: "Histogram max", unit: "bpm", min: 0, max: 300 },
      { name: "Mode", label: "Histogram mode", unit: "bpm", min: 0, max: 300 },
      { name: "Mean", label: "Histogram mean", unit: "bpm", min: 0, max: 300 },
      { name: "Median", label: "Histogram median", unit: "bpm", min: 0, max: 300 },
      { name: "Width", label: "Histogram width", unit: "bpm", min: 0, max: 250 },
      { name: "Variance", label: "Histogram variance", unit: "", min: 0, max: 1000 },
      { name: "Nmax", label: "Histogram peaks", unit: "", min: 0, max: 50 },
      { name: "Nzeros", label: "Histogram zeros", unit: "", min: 0, max: 50 },
      { name: "Tendency", label: "Histogram tendency", unit: "−1/0/1", min: -1, max: 1 },
    ],
  },
  {
    title: "Activity",
    icon: Activity,
    blurb: "Events per second",
    fields: [
      { name: "AC", label: "Accelerations", unit: "/s", min: 0, max: 1 },
      { name: "FM", label: "Fetal movements", unit: "/s", min: 0, max: 1 },
      { name: "UC", label: "Uterine contractions", unit: "/s", min: 0, max: 1 },
    ],
  },
  {
    title: "Decelerations",
    icon: TrendingDown,
    blurb: "Events per second",
    fields: [
      { name: "DL", label: "Light decelerations", unit: "/s", min: 0, max: 1 },
      { name: "DS", label: "Severe decelerations", unit: "/s", min: 0, max: 1 },
      { name: "DP", label: "Prolonged decelerations", unit: "/s", min: 0, max: 1 },
    ],
  },
  {
    title: "Variability",
    icon: Waves,
    blurb: "Short- and long-term variability",
    fields: [
      { name: "ASTV", label: "Abnormal STV time", unit: "%", min: 0, max: 100 },
      { name: "MSTV", label: "Mean STV", unit: "bpm", min: 0, max: 20 },
      { name: "ALTV", label: "Abnormal LTV time", unit: "%", min: 0, max: 100 },
      { name: "MLTV", label: "Mean LTV", unit: "bpm", min: 0, max: 100 },
    ],
  },
];
const FIELDS = GROUPS.flatMap((g) => g.fields);
const ORDER = ["LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV", "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean", "Median", "Variance", "Tendency"];

const CLASS_COLOR: Record<CtgClass, string> = {
  Normal: "hsl(var(--ok))",
  Suspect: "hsl(var(--warn))",
  Pathological: "hsl(var(--alarm))",
};

/** Semicircle gauge: three arcs sized by class probability, needle at the predicted class. */
function ProbabilityGauge({ result }: { result: CtgPrediction }) {
  const classes: CtgClass[] = ["Normal", "Suspect", "Pathological"];
  const R = 80;
  const C = Math.PI * R;
  let offset = 0;
  const arcs = classes.map((c) => {
    const len = result.probabilities[c] * C;
    const a = { c, len, offset };
    offset += len;
    return a;
  });
  // needle: middle of the predicted class arc
  const pred = arcs.find((a) => a.c === result.predicted_class)!;
  const frac = (pred.offset + pred.len / 2) / C;
  const ang = Math.PI * (1 - frac);
  return (
    <svg viewBox="0 0 200 116" className="mx-auto w-full max-w-[260px]" role="img"
      aria-label={`Predicted ${result.predicted_class}: ${(result.probabilities[result.predicted_class] * 100).toFixed(1)}%`}>
      <path d={`M20 100 A${R} ${R} 0 0 1 180 100`} fill="none" stroke="hsl(var(--muted))" strokeWidth="14" />
      {arcs.map((a) => (
        <path key={a.c} d={`M20 100 A${R} ${R} 0 0 1 180 100`} fill="none" stroke={CLASS_COLOR[a.c]} strokeWidth="14"
          strokeDasharray={`${a.len} ${C}`} strokeDashoffset={-a.offset} className="transition-all duration-700" />
      ))}
      <line x1="100" y1="100" x2={100 + 62 * Math.cos(ang)} y2={100 - 62 * Math.sin(ang)} stroke="hsl(var(--foreground))" strokeWidth="2.5" strokeLinecap="round" className="transition-all duration-700" />
      <circle cx="100" cy="100" r="5" fill="hsl(var(--foreground))" />
    </svg>
  );
}

/** Diverging SHAP bars: right (warm) pushes toward the predicted class, left (cool) away from it. */
function ShapChart({ result }: { result: CtgPrediction }) {
  const max = Math.max(...result.top_factors.map((f) => Math.abs(f.shap)), 0.01);
  return (
    <div className="space-y-2.5" data-testid="shap-chart">
      {result.top_factors.map((f) => {
        const w = (Math.abs(f.shap) / max) * 50;
        const pos = f.shap > 0;
        return (
          <div key={f.feature} className="grid grid-cols-[110px_1fr_56px] items-center gap-3 text-xs">
            <span className="truncate text-right" title={f.description}>
              <span className="font-mono text-foreground">{f.feature}</span>{" "}
              <span className="text-muted-foreground">= {Number(f.value.toPrecision(4))}</span>
            </span>
            <div className="relative h-5 rounded bg-muted/40">
              <span className="absolute inset-y-0 left-1/2 w-px bg-border" />
              <span
                className={cn("absolute inset-y-1 rounded-sm transition-all duration-700", pos ? "bg-alarm/80" : "bg-ok/80")}
                style={pos ? { left: "50%", width: `${w}%` } : { right: "50%", width: `${w}%` }}
              />
            </div>
            <span className={cn("font-mono tabular-nums", pos ? "text-alarm" : "text-ok")}>
              {pos ? "+" : ""}
              {f.shap.toFixed(2)}
            </span>
          </div>
        );
      })}
      <div className="grid grid-cols-[110px_1fr_56px] gap-3 font-mono text-[10px] text-muted-foreground">
        <span />
        <span className="flex justify-between">
          <span>← away from {result.predicted_class}</span>
          <span>towards →</span>
        </span>
        <span />
      </div>
    </div>
  );
}

export function CtgLab() {
  const { ctg, setCtg, scan, summary } = useSession();
  const { values, loaded, result } = ctg;
  const [examples, setExamples] = useState<CtgExample[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getCtgExamples().then(setExamples).catch(() => setExamples([]));
  }, []);

  const invalid = FIELDS.filter((f) => {
    const v = values[f.name];
    if (v === undefined || v.trim() === "") return true;
    const n = Number(v);
    return !Number.isFinite(n) || n < f.min || n > f.max;
  }).map((f) => f.name);
  const invalidOrdered = ORDER.filter((n) => invalid.includes(n));
  const touched = Object.keys(values).length > 0;

  function loadExample(label: CtgClass) {
    const ex = examples.find((e) => e.label === label);
    if (!ex) return;
    setCtg({
      values: Object.fromEntries(FIELDS.map((f) => [f.name, String(ex.features[f.name])])),
      loaded: `${ex.record} (expert label: ${ex.label}, held-out test record)`,
      result: null,
    });
    setError(null);
  }

  async function classify() {
    if (invalid.length) return;
    setBusy(true);
    setError(null);
    try {
      setCtg({ result: await predictCtg(Object.fromEntries(FIELDS.map((f) => [f.name, Number(values[f.name])]))) });
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not classify the CTG record.");
      setCtg({ result: null });
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto grid max-w-7xl gap-6 px-4 py-6 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]" data-testid="ctg-panel">
      {/* ------------ input bench ------------ */}
      <section className="space-y-4" aria-label="CTG features">
        <div className="panel p-4">
          <div className="flex flex-wrap items-center gap-3">
            <FlaskConical className="size-4 text-primary" />
            <span className="text-sm">Load example from the held-out test set:</span>
            <div className="flex gap-2">
              {(["Normal", "Suspect", "Pathological"] as CtgClass[]).map((c) => (
                <button
                  key={c}
                  type="button"
                  onClick={() => loadExample(c)}
                  disabled={!examples.some((e) => e.label === c)}
                  className="rounded-md border px-3 py-1.5 text-xs font-medium transition-colors hover:border-primary/60 disabled:opacity-40"
                  style={{ boxShadow: `inset 3px 0 0 ${CLASS_COLOR[c]}` }}
                >
                  {c}
                </button>
              ))}
            </div>
          </div>
          {loaded && <p className="mt-2 font-mono text-[11px] text-muted-foreground">Loaded record {loaded}.</p>}
        </div>

        <div className="grid gap-4 md:grid-cols-2">
          {GROUPS.map((g) => {
            const I = g.icon;
            return (
              <div key={g.title} className={cn("panel", g.title === "Heart rate" && "md:row-span-3")}>
                <PanelTitle icon={<I className="size-4 text-primary" />} right={<span className="readout">{g.blurb}</span>}>
                  {g.title}
                </PanelTitle>
                <div className="grid grid-cols-2 gap-3 p-4">
                  {g.fields.map((f) => {
                    const bad = touched && invalid.includes(f.name);
                    return (
                      <label key={f.name} htmlFor={`ctg-${f.name}`} className="space-y-1" title={`${f.label}: ${f.min} to ${f.max}`}>
                        <span className="flex items-baseline justify-between text-xs">
                          <span className="font-mono text-foreground">{f.name}</span>
                          <span className="text-[10px] text-muted-foreground">{f.unit}</span>
                        </span>
                        <Input
                          id={`ctg-${f.name}`}
                          inputMode="decimal"
                          className={cn("h-8 font-mono text-xs", bad && "border-destructive")}
                          aria-invalid={bad}
                          placeholder={f.label}
                          value={values[f.name] ?? ""}
                          onChange={(e) => setCtg({ values: { ...values, [f.name]: e.target.value }, loaded: null })}
                        />
                      </label>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex flex-wrap items-center gap-4">
          <button
            type="button"
            onClick={classify}
            disabled={busy || invalid.length > 0}
            className="inline-flex h-11 items-center gap-2 rounded-lg bg-primary px-6 font-medium text-primary-foreground shadow-[0_0_24px_-6px_hsl(var(--primary))] disabled:cursor-not-allowed disabled:opacity-50 disabled:shadow-none"
          >
            {busy && <Loader2 className="size-4 animate-spin" />} Classify CTG
          </button>
          {touched && invalid.length > 0 && (
            <p className="text-xs text-destructive">Missing or out-of-range: {invalidOrdered.join(", ")}. Hover a field for its range.</p>
          )}
        </div>
        {error && (
          <div role="alert" className="flex gap-2 rounded-xl border border-destructive/50 bg-destructive/10 p-3 text-sm">
            <AlertCircle className="size-4 text-destructive" /> {error}
          </div>
        )}
      </section>

      {/* ------------ result ------------ */}
      <aside className="space-y-4 lg:sticky lg:top-20 lg:h-fit">
        {result ? (
          <div className="panel grain overflow-hidden" data-testid="ctg-result">
            <div className="p-5 text-center">
              <p className="readout">NSP classification</p>
              <ProbabilityGauge result={result} />
              <p className="-mt-2 font-display text-3xl font-semibold tracking-tight" style={{ color: CLASS_COLOR[result.predicted_class] }}>
                {result.predicted_class.toUpperCase()}
              </p>
              <p className="font-mono text-xs text-muted-foreground">
                p = {(result.probabilities[result.predicted_class] * 100).toFixed(1)}% · {result.model_version}
              </p>
              <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
                {(Object.entries(result.probabilities) as [CtgClass, number][]).map(([c, p]) => (
                  <div key={c} className="rounded-lg border p-2">
                    <span className="mx-auto mb-1 block h-1 w-6 rounded-full" style={{ background: CLASS_COLOR[c] }} />
                    <span className="block text-muted-foreground">{c}</span>
                    <span className="font-mono">{(p * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="border-t p-5">
              <p className="mb-3 text-sm font-medium">Top factors (SHAP, toward {result.predicted_class})</p>
              <ShapChart result={result} />
              <p className="mt-4 text-xs text-muted-foreground">{result.note}</p>
            </div>
          </div>
        ) : (
          <div className="panel p-6">
            <p className="readout">Result</p>
            <p className="mt-3 font-display text-xl font-semibold">Waiting for a CTG record</p>
            <p className="mt-2 text-sm text-muted-foreground">
              Load a held-out example or enter the 21 SisPorto features of a record, then classify. The model returns
              Normal, Suspect or Pathological with the features that drove the decision.
            </p>
            <svg viewBox="0 0 300 60" className="mt-6 w-full text-primary/60" aria-hidden>
              <path d="M0 30 H80 l8 -3 l6 3 h14 l6 -18 l8 36 l7 -26 l5 8 h40 H300" fill="none" stroke="currentColor" strokeWidth="1.5" className="animate-ecg" />
            </svg>
          </div>
        )}
        <Link href="/report" className="panel group flex items-center justify-between p-4 transition-colors hover:border-primary/50">
          <span>
            <span className="readout block">Combined report</span>
            <span className="font-display font-semibold">{summary ? `Current level: ${summary.level}` : "Open the screening report"}</span>
            <span className="block text-xs text-muted-foreground">
              Ultrasound: {scan.result ? "analysed" : "not yet"} · CTG: {result ? result.predicted_class : "not yet"}
            </span>
          </span>
          <ArrowRight className="size-4 text-primary transition-transform group-hover:translate-x-1" />
        </Link>
      </aside>
    </div>
  );
}
