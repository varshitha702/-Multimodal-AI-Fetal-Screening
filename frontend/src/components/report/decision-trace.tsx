"use client";

import { Activity, ArrowRight, Crosshair, GitMerge, Printer, ShieldCheck, Sigma } from "lucide-react";
import Link from "next/link";

import { dataUrl, formatGa, type ScreeningSummary } from "@/lib/api";
import { useSession } from "@/lib/session";
import { cn } from "@/lib/utils";

const LEVEL = {
  Routine: { color: "hsl(var(--ok))", text: "text-ok", ring: "border-ok/40", blurb: "Everything that was assessed is unremarkable." },
  "Review recommended": { color: "hsl(var(--warn))", text: "text-warn", ring: "border-warn/50", blurb: "At least one input needs a second look." },
  "Urgent review": { color: "hsl(var(--alarm))", text: "text-alarm", ring: "border-alarm/50", blurb: "A pathological CTG always escalates." },
  "Insufficient data": { color: "hsl(var(--muted-foreground))", text: "text-muted-foreground", ring: "border-border", blurb: "No input has been assessed yet." },
} as const;

function Verdict({ s }: { s: ScreeningSummary }) {
  const l = LEVEL[s.level];
  return (
    <div className={cn("panel relative overflow-hidden p-8", l.ring)} data-testid="screening-summary" data-level={s.level}>
      <div className="pointer-events-none absolute -right-24 -top-24 size-72 rounded-full opacity-25 blur-3xl" style={{ background: l.color }} />
      <div className="relative flex flex-wrap items-center gap-6">
        <span className="relative grid size-16 place-items-center">
          <span className="absolute inset-0 animate-ping-slow rounded-full opacity-40" style={{ background: l.color }} />
          <span className="relative size-6 rounded-full" style={{ background: l.color, boxShadow: `0 0 24px ${l.color}` }} />
        </span>
        <div className="min-w-0 flex-1">
          <p className="readout">Screening summary</p>
          <h2 className={cn("font-display text-4xl font-semibold tracking-tight sm:text-5xl", l.text)}>{s.level}</h2>
          <p className="mt-1 text-muted-foreground">{l.blurb}</p>
        </div>
        <button
          type="button"
          onClick={() => window.print()}
          className="inline-flex items-center gap-2 rounded-lg border px-4 py-2 text-sm hover:border-primary/50 print:hidden"
        >
          <Printer className="size-4" /> Print
        </button>
      </div>
      {s.drivers.length > 0 && (
        <div className="relative mt-6 border-t pt-4">
          <p className="readout mb-2">Driven by</p>
          <ul className="space-y-1 text-sm">
            {s.drivers.map((d) => (
              <li key={d.rule + d.input}>
                <span className="font-mono text-xs text-muted-foreground">{d.rule}</span>{" "}
                <span className="font-medium">{d.input}:</span> {d.finding} → {d.level}
              </li>
            ))}
          </ul>
        </div>
      )}
      {s.not_assessed.length > 0 && (
        <p className="relative mt-3 text-sm text-muted-foreground">
          <span className="font-medium text-foreground">Not assessed:</span> {s.not_assessed.join("; ")}.
        </p>
      )}
    </div>
  );
}

function InputTiles() {
  const { scan, ctg } = useSession();
  const r = scan.result;
  const tiles = [
    {
      icon: ShieldCheck,
      label: "Plane check",
      value: r ? r.plane_check.status.toUpperCase() : "—",
      sub: r ? (r.plane_check.plane ?? "unavailable") : "no ultrasound yet",
      tone: r ? (r.plane_check.status === "pass" ? "ok" : r.plane_check.status === "warn" ? "warn" : "") : "",
      href: "/scan",
    },
    {
      icon: Crosshair,
      label: "HC centile",
      value: r?.growth ? `${Math.round(r.growth.centile)}th` : "—",
      sub: r?.growth
        ? `z ${r.growth.z_score >= 0 ? "+" : ""}${r.growth.z_score.toFixed(2)} at ${formatGa(r.growth.gestational_age_weeks)} wk`
        : r
          ? "needs HC in mm and GA"
          : "no ultrasound yet",
      tone: r?.growth ? (r.growth.within_p3_p97 ? "ok" : "warn") : "",
      href: "/scan",
    },
    {
      icon: Sigma,
      label: "Reliability",
      value: r?.uncertainty ? (r.uncertainty.low_reliability ? "LOW" : "OK") : "—",
      sub: r?.uncertainty?.hc_sd_mm != null ? `HC SD ${r.uncertainty.hc_sd_mm.toFixed(1)} mm` : r ? "TTA not available" : "no ultrasound yet",
      tone: r?.uncertainty ? (r.uncertainty.low_reliability ? "warn" : "ok") : "",
      href: "/scan",
    },
    {
      icon: Activity,
      label: "CTG",
      value: ctg.result ? ctg.result.predicted_class.toUpperCase() : "—",
      sub: ctg.result ? `p ${(ctg.result.probabilities[ctg.result.predicted_class] * 100).toFixed(1)}%` : "no CTG yet",
      tone: ctg.result ? (ctg.result.predicted_class === "Normal" ? "ok" : ctg.result.predicted_class === "Suspect" ? "warn" : "alarm") : "",
      href: "/ctg",
    },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
      {tiles.map((t) => {
        const I = t.icon;
        const missing = t.value === "—";
        return (
          <Link
            key={t.label}
            href={t.href}
            className={cn("panel group p-4 transition-colors hover:border-primary/50", missing && "border-dashed")}
          >
            <div className="flex items-center justify-between">
              <I className="size-4 text-primary" />
              {missing && <span className="readout text-[9px]">assess →</span>}
            </div>
            <p className="readout mt-3">{t.label}</p>
            <p className={cn("font-display text-2xl font-semibold", t.tone && `text-${t.tone}`)}>{t.value}</p>
            <p className="text-xs text-muted-foreground">{t.sub}</p>
          </Link>
        );
      })}
    </div>
  );
}

function RuleLadder({ s }: { s: ScreeningSummary }) {
  const fired = new Set(s.drivers.map((d) => d.rule));
  if (s.level === "Routine") fired.add("R6");
  return (
    <div className="panel p-5" data-testid="rule-ladder">
      <p className="flex items-center gap-2 font-display text-sm font-semibold">
        <GitMerge className="size-4 text-primary" /> Decision trace ({s.rules_version})
      </p>
      <ol className="relative mt-5 space-y-2">
        <span className="absolute bottom-3 left-[15px] top-3 w-px bg-border" aria-hidden />
        {s.rules.map((rule) => {
          const id = rule.split(" ")[0];
          const on = fired.has(id);
          const [cond, outcome] = rule.slice(id.length + 1).split(" -> ");
          return (
            <li key={id} className="relative flex items-start gap-3" data-fired={on}>
              <span
                className={cn(
                  "relative z-10 grid size-8 shrink-0 place-items-center rounded-full border font-mono text-[10px] transition-all",
                  on ? "border-primary bg-primary text-primary-foreground glow-ring" : "bg-card text-muted-foreground",
                )}
              >
                {id}
              </span>
              <div className={cn("flex-1 rounded-lg border px-3 py-2 text-sm transition-colors", on ? "border-primary/40 bg-primary/[0.06]" : "border-transparent opacity-55")}>
                <span>{cond}</span>
                {outcome && (
                  <span className="mt-0.5 flex items-center gap-1 text-xs text-muted-foreground">
                    <ArrowRight className="size-3" /> {outcome}
                  </span>
                )}
              </div>
            </li>
          );
        })}
      </ol>
      <p className="mt-4 text-xs text-muted-foreground">Most severe level wins. Highlighted rules fired for this session.</p>
    </div>
  );
}

export function ReportView() {
  const { scan, ctg, summary, reset } = useSession();
  if (!summary) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-10">
        <div className="panel grid place-items-center gap-4 px-6 py-16 text-center" data-testid="report-empty">
          <GitMerge className="size-8 text-primary" />
          <p className="font-display text-2xl font-semibold">Nothing to combine yet</p>
          <p className="max-w-md text-muted-foreground">
            Analyse an ultrasound and/or classify a CTG record. The report combines whatever is available and lists what
            was not assessed.
          </p>
          <div className="flex gap-3">
            <Link href="/scan" className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground">Start a scan</Link>
            <Link href="/ctg" className="rounded-lg border px-4 py-2 text-sm">Classify a CTG</Link>
          </div>
        </div>
      </div>
    );
  }
  const r = scan.result;
  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-6">
      <Verdict s={summary} />
      <InputTiles />
      <div className="grid gap-6 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
        <RuleLadder s={summary} />
        <div className="space-y-4">
          {r && scan.file && (
            <div className="panel overflow-hidden">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={dataUrl(r.overlay_png_base64)} alt="Ultrasound overlay" className="aspect-[4/3] w-full bg-black object-contain" />
              <div className="flex items-center justify-between border-t px-4 py-2 font-mono text-xs">
                <span className="truncate">{scan.file.name}</span>
                <span className="text-echo">
                  {r.head_circumference_mm !== null ? `HC ${r.head_circumference_mm.toFixed(1)}${r.uncertainty?.hc_sd_mm != null ? ` ± ${r.uncertainty.hc_sd_mm.toFixed(1)}` : ""} mm` : "HC withheld"}
                </span>
              </div>
            </div>
          )}
          {ctg.result && (
            <div className="panel p-4 text-sm">
              <p className="readout">CTG top factors</p>
              <ul className="mt-2 space-y-1">
                {ctg.result.top_factors.slice(0, 3).map((f) => (
                  <li key={f.feature} className="flex justify-between gap-2">
                    <span>{f.description}</span>
                    <span className={cn("font-mono", f.shap > 0 ? "text-alarm" : "text-ok")}>{f.shap > 0 ? "+" : ""}{f.shap.toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <div className="rounded-2xl border border-warn/40 bg-warn/[0.06] p-4 text-xs leading-relaxed">
            {summary.caveats.map((c) => (
              <p key={c} className="mb-1 last:mb-0">{c}</p>
            ))}
          </div>
          <button type="button" onClick={reset} className="text-xs text-muted-foreground hover:text-foreground print:hidden">
            Clear this session
          </button>
        </div>
      </div>
    </div>
  );
}
