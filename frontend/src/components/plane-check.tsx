"use client";

import { CircleHelp, Loader2, ShieldAlert, ShieldCheck } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import type { PlaneCheck } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  check: PlaneCheck;
  withheld: boolean;
  onOverride?: () => void;
  overriding?: boolean;
}

const STYLE = {
  pass: {
    icon: ShieldCheck,
    label: "PASS",
    title: "Standard plane confirmed",
    box: "border-ok/40 bg-ok/[0.06]",
    pill: "bg-ok text-black",
    iconColor: "text-ok",
  },
  warn: {
    icon: ShieldAlert,
    label: "WARN",
    title: "Not a confirmed trans-thalamic plane",
    box: "border-warn/50 bg-warn/[0.06]",
    pill: "bg-warn text-black",
    iconColor: "text-warn",
  },
  unavailable: {
    icon: CircleHelp,
    label: "N/A",
    title: "Plane check unavailable",
    box: "border-muted bg-muted/30",
    pill: "bg-muted-foreground text-background",
    iconColor: "text-muted-foreground",
  },
} as const;

const pct = (v: number | null | undefined) => (v === null || v === undefined ? "—" : `${(v * 100).toFixed(0)}%`);

/** Plane-check badge shown BEFORE the HC result: plane, calibrated confidence, pass / warn. */
export function PlaneCheckCard({ check, withheld, onOverride, overriding }: Props) {
  const s = STYLE[check.status];
  const Icon = s.icon;
  const scorePct = check.tt_probability !== null ? Math.round(check.tt_probability * 100) : null;
  const thrPct = check.threshold !== null ? Math.round(check.threshold * 100) : null;
  return (
    <Card className={cn("rounded-2xl border bg-card/70", s.box)} data-testid="plane-check" data-status={check.status}>
      <CardContent className="space-y-3 p-4">
        <div className="flex flex-wrap items-center gap-3">
          <Icon className={cn("size-6 shrink-0", s.iconColor)} aria-hidden />
          <span className={cn("rounded px-2 py-0.5 text-xs font-bold tracking-wide", s.pill)}>{s.label}</span>
          <div className="min-w-0 flex-1">
            <p className="font-semibold leading-tight">Plane check · {s.title}</p>
            {check.plane && (
              <p className="text-sm text-muted-foreground">
                {check.plane} {pct(check.plane_confidence)}
                {check.brain_subplane && ` · ${check.brain_subplane} ${pct(check.brain_subplane_confidence)}`}
              </p>
            )}
          </div>
        </div>

        {scorePct !== null && thrPct !== null && (
          <div>
            <div className="mb-1 flex justify-between text-xs text-muted-foreground">
              <span>P(trans-thalamic), calibrated: {scorePct}%</span>
              <span>pass threshold {thrPct}%</span>
            </div>
            <div className="relative h-2 rounded-full bg-muted" aria-hidden>
              <div
                className={cn("h-2 rounded-full", check.status === "pass" ? "bg-ok" : "bg-warn")}
                style={{ width: `${scorePct}%` }}
              />
              <div className="absolute -top-1 h-4 w-0.5 bg-foreground" style={{ left: `${thrPct}%` }} />
            </div>
          </div>
        )}

        <p className="text-sm">{check.message}</p>

        {withheld && onOverride && (
          <div className="flex flex-wrap items-center gap-3 print:hidden">
            <Button size="sm" variant="outline" onClick={onOverride} disabled={overriding}>
              {overriding && <Loader2 className="size-4 animate-spin" />} Measure anyway (override)
            </Button>
            <span className="text-xs text-muted-foreground">
              HC is withheld until you override; the warning stays on the report.
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
