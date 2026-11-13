"use client";

import { ClipboardCheck } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ScreeningSummary } from "@/lib/api";
import { cn } from "@/lib/utils";

const LEVEL_STYLE: Record<ScreeningSummary["level"], string> = {
  Routine: "bg-emerald-600 text-white",
  "Review recommended": "bg-amber-500 text-black",
  "Urgent review": "bg-red-600 text-white",
  "Insufficient data": "bg-muted-foreground text-background",
};

/** Combined, rule-based screening summary (ultrasound + CTG). Also rendered in the printed report. */
export function ScreeningSummaryCard({ summary }: { summary: ScreeningSummary }) {
  return (
    <Card className="border-2" data-testid="screening-summary" data-level={summary.level}>
      <CardHeader className="pb-3">
        <CardTitle className="flex flex-wrap items-center gap-3 text-lg">
          <ClipboardCheck className="size-5 text-primary" />
          Screening summary
          <span className={cn("rounded px-2 py-0.5 text-sm font-bold", LEVEL_STYLE[summary.level])}>
            {summary.level}
          </span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-sm">
        {summary.drivers.length > 0 && (
          <div>
            <p className="font-medium">Driven by</p>
            <ul className="mt-1 list-disc space-y-0.5 pl-5">
              {summary.drivers.map((d) => (
                <li key={d.rule + d.input}>
                  <span className="font-medium">{d.input}:</span> {d.finding} → {d.level}{" "}
                  <span className="text-muted-foreground">({d.rule})</span>
                </li>
              ))}
            </ul>
          </div>
        )}
        {summary.unremarkable.length > 0 && (
          <div>
            <p className="font-medium">Unremarkable</p>
            <ul className="mt-1 list-disc space-y-0.5 pl-5 text-muted-foreground">
              {summary.unremarkable.map((d) => (
                <li key={d.input}>
                  {d.input}: {d.finding}
                </li>
              ))}
            </ul>
          </div>
        )}
        {summary.not_assessed.length > 0 && (
          <p className="text-muted-foreground">
            <span className="font-medium text-foreground">Not assessed:</span> {summary.not_assessed.join("; ")}.
          </p>
        )}
        <details className="text-xs text-muted-foreground print:hidden">
          <summary className="cursor-pointer">Rules ({summary.rules_version})</summary>
          <ul className="mt-1 list-disc pl-5">
            {summary.rules.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        </details>
        <div className="rounded-md border border-amber-500/50 bg-amber-500/5 p-2 text-xs">
          {summary.caveats.map((c) => (
            <p key={c}>{c}</p>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
