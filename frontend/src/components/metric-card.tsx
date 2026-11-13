import { cn } from "@/lib/utils";

interface Props {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  highlight?: boolean;
  tone?: "default" | "ok" | "warn" | "alarm";
}

const TONE = {
  default: "",
  ok: "text-ok",
  warn: "text-warn",
  alarm: "text-alarm",
};

/** Console readout: mono label, display-font value. `data-testid` is derived from the label (used by tests). */
export function MetricCard({ label, value, unit, hint, highlight, tone = "default" }: Props) {
  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-xl border bg-card/70 p-4",
        highlight && "border-primary/40 bg-primary/[0.06] glow-ring",
      )}
    >
      {highlight && <span className="absolute left-0 top-0 h-full w-0.5 bg-primary" aria-hidden />}
      <p className="readout">{label}</p>
      <p
        className={cn("mt-1.5 font-display text-xl font-semibold tabular-nums tracking-tight sm:text-2xl", TONE[tone])}
        data-testid={`metric-${label.toLowerCase().replace(/\s+/g, "-")}`}
      >
        {value}
        {unit && <span className="ml-1 text-sm font-normal text-muted-foreground">{unit}</span>}
      </p>
      {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}
