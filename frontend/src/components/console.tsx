import { cn } from "@/lib/utils";

/** Page header used by the workspace pages: code, kicker, title, description, optional right-hand slot. */
export function PageHeader({
  code,
  kicker,
  title,
  children,
  right,
}: {
  code: string;
  kicker: string;
  title: string;
  children?: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div className="relative overflow-hidden border-b print:hidden">
      <div className="bg-grid bg-grid-fade pointer-events-none absolute inset-0 opacity-70" aria-hidden />
      <div className="relative mx-auto flex max-w-7xl flex-wrap items-end justify-between gap-6 px-4 py-10">
        <div className="max-w-2xl">
          <p className="readout">
            <span className="text-primary">{code}</span> · {kicker}
          </p>
          <h1 className="mt-2 font-display text-3xl font-semibold tracking-tight sm:text-4xl">{title}</h1>
          {children && <p className="mt-2 text-muted-foreground">{children}</p>}
        </div>
        {right}
      </div>
    </div>
  );
}

/** Corner brackets drawn over a monitor-like frame. */
export function CornerBrackets({ className }: { className?: string }) {
  const c = "absolute size-4 border-primary/70";
  return (
    <span className={cn("pointer-events-none absolute inset-2 z-10", className)} aria-hidden>
      <span className={cn(c, "left-0 top-0 border-l-2 border-t-2")} />
      <span className={cn(c, "right-0 top-0 border-r-2 border-t-2")} />
      <span className={cn(c, "bottom-0 left-0 border-b-2 border-l-2")} />
      <span className={cn(c, "bottom-0 right-0 border-b-2 border-r-2")} />
    </span>
  );
}

/** Section label inside a panel. */
export function PanelTitle({ icon, children, right }: { icon?: React.ReactNode; children: React.ReactNode; right?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-3 border-b px-4 py-3">
      <p className="flex items-center gap-2 font-display text-sm font-semibold">
        {icon}
        {children}
      </p>
      {right}
    </div>
  );
}

export type StageState = "idle" | "ready" | "running" | "done" | "warn" | "skip";

const DOT: Record<StageState, string> = {
  idle: "border-muted-foreground/30 bg-transparent",
  ready: "border-primary/50 bg-transparent",
  running: "border-primary bg-primary/40 animate-pulse",
  done: "border-ok bg-ok shadow-[0_0_10px_hsl(var(--ok))]",
  warn: "border-warn bg-warn shadow-[0_0_10px_hsl(var(--warn))]",
  skip: "border-muted-foreground/40 bg-muted",
};

/** Horizontal pipeline rail: each stage lights up as its result arrives. */
export function StageRail({ stages }: { stages: { label: string; state: StageState; note?: string }[] }) {
  return (
    <ol className="flex w-full items-start gap-0 overflow-x-auto" data-testid="stage-rail">
      {stages.map((s, i) => (
        <li key={s.label} className="flex min-w-[92px] flex-1 flex-col items-center text-center" data-state={s.state}>
          <div className="flex w-full items-center">
            <span className={cn("h-px flex-1", i === 0 ? "bg-transparent" : s.state === "idle" ? "bg-border" : "bg-primary/50")} />
            <span className={cn("size-3 rounded-full border-2 transition-all duration-500", DOT[s.state])} />
            <span
              className={cn(
                "h-px flex-1",
                i === stages.length - 1 ? "bg-transparent" : stages[i + 1].state === "idle" ? "bg-border" : "bg-primary/50",
              )}
            />
          </div>
          <span className="mt-2 font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{s.label}</span>
          {s.note && <span className="text-[10px] text-muted-foreground/80">{s.note}</span>}
        </li>
      ))}
    </ol>
  );
}
