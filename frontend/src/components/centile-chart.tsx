"use client";

import { useEffect, useState } from "react";

import { getCentileCurves, type CentileCurves, type Growth } from "@/lib/api";

const W = 640;
const H = 300;
const PAD = { l: 48, r: 40, t: 12, b: 36 };
const GA_MIN = 14;
const GA_MAX = 40;
const HC_MIN = 50;
const HC_MAX = 400;

const x = (ga: number) => PAD.l + ((ga - GA_MIN) / (GA_MAX - GA_MIN)) * (W - PAD.l - PAD.r);
const y = (hc: number) => PAD.t + (1 - (hc - HC_MIN) / (HC_MAX - HC_MIN)) * (H - PAD.t - PAD.b);

/** INTERGROWTH-21st HC centile curves (from the API) with the measured point. */
export function CentileChart({ growth, hcMm }: { growth: Growth; hcMm: number }) {
  const [curves, setCurves] = useState<CentileCurves | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    getCentileCurves()
      .then(setCurves)
      .catch(() => setFailed(true));
  }, []);

  if (failed) return <p className="text-sm text-muted-foreground">Centile curves could not be loaded.</p>;
  if (!curves) return <div className="h-[300px] animate-pulse rounded-md bg-primary/5" />;

  const line = (key: "p3" | "p10" | "p50" | "p90" | "p97") =>
    curves.points.map((p, i) => `${i ? "L" : "M"}${x(p.ga_weeks).toFixed(1)},${y(p[key]).toFixed(1)}`).join(" ");
  const band =
    curves.points.map((p, i) => `${i ? "L" : "M"}${x(p.ga_weeks).toFixed(1)},${y(p.p97).toFixed(1)}`).join(" ") +
    " " +
    [...curves.points]
      .reverse()
      .map((p) => `L${x(p.ga_weeks).toFixed(1)},${y(p.p3).toFixed(1)}`)
      .join(" ") +
    " Z";
  const px = x(growth.gestational_age_weeks);
  const py = y(Math.min(Math.max(hcMm, HC_MIN), HC_MAX));

  return (
    <figure data-testid="centile-chart">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img"
        aria-label={`Measured HC ${hcMm.toFixed(1)} mm at ${growth.gestational_age_weeks.toFixed(1)} weeks on INTERGROWTH-21st centile curves`}>
        {[100, 150, 200, 250, 300, 350].map((v) => (
          <g key={v}>
            <line x1={PAD.l} x2={W - PAD.r} y1={y(v)} y2={y(v)} className="stroke-border" strokeWidth={1} />
            <text x={PAD.l - 8} y={y(v) + 4} textAnchor="end" className="fill-muted-foreground text-[11px]">{v}</text>
          </g>
        ))}
        {[14, 18, 22, 26, 30, 34, 38, 40].map((g) => (
          <text key={g} x={x(g)} y={H - PAD.b + 18} textAnchor="middle" className="fill-muted-foreground text-[11px]">{g}</text>
        ))}
        <text x={(PAD.l + W - PAD.r) / 2} y={H - 4} textAnchor="middle" className="fill-muted-foreground text-[11px]">
          gestational age (weeks)
        </text>
        <text x={12} y={(PAD.t + H - PAD.b) / 2} textAnchor="middle" transform={`rotate(-90 12 ${(PAD.t + H - PAD.b) / 2})`}
          className="fill-muted-foreground text-[11px]">HC (mm)</text>
        <path d={band} className="fill-primary/10" />
        <path d={line("p3")} className="fill-none stroke-primary/60" strokeWidth={1.2} strokeDasharray="4 3" />
        <path d={line("p97")} className="fill-none stroke-primary/60" strokeWidth={1.2} strokeDasharray="4 3" />
        <path d={line("p10")} className="fill-none stroke-primary/35" strokeWidth={1} />
        <path d={line("p90")} className="fill-none stroke-primary/35" strokeWidth={1} />
        <path d={line("p50")} className="fill-none stroke-primary" strokeWidth={2} />
        {(["p97", "p50", "p3"] as const).map((k) => (
          <text key={k} x={x(GA_MAX) + 4} y={y(curves.points.at(-1)![k]) + 4} className="fill-muted-foreground text-[10px]">
            {k === "p97" ? "97th" : k === "p50" ? "50th" : "3rd"}
          </text>
        ))}
        <circle cx={px} cy={py} r={7} className={growth.within_p3_p97 ? "fill-emerald-500" : "fill-amber-500"} stroke="white" strokeWidth={2} />
      </svg>
      <figcaption className="mt-2 text-xs text-muted-foreground">
        Shaded band: 3rd–97th centile. Lines: 3rd, 10th, 50th, 90th, 97th centiles. {curves.reference}.
      </figcaption>
    </figure>
  );
}
