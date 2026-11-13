"use client";

import { useEffect, useRef, useState } from "react";

// Geometry of the sector (SVG user units). Apex at the top, like a curvilinear probe.
const W = 520;
const H = 440;
const APEX = { x: W / 2, y: 18 };
const R = 400;
const HALF = (38 * Math.PI) / 180;
const HEAD = { cx: W / 2, cy: 262, rx: 112, ry: 92, rot: -8 };

const pt = (a: number, r: number) => ({ x: APEX.x + r * Math.sin(a), y: APEX.y + r * Math.cos(a) });
const p1 = pt(-HALF, R);
const p2 = pt(HALF, R);
const SECTOR = `M${APEX.x},${APEX.y} L${p1.x},${p1.y} A${R},${R} 0 0 0 ${p2.x},${p2.y} Z`;

// Readouts replay a real analysis of 544_HC.png (HC18 held-out test image) by this system.
const READOUTS = [
  { k: "PLANE", v: "TRANS-THALAMIC", s: "P 0.72 ≥ 0.14", tone: "ok" },
  { k: "HC", v: "170.9 ± 1.6 mm", s: "ref 171.4 mm", tone: "echo" },
  { k: "CENTILE", v: "42nd @ 20+0 wk", s: "INTERGROWTH-21st", tone: "echo" },
  { k: "RELIABILITY", v: "OK · CV 0.90 %", s: "8 TTA passes", tone: "ok" },
];

/** Animated ultrasound speckle inside the sector (canvas, ~15 fps, paused off-screen / reduced motion). */
function Speckle() {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    const scale = 0.5; // render at half resolution, upscale with CSS: softer, cheaper speckle
    canvas.width = W * scale;
    canvas.height = H * scale;
    const img = ctx.createImageData(canvas.width, canvas.height);
    const cos = Math.cos((HEAD.rot * Math.PI) / 180);
    const sin = Math.sin((HEAD.rot * Math.PI) / 180);
    let raf = 0;
    let last = 0;
    let visible = true;
    const io = new IntersectionObserver(([e]) => (visible = e.isIntersecting));
    io.observe(canvas);

    const draw = (t: number) => {
      raf = requestAnimationFrame(draw);
      if (!visible || t - last < 66) return;
      last = t;
      const d = img.data;
      for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
          const X = x / scale;
          const Y = y / scale;
          // distance from the head ellipse boundary (skull = bright ring), interior = darker fluid/brain
          const dx = X - HEAD.cx;
          const dy = Y - HEAD.cy;
          const ex = (dx * cos + dy * sin) / HEAD.rx;
          const ey = (-dx * sin + dy * cos) / HEAD.ry;
          const rr = Math.sqrt(ex * ex + ey * ey);
          const depth = (Y - APEX.y) / R; // attenuation with depth
          let b = 38 + 50 * Math.random() * (1 - depth * 0.7);
          if (Math.abs(rr - 1) < 0.07) b += 150 * (1 - Math.abs(rr - 1) / 0.07) * (0.6 + 0.4 * Math.random());
          else if (rr < 1) b *= 0.55;
          // midline (falx) along the long axis, and the two hypoechoic thalami beside it (trans-thalamic plane)
          if (Math.abs(ey) < 0.03 && Math.abs(ex) < 0.88 && rr < 1) b += 70 * Math.random() + 30;
          const t1 = Math.hypot((ex - 0.05) / 0.2, (ey - 0.17) / 0.12);
          const t2 = Math.hypot((ex - 0.05) / 0.2, (ey + 0.17) / 0.12);
          if (t1 < 1 || t2 < 1) b *= 0.45;
          const i = (y * canvas.width + x) * 4;
          d[i] = b * 0.82;
          d[i + 1] = b;
          d[i + 2] = b * 0.98;
          d[i + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      if (reduce) cancelAnimationFrame(raf);
    };
    raf = requestAnimationFrame(draw);
    return () => {
      cancelAnimationFrame(raf);
      io.disconnect();
    };
  }, []);
  return <canvas ref={ref} className="absolute inset-0 h-full w-full opacity-80 [image-rendering:auto]" aria-hidden />;
}

export function HeroScan() {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (reduce) {
      setStep(READOUTS.length + 1);
      return;
    }
    // play the analysis, hold the finished state for ~10 s, then replay
    const id = setInterval(() => setStep((s) => (s > READOUTS.length + 12 ? 0 : s + 1)), 900);
    return () => clearInterval(id);
  }, []);
  const traced = step >= 2;

  return (
    <div className="relative mx-auto w-full max-w-[560px]" data-testid="hero-scan">
      {/* monitor frame */}
      <div className="panel glow-ring overflow-hidden p-3">
        <div className="mb-2 flex items-center justify-between px-1 font-mono text-[10px] uppercase tracking-[0.2em] text-muted-foreground">
          <span className="flex items-center gap-2">
            <span className="relative flex size-2">
              <span className="absolute inline-flex size-full animate-ping-slow rounded-full bg-primary opacity-60" />
              <span className="relative inline-flex size-2 rounded-full bg-primary" />
            </span>
            live analysis replay
          </span>
          <span>C5-1 · 3.5 MHz · TT plane</span>
        </div>
        <div className="relative aspect-[520/440] overflow-hidden rounded-xl bg-black">
          <svg viewBox={`0 0 ${W} ${H}`} className="absolute inset-0 h-full w-full">
            <defs>
              <clipPath id="sector">
                <path d={SECTOR} />
              </clipPath>
              <linearGradient id="beam" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0" stopColor="hsl(var(--echo))" stopOpacity="0.35" />
                <stop offset="1" stopColor="hsl(var(--echo))" stopOpacity="0" />
              </linearGradient>
            </defs>
            <foreignObject x="0" y="0" width={W} height={H} clipPath="url(#sector)">
              <div className="relative h-full w-full">
                <Speckle />
              </div>
            </foreignObject>
            {/* sweeping beam */}
            <g clipPath="url(#sector)">
              <g className="animate-sweep" style={{ transformOrigin: `${APEX.x}px ${APEX.y}px` }}>
                <path d={`M${APEX.x},${APEX.y} L${APEX.x - 22},${APEX.y + R} L${APEX.x + 22},${APEX.y + R} Z`} fill="url(#beam)" />
              </g>
            </g>
            <path d={SECTOR} fill="none" stroke="hsl(var(--echo) / 0.35)" strokeWidth="1" />
            {/* depth scale */}
            {[0.2, 0.4, 0.6, 0.8].map((f) => {
              const p = pt(HALF, R * f);
              return (
                <g key={f}>
                  <line x1={p.x + 6} y1={p.y} x2={p.x + 14} y2={p.y} stroke="hsl(var(--echo) / 0.6)" />
                  <text x={p.x + 18} y={p.y + 3} className="fill-[hsl(var(--echo)/0.7)] font-mono text-[9px]">
                    {Math.round(f * 16)}
                  </text>
                </g>
              );
            })}
            {/* fitted ellipse, traced */}
            {traced && (
              <g transform={`rotate(${HEAD.rot} ${HEAD.cx} ${HEAD.cy})`}>
                <ellipse
                  key={`e${step >= 2 ? 1 : 0}`}
                  cx={HEAD.cx}
                  cy={HEAD.cy}
                  rx={HEAD.rx}
                  ry={HEAD.ry}
                  fill="hsl(var(--primary) / 0.08)"
                  stroke="hsl(var(--warn))"
                  strokeWidth="2"
                  pathLength={1000}
                  className="animate-trace"
                  style={{ ["--len" as string]: 1000 }}
                />
                {/* calliper axes */}
                <line x1={HEAD.cx - HEAD.rx} y1={HEAD.cy} x2={HEAD.cx + HEAD.rx} y2={HEAD.cy} stroke="hsl(var(--echo))" strokeDasharray="3 4" strokeWidth="1" />
                <line x1={HEAD.cx} y1={HEAD.cy - HEAD.ry} x2={HEAD.cx} y2={HEAD.cy + HEAD.ry} stroke="hsl(var(--echo))" strokeDasharray="3 4" strokeWidth="1" />
                {[
                  [HEAD.cx - HEAD.rx, HEAD.cy],
                  [HEAD.cx + HEAD.rx, HEAD.cy],
                  [HEAD.cx, HEAD.cy - HEAD.ry],
                  [HEAD.cx, HEAD.cy + HEAD.ry],
                ].map(([x, y]) => (
                  <g key={`${x}-${y}`} stroke="hsl(var(--echo))" strokeWidth="1.5">
                    <line x1={x - 6} y1={y} x2={x + 6} y2={y} />
                    <line x1={x} y1={y - 6} x2={x} y2={y + 6} />
                  </g>
                ))}
              </g>
            )}
          </svg>
          {/* HUD readouts */}
          <div className="pointer-events-none absolute left-3 top-3 space-y-1.5">
            {READOUTS.map((r, i) => (
              <div
                key={r.k}
                className="rounded-md border border-white/10 bg-black/55 px-2 py-1 font-mono text-[10px] leading-tight text-white/90 backdrop-blur transition-all duration-500"
                style={{ opacity: step > i + 1 ? 1 : 0, transform: step > i + 1 ? "none" : "translateX(-8px)" }}
              >
                <span className="text-white/50">{r.k} </span>
                <span className={r.tone === "ok" ? "text-ok" : "text-echo"}>{r.v}</span>
                <div className="text-[9px] text-white/40">{r.s}</div>
              </div>
            ))}
          </div>
          <div className="pointer-events-none absolute bottom-2 right-3 font-mono text-[9px] text-white/40">
            544_HC.png · HC18 held-out test
          </div>
        </div>
      </div>
      <p className="mt-3 text-center font-mono text-[10px] text-muted-foreground">
        Illustrative render; the readouts are this system&apos;s real output for the image named above.
      </p>
    </div>
  );
}
