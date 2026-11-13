"use client";

import { Activity, ArrowRight, Crosshair, GitMerge, ScanLine, ShieldCheck, Sigma, Waves } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { spotlightHandler, useCountUp, useInView, useReveal } from "@/lib/hooks";
import { cn } from "@/lib/utils";

// Every number on this page comes from outputs/*.json in the repository (held-out evaluations).
const STAGES = [
  {
    id: "plane",
    icon: ShieldCheck,
    title: "Plane gate",
    tag: "EfficientNet-B0 · 2 heads",
    body: "Before anything is measured, a classifier trained on FETAL_PLANES_DB (split by patient) asks: is this the trans-thalamic plane HC is defined on? If not, the measurement is withheld with a plain re-acquire message.",
    metric: "94.0 %",
    metricLabel: "plane accuracy, 5,271 unseen-patient images",
  },
  {
    id: "segment",
    icon: ScanLine,
    title: "Segment",
    tag: "U-Net · 7.8 M params",
    body: "A U-Net trained on HC18 outlines the fetal skull. The mask is mapped back to the original resolution so the aspect ratio of the scanner image is preserved.",
    metric: "0.973",
    metricLabel: "Dice on the held-out test split",
  },
  {
    id: "measure",
    icon: Crosshair,
    title: "Measure",
    tag: "Ellipse fit · Ramanujan",
    body: "An ellipse is fitted to the skull and its perimeter becomes HC in millimetres, then placed on the INTERGROWTH-21st growth standard for the gestational age.",
    metric: "2.18 mm",
    metricLabel: "mean absolute HC error (151 test images)",
  },
  {
    id: "trust",
    icon: Sigma,
    title: "Trust",
    tag: "8-pass test-time augmentation",
    body: "The same image is re-measured under flips, zooms and contrast changes. The spread becomes HC ± SD and a per-pixel uncertainty map; large spreads raise a re-scan flag.",
    metric: "3.68 vs 1.99",
    metricLabel: "mm error, flagged vs unflagged images",
  },
  {
    id: "fuse",
    icon: GitMerge,
    title: "Fuse",
    tag: "XGBoost + SHAP · rules R1–R6",
    body: "A CTG record is classified Normal / Suspect / Pathological with the features that drove it. Documented rules combine everything into Routine, Review recommended or Urgent review.",
    metric: "0.963",
    metricLabel: "CTG macro-F1 · 27/27 pathological caught",
  },
];

export function Pipeline() {
  const [active, setActive] = useState(0);
  const [paused, setPaused] = useState(false);
  const ref = useReveal<HTMLElement>();
  useEffect(() => {
    if (paused) return;
    const id = setInterval(() => setActive((a) => (a + 1) % STAGES.length), 4200);
    return () => clearInterval(id);
  }, [paused]);
  const s = STAGES[active];
  const Icon = s.icon;
  return (
    <section ref={ref} id="pipeline" className="mx-auto max-w-7xl px-4 py-24" data-testid="pipeline">
      <div className="reveal max-w-2xl">
        <p className="readout text-primary">01 · The pipeline</p>
        <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-4xl">
          Five stages between an image and a decision.
        </h2>
        <p className="mt-3 text-muted-foreground">
          Each stage is a separate, tested module with its own held-out evaluation. Pick a stage.
        </p>
      </div>
      <div
        className="reveal mt-10 grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)]"
        onMouseEnter={() => setPaused(true)}
        onMouseLeave={() => setPaused(false)}
      >
        <ol className="relative space-y-2">
          <span className="absolute bottom-6 left-[27px] top-6 w-px bg-gradient-to-b from-primary/60 via-border to-transparent" aria-hidden />
          {STAGES.map((st, i) => {
            const I = st.icon;
            const on = i === active;
            return (
              <li key={st.id}>
                <button
                  type="button"
                  onClick={() => setActive(i)}
                  aria-pressed={on}
                  className={cn(
                    "relative flex w-full items-center gap-4 rounded-xl border px-3 py-3 text-left transition-all",
                    on ? "border-primary/50 bg-primary/[0.06]" : "border-transparent hover:bg-muted/40",
                  )}
                >
                  <span
                    className={cn(
                      "relative z-10 grid size-10 shrink-0 place-items-center rounded-lg border bg-background transition-all",
                      on && "border-primary text-primary glow-ring",
                    )}
                  >
                    <I className="size-5" />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-2">
                      <span className="font-mono text-[10px] text-muted-foreground">0{i + 1}</span>
                      <span className="font-display font-medium">{st.title}</span>
                    </span>
                    <span className="block truncate text-xs text-muted-foreground">{st.tag}</span>
                  </span>
                  {on && (
                    <span className="absolute inset-x-3 bottom-0 h-px overflow-hidden">
                      <span key={`${active}-${paused}`} className={cn("block h-px bg-primary", !paused && "animate-[grow_4.2s_linear]")} />
                    </span>
                  )}
                </button>
              </li>
            );
          })}
        </ol>
        <div className="panel grain relative overflow-hidden p-8" onMouseMove={spotlightHandler}>
          <div className="spotlight pointer-events-none absolute inset-0" />
          <div className="bg-grid bg-grid-fade pointer-events-none absolute inset-0 opacity-60" />
          <div key={s.id} className="relative animate-in fade-in slide-in-from-bottom-2 duration-500">
            <div className="flex items-center gap-3">
              <span className="grid size-12 place-items-center rounded-xl bg-primary/10 text-primary">
                <Icon className="size-6" />
              </span>
              <div>
                <p className="readout">Stage 0{active + 1}</p>
                <h3 className="font-display text-2xl font-semibold">{s.title}</h3>
              </div>
            </div>
            <p className="mt-6 max-w-xl text-[15px] leading-relaxed text-muted-foreground">{s.body}</p>
            <div className="mt-8 flex items-end gap-4 border-t pt-6">
              <span className="font-display text-5xl font-semibold tracking-tight text-glow text-primary">{s.metric}</span>
              <span className="pb-2 text-sm text-muted-foreground">{s.metricLabel}</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

const QUESTIONS = [
  {
    q: "Is this even the right image?",
    a: "Scanners measure whatever they are shown. FetalAI checks the plane first and refuses to report an HC on a femur, an abdomen or the wrong brain plane, unless the user explicitly overrides.",
    stat: "99.8 %",
    statLabel: "non-brain images rejected",
    icon: ShieldCheck,
  },
  {
    q: "How much should I trust this number?",
    a: "Every HC comes with ± SD from eight re-measurements and a map of where the outline is uncertain. Thresholds were fixed on validation data before testing.",
    stat: "ρ 0.27",
    statLabel: "SD vs real error (modest, reported as is)",
    icon: Waves,
  },
  {
    q: "What does the rest of the picture say?",
    a: "A CTG record is classified with its reasons (SHAP), and a transparent rule set merges ultrasound and CTG into one summary that lists what drove it and what was not assessed.",
    stat: "27 / 27",
    statLabel: "pathological CTGs caught on test",
    icon: Activity,
  },
];

export function Questions() {
  const ref = useReveal<HTMLElement>();
  return (
    <section ref={ref} className="relative border-y bg-muted/20 py-24">
      <div className="mx-auto max-w-7xl px-4">
        <div className="reveal max-w-2xl">
          <p className="readout text-primary">02 · What is different</p>
          <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-4xl">
            Three questions an ultrasound machine does not ask.
          </h2>
        </div>
        <div className="mt-12 grid gap-5 md:grid-cols-3">
          {QUESTIONS.map((x, i) => {
            const I = x.icon;
            return (
              <article
                key={x.q}
                onMouseMove={spotlightHandler}
                className="reveal panel group relative overflow-hidden p-6 transition-transform duration-300 hover:-translate-y-1"
                style={{ transitionDelay: `${i * 90}ms` }}
              >
                <div className="spotlight pointer-events-none absolute inset-0 opacity-0 transition-opacity group-hover:opacity-100" />
                <div className="relative">
                  <div className="flex items-center justify-between">
                    <I className="size-5 text-primary" />
                    <span className="font-mono text-[10px] text-muted-foreground">Q.0{i + 1}</span>
                  </div>
                  <h3 className="mt-6 font-display text-xl font-semibold leading-snug">{x.q}</h3>
                  <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{x.a}</p>
                  <div className="mt-6 border-t pt-4">
                    <span className="font-display text-3xl font-semibold text-primary">{x.stat}</span>
                    <p className="text-xs text-muted-foreground">{x.statLabel}</p>
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function Counter({ value, decimals = 0, suffix = "", label, sub }: { value: number; decimals?: number; suffix?: string; label: string; sub: string }) {
  const [ref, seen] = useInView<HTMLDivElement>();
  const v = useCountUp(value, seen);
  return (
    <div ref={ref} className="relative border-l pl-5">
      <span className="absolute -left-px top-0 h-8 w-px bg-primary" />
      <div className="font-display text-4xl font-semibold tracking-tight tabular-nums sm:text-5xl">
        {v.toFixed(decimals)}
        <span className="text-2xl text-primary">{suffix}</span>
      </div>
      <p className="mt-2 text-sm font-medium">{label}</p>
      <p className="text-xs text-muted-foreground">{sub}</p>
    </div>
  );
}

export function MetricsWall() {
  const ref = useReveal<HTMLElement>();
  return (
    <section ref={ref} className="mx-auto max-w-7xl px-4 py-24">
      <div className="reveal flex flex-wrap items-end justify-between gap-4">
        <div className="max-w-2xl">
          <p className="readout text-primary">03 · Measured, not claimed</p>
          <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-4xl">Held-out numbers.</h2>
          <p className="mt-3 text-muted-foreground">
            Every figure was produced on data the models never saw during training or tuning.
          </p>
        </div>
        <Link href="/evidence" className="group inline-flex items-center gap-2 text-sm text-primary">
          Open the evidence <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
        </Link>
      </div>
      <div className="reveal mt-12 grid gap-10 sm:grid-cols-2 lg:grid-cols-4">
        <Counter value={0.973} decimals={3} label="Dice, head segmentation" sub="HC18 test split, n = 151" />
        <Counter value={2.18} decimals={2} suffix=" mm" label="HC mean absolute error" sub="Bland–Altman bias −0.10 mm" />
        <Counter value={94.0} decimals={1} suffix=" %" label="Plane accuracy" sub="FETAL_PLANES_DB official test, n = 5,271" />
        <Counter value={0.963} decimals={3} label="CTG macro-F1" sub="UCI CTG test split, n = 321" />
        <Counter value={0.014} decimals={3} label="Calibration error (ECE)" sub="plane head, after temperature scaling" />
        <Counter value={95.7} decimals={1} suffix=" %" label="Trans-thalamic accepted" sub="at the validation-chosen threshold" />
        <Counter value={0.06} decimals={2} suffix=" mm" label="Growth-standard fidelity" sub="all 189 INTERGROWTH-21st cells" />
        <Counter value={183} label="Automated tests" sub="unit, API, browser end-to-end" />
      </div>
    </section>
  );
}

const DATASETS = [
  { name: "HC18", what: "Fetal head ultrasound with HC annotations", n: "999 images · 806 exams", role: "Segment + measure" },
  { name: "FETAL_PLANES_DB", what: "Six maternal-fetal planes, brain sub-planes", n: "12,400 images · 1,792 patients", role: "Plane gate" },
  { name: "UCI Cardiotocography", what: "21 SisPorto CTG features, expert NSP labels", n: "2,126 records", role: "CTG branch" },
];

export function Datasets() {
  const ref = useReveal<HTMLElement>();
  return (
    <section ref={ref} className="mx-auto max-w-7xl px-4 pb-24">
      <div className="grid gap-4 md:grid-cols-3">
        {DATASETS.map((d, i) => (
          <div key={d.name} className="reveal panel p-5" style={{ transitionDelay: `${i * 80}ms` }}>
            <p className="readout">{d.role}</p>
            <h3 className="mt-2 font-display text-lg font-semibold">{d.name}</h3>
            <p className="mt-1 text-sm text-muted-foreground">{d.what}</p>
            <p className="mt-4 font-mono text-xs text-primary">{d.n}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

const NOT_CLAIMED = [
  ["Clinical validation", "Trained and tested on public research datasets only. Not a medical device."],
  ["Plane gate everywhere", "Accepts only 60.3 % of HC18 images (31.5 % of early-gestation heads): different machines, earlier pregnancies."],
  ["A calibrated combined risk", "Ultrasound and CTG data come from different patients, so the summary is rule-based and illustrative."],
  ["Perfect uncertainty", "The SD tracks the error only modestly (ρ 0.27); 10 of the 16 largest errors are not flagged."],
];

export function Honesty() {
  const ref = useReveal<HTMLElement>();
  return (
    <section ref={ref} className="border-y bg-[repeating-linear-gradient(135deg,transparent_0_14px,hsl(var(--warn)/0.035)_14px_15px)] py-24">
      <div className="mx-auto grid max-w-7xl gap-10 px-4 lg:grid-cols-[1fr_1.4fr]">
        <div className="reveal">
          <p className="readout text-warn">04 · What we do not claim</p>
          <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-4xl">
            The limits are part of the result.
          </h2>
          <p className="mt-3 text-muted-foreground">
            A model that hides its failure modes is not finished. These are measured and written into the model cards.
          </p>
        </div>
        <ul className="grid gap-3 sm:grid-cols-2">
          {NOT_CLAIMED.map(([t, d], i) => (
            <li key={t} className="reveal rounded-xl border border-warn/25 bg-card/60 p-5" style={{ transitionDelay: `${i * 70}ms` }}>
              <p className="flex items-center gap-2 font-display font-semibold">
                <span className="font-mono text-xs text-warn">✕</span> {t}
              </p>
              <p className="mt-2 text-sm text-muted-foreground">{d}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

const STACK = ["PyTorch 2.5 · CUDA", "U-Net", "EfficientNet-B0", "Temperature scaling", "Test-time augmentation", "XGBoost", "SHAP", "FastAPI", "Pydantic v2", "Next.js 15", "React 19", "TypeScript", "Tailwind", "Docker Compose", "pytest", "Playwright", "GitHub Actions"];

export function StackMarquee() {
  return (
    <div className="relative overflow-hidden border-b py-5 [mask-image:linear-gradient(90deg,transparent,black_12%,black_88%,transparent)]">
      <div className="flex w-max animate-marquee gap-10 font-mono text-xs uppercase tracking-[0.2em] text-muted-foreground">
        {[...STACK, ...STACK].map((s, i) => (
          <span key={i} className="flex items-center gap-10">
            {s} <span className="text-primary">◆</span>
          </span>
        ))}
      </div>
    </div>
  );
}

const TEAM = [
  ["Pujari Varshitha", "Data and preprocessing", "HC18 pipeline, FETAL_PLANES_DB data"],
  ["Nachiketha K G", "Models and evaluation", "U-Net, plane classifier, TTA uncertainty"],
  ["Vishwas K", "Backend and integration", "FastAPI, gate and uncertainty APIs, fusion"],
  ["Rakshitha V", "Frontend", "Web app, workstation, report"],
  ["Yashwanth Gowda M N", "Testing and CTG", "Test suites, CI, CTG model"],
];

export function Team() {
  const ref = useReveal<HTMLElement>();
  return (
    <section ref={ref} id="team" className="mx-auto max-w-7xl scroll-mt-20 px-4 py-24">
      <div className="reveal max-w-2xl">
        <p className="readout text-primary">05 · Team</p>
        <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight sm:text-4xl">B.Tech Major Project, REVA University.</h2>
      </div>
      <div className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        {TEAM.map(([n, r, d], i) => (
          <div key={n} className="reveal panel p-5" style={{ transitionDelay: `${i * 60}ms` }}>
            <div className="grid size-10 place-items-center rounded-full border border-primary/40 bg-primary/10 font-display text-sm font-semibold text-primary">
              {n.split(" ").map((w) => w[0]).slice(0, 2).join("")}
            </div>
            <p className="mt-4 font-display font-semibold leading-tight">{n}</p>
            <p className="text-xs text-primary">{r}</p>
            <p className="mt-2 text-xs text-muted-foreground">{d}</p>
          </div>
        ))}
      </div>
    </section>
  );
}
