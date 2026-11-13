import { ArrowRight, FileBarChart2 } from "lucide-react";
import Link from "next/link";

import { HeroScan } from "@/components/landing/hero-scan";
import { Datasets, Honesty, MetricsWall, Pipeline, Questions, StackMarquee, Team } from "@/components/landing/sections";

export default function Landing() {
  return (
    <>
      {/* ---------- hero ---------- */}
      <section className="relative overflow-hidden border-b">
        <div className="bg-grid bg-grid-fade pointer-events-none absolute inset-0" aria-hidden />
        <div
          className="pointer-events-none absolute -top-40 left-1/2 h-[520px] w-[900px] -translate-x-1/2 rounded-full bg-primary/15 blur-[120px]"
          aria-hidden
        />
        {/* decorative heartbeat trace across the hero (not data) */}
        <svg className="pointer-events-none absolute bottom-10 left-0 h-16 w-full opacity-40" viewBox="0 0 1200 60" preserveAspectRatio="none" aria-hidden>
          <path
            d="M0 30 H340 l14 -2 l10 2 h20 l8 -22 l10 44 l9 -30 l7 8 h40 H700 l14 -2 l10 2 h20 l8 -22 l10 44 l9 -30 l7 8 h40 H1200"
            fill="none"
            stroke="hsl(var(--primary))"
            strokeWidth="1.5"
            className="animate-ecg"
          />
        </svg>
        <div className="relative mx-auto grid max-w-7xl items-center gap-12 px-4 pb-24 pt-16 lg:grid-cols-[1.05fr_1fr] lg:pt-24">
          <div>
            <p className="inline-flex items-center gap-2 rounded-full border bg-card/60 px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-muted-foreground">
              <span className="size-1.5 rounded-full bg-primary shadow-[0_0_8px_hsl(var(--primary))]" />
              Multimodal fetal screening · research prototype
            </p>
            <h1 className="mt-6 font-display text-5xl font-semibold leading-[1.02] tracking-tight sm:text-6xl lg:text-7xl">
              Measure only
              <br />
              what you can <span className="text-gradient">trust.</span>
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-muted-foreground">
              FetalAI checks that an ultrasound is the right plane before it measures, reports fetal head circumference
              with its uncertainty, reads the CTG, and explains one combined screening summary.
            </p>
            <div className="mt-8 flex flex-wrap items-center gap-3">
              <Link
                href="/scan"
                className="group inline-flex h-11 items-center gap-2 rounded-lg bg-primary px-5 font-medium text-primary-foreground shadow-[0_0_30px_-6px_hsl(var(--primary))] transition-transform hover:-translate-y-0.5"
              >
                Open the workstation <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
              </Link>
              <Link
                href="/evidence"
                className="inline-flex h-11 items-center gap-2 rounded-lg border bg-card/60 px-5 font-medium transition-colors hover:border-primary/50"
              >
                <FileBarChart2 className="size-4" /> See the evidence
              </Link>
            </div>
            <dl className="mt-12 grid max-w-lg grid-cols-3 gap-6 border-t pt-6">
              {[
                ["2.18 mm", "HC error"],
                ["94.0 %", "plane accuracy"],
                ["27/27", "pathological CTG"],
              ].map(([v, l]) => (
                <div key={l}>
                  <dt className="readout">{l}</dt>
                  <dd className="mt-1 font-display text-2xl font-semibold">{v}</dd>
                </div>
              ))}
            </dl>
          </div>
          <div className="animate-float">
            <HeroScan />
          </div>
        </div>
      </section>

      <StackMarquee />
      <Pipeline />
      <Questions />
      <MetricsWall />
      <Datasets />
      <Honesty />
      <Team />

      {/* ---------- closing call to action ---------- */}
      <section className="relative overflow-hidden border-t">
        <div className="bg-grid bg-grid-fade pointer-events-none absolute inset-0" aria-hidden />
        <div className="relative mx-auto flex max-w-7xl flex-col items-center px-4 py-24 text-center">
          <p className="readout text-primary">Try it</p>
          <h2 className="mt-3 max-w-2xl font-display text-3xl font-semibold tracking-tight sm:text-5xl">
            Drop a scan. Watch every stage decide.
          </h2>
          <div className="mt-8 flex flex-wrap justify-center gap-3">
            <Link href="/scan" className="inline-flex h-11 items-center gap-2 rounded-lg bg-primary px-5 font-medium text-primary-foreground">
              Start with an ultrasound <ArrowRight className="size-4" />
            </Link>
            <Link href="/ctg" className="inline-flex h-11 items-center gap-2 rounded-lg border px-5 font-medium hover:border-primary/50">
              Or a CTG record
            </Link>
          </div>
        </div>
      </section>
    </>
  );
}
