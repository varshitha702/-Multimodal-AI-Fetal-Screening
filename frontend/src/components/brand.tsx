import { cn } from "@/lib/utils";

/** Logo mark: an ultrasound sector fan with a fitted head ellipse. */
export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={cn("size-7", className)} aria-hidden>
      <defs>
        <linearGradient id="lm" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="hsl(var(--primary))" />
          <stop offset="1" stopColor="hsl(var(--echo))" />
        </linearGradient>
      </defs>
      <path d="M16 3 L29 26 A15 15 0 0 1 3 26 Z" fill="url(#lm)" opacity="0.18" />
      <path d="M16 3 L29 26 A15 15 0 0 1 3 26 Z" fill="none" stroke="url(#lm)" strokeWidth="1.4" />
      <ellipse cx="16" cy="19" rx="6.2" ry="5" fill="none" stroke="url(#lm)" strokeWidth="1.8" />
      <circle cx="16" cy="3" r="1.6" fill="hsl(var(--echo))" />
    </svg>
  );
}

export function Wordmark() {
  return (
    <span className="font-display text-[17px] font-semibold tracking-tight">
      Fetal<span className="text-primary">AI</span>
    </span>
  );
}
