"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { ApiStatus } from "@/components/api-status";
import { LogoMark, Wordmark } from "@/components/brand";
import { ThemeToggle } from "@/components/theme-toggle";
import { useSession } from "@/lib/session";
import { cn } from "@/lib/utils";

export const NAV = [
  { href: "/", label: "Overview", code: "00" },
  { href: "/scan", label: "Scan", code: "01" },
  { href: "/ctg", label: "CTG", code: "02" },
  { href: "/report", label: "Report", code: "03" },
  { href: "/evidence", label: "Evidence", code: "04" },
];

const LEVEL_DOT: Record<string, string> = {
  Routine: "bg-ok",
  "Review recommended": "bg-warn",
  "Urgent review": "bg-alarm",
  "Insufficient data": "bg-muted-foreground",
};

/** Session tray: what has been analysed in this session, visible from every page. */
function SessionTray() {
  const { scan, ctg, summary } = useSession();
  const items = [
    { label: "Scan", done: !!scan.result, href: "/scan" },
    { label: "CTG", done: !!ctg.result, href: "/ctg" },
  ];
  return (
    <Link
      href="/report"
      data-testid="session-tray"
      className="hidden items-center gap-3 rounded-full border bg-card/60 px-3 py-1 font-mono text-[11px] uppercase tracking-wider text-muted-foreground transition-colors hover:border-primary/50 lg:flex"
      title="Current session: open the combined report"
    >
      {items.map((i) => (
        <span key={i.label} className="flex items-center gap-1.5">
          <span className={cn("size-1.5 rounded-full", i.done ? "bg-primary shadow-[0_0_8px_hsl(var(--primary))]" : "bg-muted-foreground/40")} />
          {i.label}
        </span>
      ))}
      <span className="h-3 w-px bg-border" />
      <span className="flex items-center gap-1.5">
        <span className={cn("size-1.5 rounded-full", summary ? LEVEL_DOT[summary.level] : "bg-muted-foreground/40")} />
        {summary ? summary.level : "No report"}
      </span>
    </Link>
  );
}

export function SiteHeader() {
  const pathname = usePathname();
  return (
    <header className="sticky top-0 z-40 border-b bg-background/75 backdrop-blur-xl print:hidden">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-3 px-4">
        <Link href="/" className="flex shrink-0 items-center gap-2" aria-label="FetalAI home">
          <LogoMark />
          <Wordmark />
        </Link>
        <nav className="-mx-1 flex min-w-0 flex-1 items-center gap-0.5 overflow-x-auto px-1 sm:ml-4" aria-label="Main">
          {NAV.map((l) => {
            const active = l.href === "/" ? pathname === "/" : pathname.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? "page" : undefined}
                className={cn(
                  "group relative flex shrink-0 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm transition-colors",
                  active ? "text-foreground" : "text-muted-foreground hover:text-foreground",
                )}
              >
                <span className={cn("hidden font-mono text-[10px] md:inline", active ? "text-primary" : "text-muted-foreground/60")}>
                  {l.code}
                </span>
                {l.label}
                {active && <span className="absolute inset-x-2 -bottom-[13px] h-px bg-primary shadow-[0_0_10px_hsl(var(--primary))]" />}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex shrink-0 items-center gap-2">
          <SessionTray />
          <ApiStatus />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
