import { Baby } from "lucide-react";
import Link from "next/link";

import { ApiStatus } from "@/components/api-status";
import { NavLinks } from "@/components/nav-links";
import { ThemeToggle } from "@/components/theme-toggle";

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-30 border-b bg-background/85 backdrop-blur print:hidden">
      <div className="container flex h-14 items-center gap-4">
        <Link href="/" className="flex items-center gap-2 font-semibold">
          <span className="rounded-md bg-primary p-1 text-primary-foreground">
            <Baby className="size-4" />
          </span>
          FetalAI
        </Link>
        <NavLinks />
        <div className="ml-auto flex items-center gap-2">
          <ApiStatus />
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
