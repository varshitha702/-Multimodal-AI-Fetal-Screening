"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "Analyse" },
  { href: "/model", label: "Model" },
];

export function NavLinks() {
  const pathname = usePathname();
  return (
    <nav className="flex items-center gap-1 text-sm">
      {links.map((l) => (
        <Link
          key={l.href}
          href={l.href}
          className={cn(
            "rounded-md px-3 py-1.5 transition-colors hover:bg-accent hover:text-accent-foreground",
            pathname === l.href ? "font-medium text-foreground" : "text-muted-foreground",
          )}
        >
          {l.label}
        </Link>
      ))}
    </nav>
  );
}
