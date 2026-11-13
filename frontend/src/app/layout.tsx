import type { Metadata } from "next";
import { Inter, JetBrains_Mono, Space_Grotesk } from "next/font/google";
import Link from "next/link";

import { LogoMark, Wordmark } from "@/components/brand";
import { SiteHeader } from "@/components/site-header";
import { API_URL } from "@/lib/api";
import { ThemeProvider } from "@/components/theme-provider";
import { SessionProvider } from "@/lib/session";

import "./globals.css";

const sans = Inter({ subsets: ["latin"], variable: "--font-sans", display: "swap" });
const display = Space_Grotesk({ subsets: ["latin"], variable: "--font-display", display: "swap" });
const mono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono", display: "swap" });

export const metadata: Metadata = {
  title: "FetalAI | Multimodal fetal screening prototype",
  description:
    "Standard-plane check, head-circumference measurement with uncertainty, CTG classification and an explained " +
    "screening summary. Academic research prototype, not a medical device.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning className={`${sans.variable} ${display.variable} ${mono.variable}`}>
      <body className="min-h-screen font-sans">
        <ThemeProvider attribute="class" defaultTheme="dark" enableSystem={false} disableTransitionOnChange>
          <SessionProvider>
            <SiteHeader />
            <div
              role="note"
              className="border-b border-warn/25 bg-warn/[0.07] px-4 py-1.5 text-center font-mono text-[11px] tracking-wide text-warn print:hidden"
            >
              Academic research prototype, <strong>not a medical device</strong>. Not clinically validated; do not use
              for diagnosis or pregnancy-care decisions.
            </div>
            <main>{children}</main>
            <footer className="border-t print:hidden">
              <div className="mx-auto grid max-w-7xl gap-6 px-4 py-10 text-sm sm:grid-cols-[1.4fr_1fr_1fr]">
                <div className="space-y-3">
                  <div className="flex items-center gap-2">
                    <LogoMark />
                    <Wordmark />
                  </div>
                  <p className="max-w-sm text-muted-foreground">
                    B.Tech Major Project, REVA University. Multimodal fetal screening prototype built on public data:
                    HC18, FETAL_PLANES_DB and UCI Cardiotocography.
                  </p>
                </div>
                <div className="space-y-2">
                  <p className="readout">Workspace</p>
                  <ul className="space-y-1 text-muted-foreground">
                    <li><Link className="hover:text-foreground" href="/scan">Ultrasound scan</Link></li>
                    <li><Link className="hover:text-foreground" href="/ctg">CTG lab</Link></li>
                    <li><Link className="hover:text-foreground" href="/report">Screening report</Link></li>
                  </ul>
                </div>
                <div className="space-y-2">
                  <p className="readout">Project</p>
                  <ul className="space-y-1 text-muted-foreground">
                    <li><Link className="hover:text-foreground" href="/evidence">Evidence and metrics</Link></li>
                    <li><Link className="hover:text-foreground" href="/#team">Team</Link></li>
                    <li><a className="hover:text-foreground" href={`${API_URL}/docs`}>API docs</a></li>
                  </ul>
                </div>
              </div>
            </footer>
          </SessionProvider>
        </ThemeProvider>
      </body>
    </html>
  );
}
