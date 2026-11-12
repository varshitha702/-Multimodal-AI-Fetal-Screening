import type { Metadata } from "next";

import { SiteHeader } from "@/components/site-header";
import { ThemeProvider } from "@/components/theme-provider";

import "./globals.css";

export const metadata: Metadata = {
  title: "FetalAI | Fetal head circumference",
  description: "Fetal head segmentation and head-circumference measurement (HC18 U-Net). Academic research prototype.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="min-h-screen font-sans">
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <SiteHeader />
          <div
            role="note"
            className="border-b border-amber-500/30 bg-amber-500/10 px-4 py-2 text-center text-xs font-medium text-amber-900 dark:text-amber-200 print:hidden"
          >
            Academic research prototype, <strong>not a medical device</strong>. Not clinically validated; do not use for
            diagnosis or pregnancy-care decisions.
          </div>
          <main className="container py-8">{children}</main>
          <footer className="container border-t py-6 text-xs text-muted-foreground print:hidden">
            FetalAI · B.Tech Major Project, REVA University · Trained on HC18 (van den Heuvel et al., PLoS ONE 2018)
          </footer>
        </ThemeProvider>
      </body>
    </html>
  );
}
