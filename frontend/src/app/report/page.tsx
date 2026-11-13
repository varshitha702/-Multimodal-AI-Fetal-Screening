import type { Metadata } from "next";

import { PageHeader } from "@/components/console";
import { ReportView } from "@/components/report/decision-trace";

export const metadata: Metadata = { title: "Report | FetalAI" };

export default function ReportPage() {
  return (
    <>
      <PageHeader code="03" kicker="Screening report" title="One summary, every reason shown">
        Decision-level fusion of the plane check, HC centile, HC reliability and CTG class by documented rules (R1–R6).
      </PageHeader>
      <ReportView />
    </>
  );
}
