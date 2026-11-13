import type { Metadata } from "next";

import { PageHeader } from "@/components/console";
import { EvidenceLocker } from "@/components/evidence/locker";

export const metadata: Metadata = { title: "Evidence | FetalAI" };

export default function EvidencePage() {
  return (
    <>
      <PageHeader code="04" kicker="Evidence locker" title="Every claim, with its numbers">
        Live metrics from the running backend and the evaluation figures behind them. All on held-out data.
      </PageHeader>
      <EvidenceLocker />
    </>
  );
}
