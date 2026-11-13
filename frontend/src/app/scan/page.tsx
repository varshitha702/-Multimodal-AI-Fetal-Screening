import type { Metadata } from "next";

import { PageHeader } from "@/components/console";
import { ScanWorkstation } from "@/components/scan/workstation";

export const metadata: Metadata = { title: "Scan | FetalAI" };

export default function ScanPage() {
  return (
    <>
      <PageHeader code="01" kicker="Ultrasound workstation" title="Fetal head circumference">
        Plane check first, then segmentation, HC in millimetres with its uncertainty, and the growth centile.
      </PageHeader>
      <ScanWorkstation />
    </>
  );
}
