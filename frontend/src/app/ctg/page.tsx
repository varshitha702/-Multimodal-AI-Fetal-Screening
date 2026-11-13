import type { Metadata } from "next";

import { PageHeader } from "@/components/console";
import { CtgLab } from "@/components/ctg/lab";

export const metadata: Metadata = { title: "CTG | FetalAI" };

export default function CtgPage() {
  return (
    <>
      <PageHeader code="02" kicker="CTG lab" title="Cardiotocography classification">
        21 SisPorto features of one CTG record → Normal / Suspect / Pathological, with the reasons (SHAP). Trained on UCI
        Cardiotocography, a different dataset and different patients from the ultrasound models.
      </PageHeader>
      <CtgLab />
    </>
  );
}
