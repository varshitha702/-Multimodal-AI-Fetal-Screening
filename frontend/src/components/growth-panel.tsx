import { CentileChart } from "@/components/centile-chart";
import { MetricCard } from "@/components/metric-card";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { formatGa, type Growth } from "@/lib/api";

function ordinal(n: number) {
  const r = Math.round(n);
  const s = ["th", "st", "nd", "rd"];
  const v = r % 100;
  return `${r}${s[(v - 20) % 10] || s[v] || s[0]}`;
}

export function GrowthPanel({ growth, hcMm }: { growth: Growth; hcMm: number }) {
  const centileText = growth.centile < 1 ? "<1st" : growth.centile > 99 ? ">99th" : ordinal(growth.centile);
  return (
    <Card data-testid="growth-panel">
      <CardHeader>
        <CardTitle className="text-base">Head circumference for gestational age</CardTitle>
        <CardDescription>
          Measured HC compared with the INTERGROWTH-21st fetal growth standard at {formatGa(growth.gestational_age_weeks)}{" "}
          weeks. This is a comparison with a population standard, not a diagnosis.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <MetricCard label="Centile" value={centileText} hint={`z-score ${growth.z_score >= 0 ? "+" : ""}${growth.z_score.toFixed(2)}`} highlight />
          <MetricCard label="Standard median" value={growth.median_mm.toFixed(1)} unit="mm" hint="50th centile at this GA" />
          <MetricCard label="3rd – 97th centile" value={`${growth.p3_mm.toFixed(0)}–${growth.p97_mm.toFixed(0)}`} unit="mm" />
          <MetricCard
            label="Position"
            value={growth.within_p3_p97 ? "Within" : "Outside"}
            hint="3rd–97th centile range"
          />
        </div>
        <CentileChart growth={growth} hcMm={hcMm} />
      </CardContent>
    </Card>
  );
}
