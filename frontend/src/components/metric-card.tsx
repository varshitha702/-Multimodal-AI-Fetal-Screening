import { Card, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface Props {
  label: string;
  value: string;
  unit?: string;
  hint?: string;
  highlight?: boolean;
}

export function MetricCard({ label, value, unit, hint, highlight }: Props) {
  return (
    <Card className={cn(highlight && "border-primary/40 bg-primary/5")}>
      <CardContent className="p-4">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
        <p className="mt-1 text-xl font-semibold tabular-nums sm:text-2xl" data-testid={`metric-${label.toLowerCase().replace(/\s+/g, "-")}`}>
          {value}
          {unit && <span className="ml-1 text-base font-normal text-muted-foreground">{unit}</span>}
        </p>
        {hint && <p className="mt-1 text-xs text-muted-foreground">{hint}</p>}
      </CardContent>
    </Card>
  );
}
