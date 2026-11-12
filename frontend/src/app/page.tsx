import { Analyzer } from "@/components/analyzer";

export default function Home() {
  return (
    <div className="space-y-8">
      <section className="max-w-3xl print:hidden">
        <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">Fetal head circumference</h1>
        <p className="mt-2 text-muted-foreground">
          Upload a 2-D ultrasound image of the fetal head. A U-Net segments the skull, an ellipse is fitted to it, and
          its perimeter gives the head circumference in millimetres.
        </p>
      </section>
      <Analyzer />
    </div>
  );
}
