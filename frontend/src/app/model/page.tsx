import { ModelInfoView } from "@/components/model-info";

export const metadata = { title: "Model | FetalAI" };

export default function ModelPage() {
  return (
    <div className="space-y-6">
      <section className="max-w-3xl">
        <h1 className="text-3xl font-semibold tracking-tight">Model</h1>
        <p className="mt-2 text-muted-foreground">
          The checkpoint served by the backend, with metrics measured on a held-out test split that was never used for
          training or model selection.
        </p>
      </section>
      <ModelInfoView />
    </div>
  );
}
