import { redirect } from "next/navigation";

// The model page moved to /evidence; keep old links working.
export default function ModelPage() {
  redirect("/evidence");
}
