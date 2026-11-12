"use client";

import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";
import { cn } from "@/lib/utils";

type State = "checking" | "ok" | "degraded" | "offline";

const LABEL: Record<State, string> = {
  checking: "Checking API…",
  ok: "Model ready",
  degraded: "Model not loaded",
  offline: "API offline",
};

export function ApiStatus() {
  const [state, setState] = useState<State>("checking");

  useEffect(() => {
    let alive = true;
    const check = () =>
      getHealth()
        .then((h) => alive && setState(h.model_loaded ? "ok" : "degraded"))
        .catch(() => alive && setState("offline"));
    check();
    const id = setInterval(check, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <span
      className="hidden items-center gap-2 rounded-full border px-3 py-1 text-xs sm:inline-flex"
      data-testid="api-status"
      title="FastAPI backend status"
    >
      <span
        className={cn(
          "size-2 rounded-full",
          state === "ok" && "bg-emerald-500",
          state === "degraded" && "bg-amber-500",
          state === "offline" && "bg-red-500",
          state === "checking" && "animate-pulse bg-muted-foreground",
        )}
      />
      {LABEL[state]}
    </span>
  );
}
