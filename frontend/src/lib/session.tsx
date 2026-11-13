"use client";

// One analysis session shared by all pages (Scan, CTG, Report): kept in memory while the user navigates with
// client-side links. Nothing is stored on the server or in the browser's storage.
import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

import {
  getScreeningSummary,
  screeningRequest,
  type CtgPrediction,
  type Prediction,
  type ScreeningSummary,
} from "@/lib/api";
import type { PixelSizeMatch } from "@/lib/pixel-size";

export interface ScanState {
  file: File | null;
  previewUrl: string | null;
  pixelSize: string;
  match: PixelSizeMatch | null;
  gaWeeks: string;
  gaDays: string;
  result: Prediction | null;
}

export interface CtgState {
  values: Record<string, string>;
  loaded: string | null;
  result: CtgPrediction | null;
}

interface Session {
  scan: ScanState;
  setScan: (patch: Partial<ScanState>) => void;
  ctg: CtgState;
  setCtg: (patch: Partial<CtgState>) => void;
  summary: ScreeningSummary | null;
  reset: () => void;
}

const EMPTY_SCAN: ScanState = {
  file: null,
  previewUrl: null,
  pixelSize: "",
  match: null,
  gaWeeks: "",
  gaDays: "",
  result: null,
};
const EMPTY_CTG: CtgState = { values: {}, loaded: null, result: null };

const SessionContext = createContext<Session | null>(null);

export function SessionProvider({ children }: { children: React.ReactNode }) {
  const [scan, setScanState] = useState<ScanState>(EMPTY_SCAN);
  const [ctg, setCtgState] = useState<CtgState>(EMPTY_CTG);
  const [summary, setSummary] = useState<ScreeningSummary | null>(null);

  const setScan = useCallback((patch: Partial<ScanState>) => setScanState((s) => ({ ...s, ...patch })), []);
  const setCtg = useCallback((patch: Partial<CtgState>) => setCtgState((s) => ({ ...s, ...patch })), []);
  const reset = useCallback(() => {
    setScanState((s) => {
      if (s.previewUrl) URL.revokeObjectURL(s.previewUrl);
      return EMPTY_SCAN;
    });
    setCtgState(EMPTY_CTG);
  }, []);

  // The combined summary follows whatever results exist (ultrasound, CTG or both).
  useEffect(() => {
    if (!scan.result && !ctg.result) {
      setSummary(null);
      return;
    }
    let cancelled = false;
    getScreeningSummary(screeningRequest(scan.result, ctg.result))
      .then((s) => !cancelled && setSummary(s))
      .catch(() => !cancelled && setSummary(null));
    return () => {
      cancelled = true;
    };
  }, [scan.result, ctg.result]);

  const value = useMemo(() => ({ scan, setScan, ctg, setCtg, summary, reset }), [scan, setScan, ctg, setCtg, summary, reset]);
  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession(): Session {
  const s = useContext(SessionContext);
  if (!s) throw new Error("useSession must be used inside <SessionProvider>");
  return s;
}
