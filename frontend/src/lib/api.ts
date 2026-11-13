// Typed client for the FetalAI FastAPI backend (see backend/app/schemas/prediction.py).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export const MAX_UPLOAD_MB = 10;
export const ACCEPTED_TYPES = ["image/png", "image/jpeg"];

export interface Health {
  status: "ok" | "degraded";
  model_loaded: boolean;
  plane_model_loaded: boolean;
  version: string;
}

export interface PlaneCheck {
  status: "pass" | "warn" | "unavailable";
  plane: string | null;
  plane_confidence: number | null;
  brain_subplane: string | null;
  brain_subplane_confidence: number | null;
  tt_probability: number | null;
  threshold: number | null;
  plane_probabilities: Record<string, number>;
  subplane_probabilities: Record<string, number>;
  overridden: boolean;
  message: string;
  model_version: string | null;
}

export interface Ellipse {
  center_x_px: number;
  center_y_px: number;
  semi_major_px: number;
  semi_minor_px: number;
  angle_deg: number;
  semi_major_mm: number | null;
  semi_minor_mm: number | null;
}

export interface Quality {
  mean_confidence: number;
  uncertain_fraction: number;
  mask_area_fraction: number;
  touches_border: boolean;
  ellipse_fit_ratio: number | null;
  flags: string[];
}

export interface Growth {
  gestational_age_weeks: number;
  z_score: number;
  centile: number;
  median_mm: number;
  p3_mm: number;
  p97_mm: number;
  within_p3_p97: boolean;
  reference: string;
}

export interface Uncertainty {
  n_passes: number;
  hc_mean_mm: number | null;
  hc_sd_mm: number | null;
  hc_cv_percent: number | null;
  hc_passes_mm: (number | null)[];
  disagreement_fraction: number;
  low_reliability: boolean;
  reasons: string[];
  thresholds: { hc_cv_percent: number; disagreement_fraction: number } | null;
  tta_ms: number;
}

export interface CentilePoint {
  ga_weeks: number;
  p3: number;
  p10: number;
  p50: number;
  p90: number;
  p97: number;
}

export interface CentileCurves {
  reference: string;
  unit: string;
  points: CentilePoint[];
}

export interface Prediction {
  image_width: number;
  image_height: number;
  pixel_size_mm: number | null;
  plane_check: PlaneCheck;
  measurement_withheld: boolean;
  head_detected: boolean;
  head_circumference_mm: number | null;
  head_area_mm2: number | null;
  head_area_px: number;
  ellipse: Ellipse | null;
  quality: Quality;
  growth: Growth | null;
  uncertainty: Uncertainty | null;
  uncertainty_png_base64: string | null;
  mask_png_base64: string;
  overlay_png_base64: string;
  inference_ms: number;
  model_version: string;
  disclaimer: string;
}

export interface ModelInfo {
  name: string;
  architecture: string;
  input_size: number;
  device: string;
  checkpoint_epoch: number | null;
  validation_dice: number | null;
  test_metrics: Record<string, unknown> | null;
  training_data: string;
  intended_use: string;
  plane_classifier: Record<string, unknown> | null;
  uncertainty: Record<string, unknown> | null;
  ctg_classifier: Record<string, unknown> | null;
  disclaimer: string;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public code: string,
  ) {
    super(message);
  }
}

async function parse<T>(res: Response): Promise<T> {
  if (res.ok) return (await res.json()) as T;
  let code = "http_error";
  let message = `Request failed (${res.status})`;
  try {
    const body = await res.json();
    if (body?.error) {
      code = body.error.code;
      message = body.error.message;
      const first = body.error.details?.[0];
      if (first?.message) message = `${message}: ${first.field?.split(".").pop()} ${first.message}`;
    }
  } catch {
    /* non-JSON error body */
  }
  throw new ApiError(message, res.status, code);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError(`Cannot reach the FetalAI API at ${API_URL}. Is the backend running?`, 0, "network_error");
  }
  return parse<T>(res);
}

export const getHealth = () => request<Health>("/api/v1/health");
export const getModelInfo = () => request<ModelInfo>("/api/v1/model-info");

export const getCentileCurves = () => request<CentileCurves>("/api/v1/reference/hc-centiles");

export function predict(
  file: File,
  pixelSizeMm: number | null,
  gaWeeks: number | null,
  signal?: AbortSignal,
  overridePlaneCheck = false,
) {
  const form = new FormData();
  form.append("file", file);
  if (pixelSizeMm !== null) form.append("pixel_size_mm", String(pixelSizeMm));
  if (gaWeeks !== null) form.append("gestational_age_weeks", String(gaWeeks));
  if (overridePlaneCheck) form.append("override_plane_check", "true");
  return request<Prediction>("/api/v1/predict", { method: "POST", body: form, signal });
}

/** "22+3" style label for exact weeks. */
export function formatGa(weeks: number) {
  const w = Math.floor(weeks + 1e-9);
  const d = Math.round((weeks - w) * 7);
  return d === 7 ? `${w + 1}+0` : `${w}+${d}`;
}

export const dataUrl = (b64: string) => `data:image/png;base64,${b64}`;

// --- CTG branch and screening summary (Feature C) -------------------------------------------------

export type CtgClass = "Normal" | "Suspect" | "Pathological";

export interface CtgFactor {
  feature: string;
  description: string;
  unit: string;
  value: number;
  shap: number;
  direction: "towards" | "away from";
}

export interface CtgPrediction {
  predicted_class: CtgClass;
  probabilities: Record<CtgClass, number>;
  top_factors: CtgFactor[];
  model_version: string;
  note: string;
  disclaimer: string;
}

export interface CtgExample {
  record: string;
  label: CtgClass;
  split: string;
  features: Record<string, number>;
}

export interface ScreeningRequest {
  plane_status?: "pass" | "warn" | "unavailable" | null;
  plane_overridden?: boolean;
  measurement_withheld?: boolean;
  hc_mm?: number | null;
  hc_z_score?: number | null;
  hc_centile?: number | null;
  low_reliability?: boolean | null;
  ctg_class?: CtgClass | null;
  ctg_probability?: number | null;
}

export interface ScreeningDriver {
  input: string;
  finding: string;
  level: string;
  rule: string;
}

export interface ScreeningSummary {
  level: "Routine" | "Review recommended" | "Urgent review" | "Insufficient data";
  drivers: ScreeningDriver[];
  unremarkable: ScreeningDriver[];
  not_assessed: string[];
  rules_version: string;
  rules: string[];
  caveats: string[];
  disclaimer: string;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const getCtgExamples = () => request<CtgExample[]>("/api/v1/ctg/examples");
export const predictCtg = (features: Record<string, number>) =>
  request<CtgPrediction>("/api/v1/ctg/predict", json(features));
export const getScreeningSummary = (req: ScreeningRequest) =>
  request<ScreeningSummary>("/api/v1/screening/summary", json(req));

/** Build the summary request from whatever results exist (missing inputs are reported as "not assessed"). */
export function screeningRequest(us: Prediction | null, ctg: CtgPrediction | null): ScreeningRequest {
  return {
    plane_status: us?.plane_check.status ?? null,
    plane_overridden: us?.plane_check.overridden ?? false,
    measurement_withheld: us?.measurement_withheld ?? false,
    hc_mm: us?.head_circumference_mm ?? null,
    hc_z_score: us?.growth?.z_score ?? null,
    hc_centile: us?.growth?.centile ?? null,
    low_reliability: us?.uncertainty ? us.uncertainty.low_reliability : null,
    ctg_class: ctg?.predicted_class ?? null,
    ctg_probability: ctg ? ctg.probabilities[ctg.predicted_class] : null,
  };
}
