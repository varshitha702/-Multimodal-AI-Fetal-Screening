// Typed client for the FetalAI FastAPI backend (see backend/app/schemas/prediction.py).

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export const MAX_UPLOAD_MB = 10;
export const ACCEPTED_TYPES = ["image/png", "image/jpeg"];

export interface Health {
  status: "ok" | "degraded";
  model_loaded: boolean;
  version: string;
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
  head_detected: boolean;
  head_circumference_mm: number | null;
  head_area_mm2: number | null;
  head_area_px: number;
  ellipse: Ellipse | null;
  quality: Quality;
  growth: Growth | null;
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

export function predict(file: File, pixelSizeMm: number | null, gaWeeks: number | null, signal?: AbortSignal) {
  const form = new FormData();
  form.append("file", file);
  if (pixelSizeMm !== null) form.append("pixel_size_mm", String(pixelSizeMm));
  if (gaWeeks !== null) form.append("gestational_age_weeks", String(gaWeeks));
  return request<Prediction>("/api/v1/predict", { method: "POST", body: form, signal });
}

/** "22+3" style label for exact weeks. */
export function formatGa(weeks: number) {
  const w = Math.floor(weeks + 1e-9);
  const d = Math.round((weeks - w) * 7);
  return d === 7 ? `${w + 1}+0` : `${w}+${d}`;
}

export const dataUrl = (b64: string) => `data:image/png;base64,${b64}`;
