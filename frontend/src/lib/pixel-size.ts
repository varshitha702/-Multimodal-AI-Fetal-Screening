// Auto-fill pixel spacing for HC18 demo images, matched by SHA-256 of the file bytes
// (the official HC18 test set reuses training-set filenames, so names are ambiguous).
import table from "./hc18-pixel-sizes.json";

const sizes = table as Record<string, { file: string; px: number }>;

export interface PixelSizeMatch {
  source: string; // e.g. "training_set/010_HC.png"
  pixelSizeMm: number;
}

export async function lookupPixelSize(file: File): Promise<PixelSizeMatch | null> {
  if (!globalThis.crypto?.subtle) return null; // non-secure context: skip auto-fill
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  const hex = Array.from(new Uint8Array(digest).slice(0, 8), (b) => b.toString(16).padStart(2, "0")).join("");
  const hit = sizes[hex];
  return hit ? { source: hit.file, pixelSizeMm: hit.px } : null;
}
