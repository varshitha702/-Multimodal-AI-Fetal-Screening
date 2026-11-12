"use client";

import { ImageUp, X } from "lucide-react";
import { useCallback, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { ACCEPTED_TYPES, MAX_UPLOAD_MB } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  file: File | null;
  previewUrl: string | null;
  onFile: (file: File | null) => void;
  onReject: (message: string) => void;
  disabled?: boolean;
}

export function validateFile(file: File): string | null {
  if (!ACCEPTED_TYPES.includes(file.type)) return `"${file.name}" is not a PNG or JPEG image.`;
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) return `"${file.name}" is larger than ${MAX_UPLOAD_MB} MB.`;
  if (file.size === 0) return `"${file.name}" is empty.`;
  return null;
}

export function UploadDropzone({ file, previewUrl, onFile, onReject, disabled }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const accept = useCallback(
    (files: FileList | null) => {
      const f = files?.[0];
      if (!f) return;
      const problem = validateFile(f);
      if (problem) onReject(problem);
      else onFile(f);
    },
    [onFile, onReject],
  );

  if (file && previewUrl) {
    return (
      <div className="relative overflow-hidden rounded-xl border bg-black">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={previewUrl} alt="Selected ultrasound" className="mx-auto max-h-72 w-full object-contain" />
        <div className="flex items-center justify-between gap-2 border-t bg-card px-3 py-2 text-sm">
          <span className="truncate" title={file.name}>
            {file.name} <span className="text-muted-foreground">· {(file.size / 1024).toFixed(0)} KB</span>
          </span>
          <Button variant="ghost" size="sm" onClick={() => onFile(null)} disabled={disabled} aria-label="Remove image">
            <X className="size-4" /> Remove
          </Button>
        </div>
      </div>
    );
  }

  return (
    <button
      type="button"
      disabled={disabled}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        accept(e.dataTransfer.files);
      }}
      className={cn(
        "flex w-full flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors",
        "hover:border-primary/60 hover:bg-primary/5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
        dragging ? "border-primary bg-primary/10" : "border-muted-foreground/25",
      )}
    >
      <span className="rounded-full bg-primary/10 p-3 text-primary">
        <ImageUp className="size-6" />
      </span>
      <span className="font-medium">Drop an ultrasound image here, or click to browse</span>
      <span className="text-sm text-muted-foreground">PNG or JPEG, up to {MAX_UPLOAD_MB} MB · trans-thalamic head plane</span>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        className="hidden"
        data-testid="file-input"
        onChange={(e) => {
          accept(e.target.files);
          e.target.value = "";
        }}
      />
    </button>
  );
}
