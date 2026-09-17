import { useRef, useState } from "react";
import type { DragEvent } from "react";

interface UploadDropzoneProps {
  label: string;
  hint: string;
  accept: string;
  file: File | null;
  onSelect: (file: File) => void;
}

export default function UploadDropzone({ label, hint, accept, file, onSelect }: UploadDropzoneProps) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const dropped = e.dataTransfer.files?.[0];
    if (dropped) onSelect(dropped);
  };

  return (
    <div
      role="button"
      tabIndex={0}
      aria-label={file ? `${file.name} selected. Activate to replace.` : `${label}. ${hint}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      className={`group cursor-pointer rounded-sm border border-dashed px-8 py-16 text-center transition-all duration-300 ease-editorial focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sage ${
        dragging ? "border-sage bg-sage/5" : "hover:border-sage/60"
      }`}
      style={{ borderColor: dragging ? undefined : "var(--page-border)" }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onSelect(f);
        }}
      />
      {file ? (
        <div className="fade-in">
          <p className="font-display text-lg">{file.name}</p>
          <p className="mt-1 text-xs opacity-60">{(file.size / (1024 * 1024)).toFixed(2)} MB — click to replace</p>
        </div>
      ) : (
        <>
          <p className="font-display text-lg">{label}</p>
          <p className="mt-2 text-sm opacity-60">or browse files</p>
          <p className="mt-6 text-xs uppercase tracking-widest opacity-50">{hint}</p>
        </>
      )}
    </div>
  );
}
