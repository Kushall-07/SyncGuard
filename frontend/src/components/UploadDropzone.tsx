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
      className={`group relative cursor-pointer overflow-hidden bg-canvas-raised/50 bg-grid px-8 py-16 sm:py-20 text-center transition-all duration-300 ease-editorial focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-signal ${
        dragging ? "border-signal bg-signal/5" : "border hover:border-signal/50"
      }`}
      style={{ borderWidth: dragging ? 2 : 1, borderStyle: dragging ? "solid" : "dashed", borderColor: dragging ? undefined : "var(--page-border-strong)" }}
    >
      <CornerTicks active={dragging || !!file} />
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
        <div className="fade-in relative">
          <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.2em] text-signal border border-signal/40 px-2 py-0.5">
            <span className="h-1.5 w-1.5 rounded-full bg-signal" />
            Ready
          </span>
          <p className="font-display text-2xl sm:text-3xl mt-4 break-all px-4">{file.name}</p>
          <p className="mt-2 font-mono text-xs text-ink-faint">
            {(file.size / (1024 * 1024)).toFixed(2)} MB — click to replace
          </p>
        </div>
      ) : (
        <>
          <UploadIcon dragging={dragging} />
          <p className="font-display text-2xl sm:text-3xl mt-5">{label}</p>
          <p className="mt-2 text-sm sm:text-base text-ink-soft">or browse files</p>
          <p className="mt-6 font-mono text-[11px] uppercase tracking-[0.25em] text-ink-faint">{hint}</p>
        </>
      )}
    </div>
  );
}

function UploadIcon({ dragging }: { dragging: boolean }) {
  return (
    <svg
      width="40"
      height="40"
      viewBox="0 0 40 40"
      className={`mx-auto transition-transform duration-300 ${dragging ? "scale-110" : "group-hover:-translate-y-0.5"}`}
    >
      <path d="M20 26 V8 M20 8 L12 16 M20 8 L28 16" stroke="var(--page-signal)" strokeWidth="1.75" fill="none" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M6 30 V34 H34 V30" stroke="var(--page-signal)" strokeWidth="1.75" fill="none" strokeOpacity="0.6" strokeLinecap="round" />
    </svg>
  );
}

function CornerTicks({ active }: { active: boolean }) {
  const base = `absolute h-4 w-4 transition-colors duration-300 ${active ? "border-signal" : "border-ink-faint/40"}`;
  return (
    <>
      <span className={`${base} top-3 left-3 border-t border-l`} aria-hidden="true" />
      <span className={`${base} top-3 right-3 border-t border-r`} aria-hidden="true" />
      <span className={`${base} bottom-3 left-3 border-b border-l`} aria-hidden="true" />
      <span className={`${base} bottom-3 right-3 border-b border-r`} aria-hidden="true" />
    </>
  );
}
