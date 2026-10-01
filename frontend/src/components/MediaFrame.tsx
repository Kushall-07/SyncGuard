import type { RefObject } from "react";

interface MediaFrameProps {
  src: string;
  label: string;
  status?: string;
  maxHeight?: number;
  videoRef?: RefObject<HTMLVideoElement | null>;
}

// Bordered, HUD-labeled media preview shared by the Analyze workspace and the
// Synchronization Lab. `maxHeight` is deliberate, not cosmetic: an
// unconstrained <video> renders at its native aspect ratio, and a real
// (especially portrait) clip can grow tall enough to push the controls below
// it off-screen. object-fit: contain + a fixed cap keeps any aspect ratio
// bounded without cropping.
export default function MediaFrame({ src, label, status = "Loaded", maxHeight = 220, videoRef }: MediaFrameProps) {
  return (
    <div className="relative border bg-canvas-raised overflow-hidden" style={{ borderColor: "var(--page-border-strong)" }}>
      <div className="flex items-center justify-between px-4 py-2.5 border-b" style={{ borderColor: "var(--page-border)" }}>
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faint">{label}</p>
        <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.2em] text-signal">
          <span className="h-1.5 w-1.5 rounded-full bg-signal" />
          {status}
        </span>
      </div>
      <video
        ref={videoRef}
        src={src}
        controls
        className="w-full block bg-black"
        style={{ objectFit: "contain", maxHeight }}
      />
    </div>
  );
}
