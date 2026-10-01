import { useEffect, useState } from "react";

// Rotates through named processing stages while an analysis request is in flight.
// This communicates *what* is happening, never a fabricated progress percentage —
// the frontend has no visibility into real backend progress.
export default function AnalyzingState({ stages }: { stages: string[] }) {
  const [stageIndex, setStageIndex] = useState(0);

  useEffect(() => {
    setStageIndex(0);
    const id = setInterval(() => {
      setStageIndex((i) => Math.min(i + 1, stages.length - 1));
    }, 1600);
    return () => clearInterval(id);
  }, [stages]);

  return (
    <div
      className="relative h-full min-h-[360px] border bg-canvas-raised/50 bg-grid overflow-hidden flex flex-col items-center justify-center gap-8 px-8 py-10"
      style={{ borderColor: "var(--page-border-strong)" }}
      role="status"
      aria-live="polite"
    >
      <div className="absolute top-5 left-6 font-mono text-[10px] uppercase tracking-[0.25em] text-signal/80">
        Analysis Running
      </div>

      <div className="relative h-20 w-20 shrink-0">
        <span className="absolute inset-0 rounded-full border-2 border-signal/20" />
        <span className="absolute inset-0 rounded-full border-2 border-t-signal border-r-transparent border-b-transparent border-l-transparent animate-spin" />
        <span className="absolute inset-3 rounded-full" style={{ animation: "pulse-signal 1.8s ease-in-out infinite", background: "radial-gradient(circle, rgba(34,211,238,0.18), transparent 70%)" }} />
      </div>

      <ol className="w-full max-w-xs flex flex-col gap-2.5">
        {stages.map((stage, i) => {
          const done = i < stageIndex;
          const active = i === stageIndex;
          return (
            <li key={stage} className="flex items-center gap-3">
              <span
                className={`flex h-4 w-4 shrink-0 items-center justify-center border font-mono text-[9px] ${
                  done ? "border-signal bg-signal text-[#061012]" : active ? "border-signal text-signal" : "border-ink-faint/40 text-transparent"
                }`}
              >
                {done ? "✓" : active ? "●" : ""}
              </span>
              <span
                className={`font-mono text-xs uppercase tracking-[0.1em] transition-colors duration-300 ${
                  done ? "text-ink-soft" : active ? "text-ink" : "text-ink-faint/50"
                }`}
              >
                {stage}
              </span>
            </li>
          );
        })}
      </ol>

      <p className="text-xs text-ink-faint">Running multimodal inference on the server…</p>
    </div>
  );
}
