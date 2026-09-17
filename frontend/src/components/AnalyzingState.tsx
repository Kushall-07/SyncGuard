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
      className="h-full min-h-[280px] border flex flex-col items-center justify-center gap-5"
      style={{ borderColor: "var(--page-border)" }}
      role="status"
      aria-live="polite"
    >
      <div className="h-8 w-8 rounded-full border-2 border-sage border-t-transparent animate-spin" />
      <div className="text-center">
        <p className="text-sm">{stages[stageIndex]}</p>
        <p className="mt-2 text-xs opacity-50">Running multimodal inference on the server…</p>
      </div>
      <ol className="flex flex-wrap items-center justify-center gap-x-2 gap-y-1 max-w-xs">
        {stages.map((stage, i) => (
          <li key={stage} className="flex items-center gap-2">
            <span
              className={`h-1.5 w-1.5 rounded-full ${i <= stageIndex ? "bg-sage" : "opacity-25 bg-current"}`}
              aria-hidden="true"
            />
            {i < stages.length - 1 && <span className="opacity-20 text-xs">·</span>}
          </li>
        ))}
      </ol>
    </div>
  );
}
