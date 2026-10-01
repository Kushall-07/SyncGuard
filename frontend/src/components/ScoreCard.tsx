interface ScoreCardProps {
  label: string;
  value: string;
  emphasis?: boolean;
}

export default function ScoreCard({ label, value, emphasis = false }: ScoreCardProps) {
  return (
    <div className="border bg-canvas-raised/40 px-5 py-5" style={{ borderColor: "var(--page-border-strong)" }}>
      <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-faint">{label}</p>
      <p className={`mt-2 font-display ${emphasis ? "text-5xl" : "text-3xl"}`}>{value}</p>
    </div>
  );
}
