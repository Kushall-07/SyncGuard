interface RadialGaugeProps {
  /** 0-100 */
  value: number;
  label: string;
  tone?: "signal" | "anomaly" | "fusion";
  size?: number;
}

const TONE_VAR: Record<NonNullable<RadialGaugeProps["tone"]>, string> = {
  signal: "var(--page-signal)",
  anomaly: "var(--page-anomaly)",
  fusion: "var(--page-fusion)",
};

export default function RadialGauge({ value, label, tone = "signal", size = 96 }: RadialGaugeProps) {
  const r = (size - 14) / 2;
  const circumference = 2 * Math.PI * r;
  const clamped = Math.max(0, Math.min(100, value));
  const offset = circumference * (1 - clamped / 100);
  const color = TONE_VAR[tone];

  return (
    <div className="flex items-center gap-4">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90 shrink-0">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--page-border-strong)" strokeWidth="6" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          style={{ transition: "stroke-dashoffset 600ms cubic-bezier(0.23,1,0.32,1)" }}
        />
      </svg>
      <div>
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-ink-faint">{label}</p>
        <p className="font-display text-3xl mt-1" style={{ color }}>
          {clamped.toFixed(1)}%
        </p>
      </div>
    </div>
  );
}
