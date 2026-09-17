interface ScoreCardProps {
  label: string;
  value: string;
  emphasis?: boolean;
}

export default function ScoreCard({ label, value, emphasis = false }: ScoreCardProps) {
  return (
    <div
      className="border px-5 py-4"
      style={{ borderColor: "var(--page-border)" }}
    >
      <p className="text-xs uppercase tracking-widest opacity-60">{label}</p>
      <p className={`mt-2 font-display ${emphasis ? "text-4xl" : "text-2xl"}`}>{value}</p>
    </div>
  );
}
