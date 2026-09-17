import type { ReactNode } from "react";
import type { Evidence } from "../lib/evidence";

export default function EvidencePanel({ evidence }: { evidence: Evidence }) {
  return (
    <div className="border" style={{ borderColor: "var(--page-border)" }}>
      <div className="px-6 py-5 border-b" style={{ borderColor: "var(--page-border)" }}>
        <p className="text-xs uppercase tracking-widest opacity-60">Evidence</p>
        <h3 className="font-display text-xl mt-1">Why this result?</h3>
        <p className="mt-2 text-xs opacity-60 max-w-md leading-relaxed">
          Deterministic statements derived directly from model outputs — not a generated explanation.
        </p>
      </div>

      <EvidenceSection
        title="Observed"
        tone="sage"
        description="Direct model outputs."
      >
        <dl className="grid grid-cols-2 sm:grid-cols-3 gap-x-6 gap-y-4">
          {evidence.observed.map((item) => (
            <div key={item.label}>
              <dt className="text-xs uppercase tracking-widest opacity-50">{item.label}</dt>
              <dd className="mt-1 font-display text-lg">{item.value}</dd>
            </div>
          ))}
        </dl>
      </EvidenceSection>

      <EvidenceSection title="Derived" tone="copper" description="Statistics calculated from those outputs.">
        <ul className="space-y-3 text-sm opacity-85 leading-relaxed">
          {evidence.derived.map((line, i) => (
            <li key={i} className="pl-4 border-l-2" style={{ borderColor: "var(--page-border)" }}>
              {line}
            </li>
          ))}
        </ul>
      </EvidenceSection>

      <EvidenceSection title="Not Determined" tone="muted" description="Things the system cannot establish." last>
        <ul className="space-y-3 text-sm opacity-60 leading-relaxed">
          {evidence.notDetermined.map((line, i) => (
            <li key={i} className="pl-4 border-l-2" style={{ borderColor: "var(--page-border)" }}>
              {line}
            </li>
          ))}
        </ul>
      </EvidenceSection>
    </div>
  );
}

function EvidenceSection({
  title,
  description,
  tone,
  children,
  last = false,
}: {
  title: string;
  description: string;
  tone: "sage" | "copper" | "muted";
  children: ReactNode;
  last?: boolean;
}) {
  const dot = tone === "sage" ? "bg-sage" : tone === "copper" ? "bg-copper" : "bg-current opacity-40";
  return (
    <div
      className={`px-6 py-5 ${last ? "" : "border-b"}`}
      style={{ borderColor: "var(--page-border)" }}
    >
      <div className="flex items-center gap-2">
        <span className={`h-1.5 w-1.5 rounded-full ${dot}`} aria-hidden="true" />
        <p className="text-xs uppercase tracking-widest font-medium">{title}</p>
      </div>
      <p className="mt-1 text-xs opacity-50">{description}</p>
      <div className="mt-4">{children}</div>
    </div>
  );
}
