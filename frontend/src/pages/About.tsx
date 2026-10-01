import type { ReactNode } from "react";
import Reveal from "../components/scroll/Reveal";

export default function About() {
  return (
    <div className="container-page py-10 md:py-14 bg-grain relative">
      <Reveal>
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-signal/80">SyncGuard // About</p>
        <h1 className="font-display text-4xl sm:text-5xl md:text-6xl mt-3 max-w-2xl leading-tight">
          Building Trust in Digital Media
        </h1>
        <p className="mt-4 max-w-xl text-base sm:text-lg text-ink-soft leading-relaxed">
          SyncGuard is a research project exploring how audio and visual analysis can be combined
          to examine synthetic speech and audio-visual synchronization.
        </p>
      </Reveal>

      <Reveal stagger={90} className="mt-16 md:mt-20 grid sm:grid-cols-3 gap-6">
        <AboutPoint
          n="01"
          icon={<FlaskIcon />}
          title="Research Driven"
          body="SyncGuard grew out of coursework and independent research into speech spoof detection and multimodal audio-visual analysis. It is a research artifact, not a commercial product."
        />
        <AboutPoint
          n="02"
          icon={<EyeIcon />}
          title="Open &amp; Transparent"
          body="The project documents what its models were trained and evaluated on, what the results were, and where the current approach falls short, rather than presenting only favorable numbers."
        />
        <AboutPoint
          n="03"
          icon={<LayersIcon />}
          title="Multimodal Analysis"
          body="Audio and visual signals are analyzed together, using landmark-based visual representations and cross-attention, to examine whether they remain temporally consistent."
        />
      </Reveal>

      <ThesisStatement />

      <Reveal className="mt-16 md:mt-20 flex flex-wrap items-center gap-3">
        <PillLink href="https://github.com/Kushall-07/SyncGuard">GitHub ↗</PillLink>
        <PillLink disabled>Documentation <span className="opacity-60">(coming soon)</span></PillLink>
        <PillLink disabled>Research / Paper <span className="opacity-60">(coming soon)</span></PillLink>
      </Reveal>
    </div>
  );
}

function AboutPoint({ n, icon, title, body }: { n: string; icon: ReactNode; title: string; body: string }) {
  return (
    <div className="relative border bg-canvas-raised/40 p-6" style={{ borderColor: "var(--page-border-strong)" }}>
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs uppercase tracking-[0.2em] text-signal/70">{n}</span>
        <span className="text-signal/80">{icon}</span>
      </div>
      <h2 className="font-display text-xl mt-5">{title}</h2>
      <p className="mt-3 text-sm text-ink-soft leading-relaxed">{body}</p>
    </div>
  );
}

// A small, restrained signal-path diagram for the page's thesis statement:
// the AI reasons over signals, evidence backs the result, the human decides.
// Purely decorative/illustrative — no live data, nothing to fabricate.
function ThesisStatement() {
  return (
    <Reveal className="mt-20 md:mt-28 border-t border-b border-[color:var(--page-border)] py-16 md:py-20 relative overflow-hidden">
      <ThesisInner />
    </Reveal>
  );
}

function ThesisInner() {
  return (
    <div className="relative">
      <div className="absolute inset-0 bg-grid opacity-40 -mx-6 md:-mx-10 pointer-events-none" aria-hidden="true" />
      <div className="relative flex flex-col items-center gap-10">
        <p className="font-display text-2xl sm:text-3xl md:text-4xl leading-relaxed max-w-2xl mx-auto text-center">
          AI reasons.
          <br />
          Evidence supports the result.
          <br />
          The user makes the final decision.
        </p>

        <div className="flex items-center gap-3 sm:gap-6">
          <SignalNode label="AI" tone="signal" />
          <SignalConnector />
          <SignalNode label="Evidence" tone="fusion" />
          <SignalConnector />
          <SignalNode label="User" tone="neutral" />
        </div>
      </div>
    </div>
  );
}

function SignalNode({ label, tone }: { label: string; tone: "signal" | "fusion" | "neutral" }) {
  const color = tone === "signal" ? "var(--page-signal)" : tone === "fusion" ? "var(--page-fusion)" : "#edeff3";
  return (
    <div
      className="shrink-0 border px-5 py-3 sm:px-7 sm:py-4 bg-canvas-raised"
      style={{ borderColor: `color-mix(in srgb, ${color} 50%, transparent)` }}
    >
      <span className="font-mono text-[11px] uppercase tracking-[0.2em]" style={{ color }}>
        {label}
      </span>
    </div>
  );
}

function SignalConnector() {
  return (
    <svg width="40" height="2" viewBox="0 0 40 2" className="shrink-0" preserveAspectRatio="none">
      <line x1="0" y1="1" x2="40" y2="1" stroke="var(--page-border-strong)" strokeWidth="1.5" strokeDasharray="4 4" />
    </svg>
  );
}

function PillLink({ href, disabled, children }: { href?: string; disabled?: boolean; children: ReactNode }) {
  const base = "inline-flex items-center gap-1.5 font-mono text-xs uppercase tracking-[0.1em] px-4 py-2.5 border transition-colors duration-200";
  if (disabled) {
    return (
      <span className={`${base} text-ink-faint cursor-not-allowed`} style={{ borderColor: "var(--page-border)" }} aria-disabled="true">
        {children}
      </span>
    );
  }
  return (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className={`${base} text-ink-soft hover:text-signal hover:border-signal`}
      style={{ borderColor: "var(--page-border)" }}
    >
      {children}
    </a>
  );
}

function FlaskIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M9 3h6 M10 3v6l-5.5 9a1.5 1.5 0 0 0 1.3 2.3h12.4a1.5 1.5 0 0 0 1.3-2.3L14 9V3" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M7.5 15h9" strokeLinecap="round" />
    </svg>
  );
}

function EyeIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function LayersIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="m12 3 9 5-9 5-9-5 9-5Z" strokeLinecap="round" strokeLinejoin="round" />
      <path d="m3 13 9 5 9-5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
