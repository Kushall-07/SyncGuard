import type { ReactNode } from "react";
import Reveal from "../components/scroll/Reveal";
import RadialGauge from "../components/RadialGauge";

export default function Research() {
  return (
    <div className="container-page py-10 md:py-14 bg-grain relative">
      <Reveal>
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-signal/80">SyncGuard // Research</p>
        <h1 className="font-display text-4xl sm:text-5xl md:text-6xl mt-3">Research &amp; Evaluation</h1>
        <p className="mt-4 max-w-xl text-base sm:text-lg text-ink-soft leading-relaxed">
          What SyncGuard was trained and evaluated on, how it was measured, and where those
          measurements do and do not generalize.
        </p>
      </Reveal>

      <div className="mt-16 md:mt-20 space-y-20 md:space-y-24">
        <Section n="01" title="Datasets">
          <Reveal stagger={80} className="grid sm:grid-cols-2 md:grid-cols-3 gap-5">
            <DatasetCard
              icon={<WaveIcon />}
              name="ASVspoof 2019 LA"
              usedFor="Audio spoof detection"
              description="Logical-access spoofed and bonafide speech used to train and evaluate the audio model."
            />
            <DatasetCard
              icon={<FaceIcon />}
              name="Celeb-DF v2"
              usedFor="Visual landmark representation development"
              description="Face video used during development of the MediaPipe-based landmark representation. It does not provide audio-visual synchronization ground truth."
            />
            <DatasetCard
              icon={<ShiftIcon />}
              name="LAV-DF"
              usedFor="Audio-visual synchronization evaluation"
              description="Video with controlled temporal audio-visual shifts, used to evaluate the synchronization head."
            />
          </Reveal>
        </Section>

        <Section n="02" title="Experiments">
          <Reveal>
            <p className="max-w-2xl text-sm sm:text-base text-ink-soft leading-relaxed">
              Evaluation covered two tasks: audio spoof classification on held-out ASVspoof 2019 LA
              utterances, and audio-visual synchronization detection on LAV-DF clips with controlled
              temporal shifts introduced between the audio and visual streams. Both tasks were
              evaluated independently, reflecting the two operating modes described under
              Technology.
            </p>
          </Reveal>
        </Section>

        <Section n="03" title="Metrics">
          <Reveal stagger={80} className="grid sm:grid-cols-2 gap-10">
            <MetricGroup title="Audio" items={["ROC-AUC", "EER", "Accuracy", "Precision", "Recall", "F1"]} />
            <MetricGroup
              title="Audio-Visual"
              items={["Synchronization score", "Per-window synchronization analysis", "Video-level synchronization AUC"]}
            />
          </Reveal>

          <Reveal className="mt-12">
            <p className="font-mono text-xs uppercase tracking-[0.25em] text-signal/80 mb-6">
              Validated audio spoof-detection results
            </p>

            <div className="grid sm:grid-cols-3 gap-5 mb-8">
              <GaugeCard label="CNN" value={98.8} />
              <GaugeCard label="Transformer" value={99.3} />
              <GaugeCard label="Ensemble" value={99.8} emphasis />
            </div>

            <div className="overflow-x-auto border bg-canvas-raised/40" style={{ borderColor: "var(--page-border-strong)" }}>
              <table className="w-full text-sm min-w-[420px]">
                <thead>
                  <tr className="border-b" style={{ borderColor: "var(--page-border)" }}>
                    <Th>Model</Th>
                    <Th>AUC</Th>
                    <Th>EER</Th>
                  </tr>
                </thead>
                <tbody>
                  <ResultRow model="CNN" auc="0.988" eer="4.98%" />
                  <ResultRow model="Transformer" auc="0.993" eer="3.74%" />
                  <ResultRow model="CNN + Transformer ensemble" auc="0.998" eer="1.93%" emphasis />
                </tbody>
              </table>
            </div>
          </Reveal>
        </Section>

        <Section n="04" title="Visual Results">
          <Reveal>
            <p className="max-w-2xl text-sm sm:text-base text-ink-soft leading-relaxed">
              The visual branch is landmark-based, not image-based: MediaPipe face and mouth landmarks are encoded
              by a transformer and combined with the audio branch through cross-attention. It was retained as an
              auxiliary multimodal representation that feeds the synchronization head, not as a standalone spoof
              or deepfake detector. No standalone visual-only accuracy figure is reported, because the visual
              branch was never evaluated as an independent classifier.
            </p>
          </Reveal>
        </Section>

        <Section n="05" title="AV Results">
          <Reveal>
            <p className="max-w-2xl text-sm sm:text-base text-ink-soft leading-relaxed">
              The sync head was evaluated on LAV-DF with controlled, artificial audio-video temporal shifts: a
              zero-shift (natively aligned) pair is treated as positive, and every non-zero shift — introduced by
              the evaluation script itself, not a naturally occurring deepfake artifact — is treated as negative.
              The mean sync score below is therefore a measure of sensitivity to artificial temporal misalignment,
              not a measure of real-world deepfake detection accuracy. Natural, unmodified clips do not provide
              independent synchronization ground truth in this setup, since each clip's own audio is the only
              known-positive pairing available.
            </p>
          </Reveal>

          <Reveal className="mt-8 border border-[color:var(--page-border-strong)] bg-canvas-raised/40 p-6">
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-faint mb-5">
              Mean Sync Score vs. Audio Shift
            </p>
            <div className="space-y-3">
              <ShiftBar label="0.0s (aligned)" value={0.9909} />
              <ShiftBar label="+0.5s" value={0.2545} />
              <ShiftBar label="+1.0s" value={0.1552} />
              <ShiftBar label="+2.0s" value={0.0} />
            </div>
          </Reveal>

          <Reveal className="mt-6 overflow-x-auto border border-[color:var(--page-border-strong)] bg-canvas-raised/40">
            <table className="w-full text-sm min-w-[420px]">
              <thead>
                <tr className="border-b" style={{ borderColor: "var(--page-border)" }}>
                  <Th>Audio Shift</Th>
                  <Th>Mean Sync Score</Th>
                </tr>
              </thead>
              <tbody>
                <DataRow col1="0.0s (aligned)" col2="0.9909" />
                <DataRow col1="+0.5s" col2="0.2545" />
                <DataRow col1="+1.0s" col2="0.1552" />
                <DataRow col1="+2.0s" col2="0.0000" />
              </tbody>
            </table>
          </Reveal>
          <p className="mt-3 font-mono text-xs text-ink-faint max-w-2xl">
            Phase 11 controlled shift-sensitivity evaluation, LAV-DF dev split, 1,000 clips. Negative shifts are
            not evaluable under this alignment scheme (zero valid overlapping windows) and are excluded from the
            Synchronization Lab for the same reason.
          </p>
          <Reveal>
            <p className="mt-6 max-w-2xl text-sm sm:text-base text-ink-soft leading-relaxed">
              A separate, independent check runs the same frozen pipeline at native (unshifted) timing over LAV-DF
              clips grouped by manipulation category rather than by artificial shift. Real, audio-only-fake,
              video-only-fake, and audio+video-fake clips all score within ≈0.001 of each other (~0.991 mean sync
              score, n=250 per category), confirming that manipulation is not treated as desynchronization by this
              model — content manipulation without temporal misalignment is not what the sync head is designed to
              detect. See <code className="font-mono text-xs bg-canvas-elevated px-1.5 py-0.5">docs/experiments.md</code> (Section 3.3) for the full table.
            </p>
          </Reveal>
        </Section>

        <Section n="06" title="Contrastive Ablation">
          <Reveal>
            <p className="max-w-2xl text-sm sm:text-base text-ink-soft leading-relaxed">
              Phase 12 adds an optional InfoNCE contrastive objective on top of the Phase 11 sync-only baseline,
              weighted by λ. Both configurations below were trained for 120 epochs on the same controlled-shift
              training setup and monitored on video-level synchronization AUC.
            </p>
          </Reveal>
          <Reveal className="mt-8 overflow-x-auto border border-[color:var(--page-border-strong)] bg-canvas-raised/40">
            <table className="w-full text-sm min-w-[420px]">
              <thead>
                <tr className="border-b" style={{ borderColor: "var(--page-border)" }}>
                  <Th>Configuration</Th>
                  <Th>Best Val. Video Sync AUC</Th>
                  <Th>Best Epoch</Th>
                </tr>
              </thead>
              <tbody>
                <DataRow col1="λ = 0 (baseline)" col2="1.0000" col3="7 / 120" />
                <DataRow col1="λ = 0.1 (contrastive)" col2="1.0000" col3="7 / 120" />
              </tbody>
            </table>
          </Reveal>
          <Reveal className="mt-5 max-w-2xl border border-[color:var(--page-anomaly)] bg-anomaly/5 px-5 py-4">
            <p className="text-sm text-ink-soft leading-relaxed">
              Both settings saturate at the same ceiling video-level AUC on the controlled-shift validation split.
              This experiment does not establish a measurable improvement on the controlled-shift validation
              setup — the comparison is inconclusive at this ceiling, not a demonstrated benefit of contrastive
              learning.
            </p>
          </Reveal>
        </Section>

        <Section n="07" title="Limitations">
          <Reveal className="max-w-2xl border border-[color:var(--page-anomaly)] bg-anomaly/5 px-5 py-5">
            <p className="font-mono text-xs uppercase tracking-[0.15em] text-anomaly mb-4">Read before relying on results</p>
            <ul className="space-y-3 text-sm text-ink-soft leading-relaxed">
              <li className="flex gap-3"><Bullet />AV synchronization analysis is not equivalent to universal deepfake detection.</li>
              <li className="flex gap-3"><Bullet />Manipulated content can remain synchronized.</li>
              <li className="flex gap-3"><Bullet />LAV-DF controlled shift evaluation does not establish natural real-world synchronization ground truth.</li>
              <li className="flex gap-3"><Bullet />Raw model scores are not calibrated probabilities.</li>
              <li className="flex gap-3"><Bullet />Visual landmark representation is an auxiliary multimodal representation rather than a standalone deepfake detector.</li>
              <li className="flex gap-3"><Bullet />Performance depends on input quality and preprocessing.</li>
              <li className="flex gap-3"><Bullet />No claim of frame-level manipulation localization is made.</li>
            </ul>
          </Reveal>
        </Section>
      </div>
    </div>
  );
}

function Bullet() {
  return <span className="mt-2 h-1 w-1 shrink-0 rounded-full bg-anomaly" aria-hidden="true" />;
}

function Section({ n, title, children }: { n: string; title: string; children: ReactNode }) {
  return (
    <section>
      <Reveal className="flex items-baseline gap-4">
        <span className="font-mono text-xs text-signal/70">{n}</span>
        <h2 className="font-display text-3xl md:text-4xl">{title}</h2>
      </Reveal>
      <div className="mt-7">{children}</div>
    </section>
  );
}

function DatasetCard({ icon, name, usedFor, description }: { icon: ReactNode; name: string; usedFor: string; description: string }) {
  return (
    <div className="border bg-canvas-raised/40 p-6" style={{ borderColor: "var(--page-border-strong)" }}>
      <span className="text-signal/80">{icon}</span>
      <h3 className="font-display text-lg mt-4">{name}</h3>
      <p className="mt-3 font-mono text-[10px] uppercase tracking-[0.2em] text-ink-faint">Used for</p>
      <p className="mt-1 text-sm text-ink-soft">{usedFor}</p>
      <p className="mt-4 text-sm text-ink-faint leading-relaxed">{description}</p>
    </div>
  );
}

function MetricGroup({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-signal/80">{title}</p>
      <ul className="mt-4 flex flex-wrap gap-2">
        {items.map((item) => (
          <li key={item} className="font-mono text-xs text-ink-soft border px-3 py-1.5" style={{ borderColor: "var(--page-border)" }}>
            {item}
          </li>
        ))}
      </ul>
    </div>
  );
}

function GaugeCard({ label, value, emphasis = false }: { label: string; value: number; emphasis?: boolean }) {
  return (
    <div
      className={`border p-5 ${emphasis ? "bg-canvas-raised" : "bg-canvas-raised/40"}`}
      style={{ borderColor: emphasis ? "var(--page-signal)" : "var(--page-border-strong)" }}
    >
      <RadialGauge value={value} label={label} tone="signal" size={84} />
    </div>
  );
}

function ShiftBar({ label, value }: { label: string; value: number }) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex items-center gap-4">
      <span className="w-28 shrink-0 font-mono text-xs text-ink-soft">{label}</span>
      <div className="relative flex-1 h-2.5 bg-canvas-elevated">
        <div className="absolute inset-y-0 left-0 bg-signal" style={{ width: `${pct}%` }} />
      </div>
      <span className="w-16 shrink-0 text-right font-mono text-xs text-ink-faint">{value.toFixed(4)}</span>
    </div>
  );
}

function Th({ children }: { children: ReactNode }) {
  return (
    <th className="text-left font-mono font-normal text-ink-faint uppercase text-[11px] tracking-[0.15em] px-5 py-3">
      {children}
    </th>
  );
}

function ResultRow({ model, auc, eer, emphasis = false }: { model: string; auc: string; eer: string; emphasis?: boolean }) {
  return (
    <tr className="border-b last:border-b-0" style={{ borderColor: "var(--page-border)" }}>
      <td className={`px-5 py-3 ${emphasis ? "font-medium text-ink" : "text-ink-soft"}`}>{model}</td>
      <td className={`px-5 py-3 font-display ${emphasis ? "text-signal" : ""}`}>≈ {auc}</td>
      <td className={`px-5 py-3 font-display ${emphasis ? "text-signal" : ""}`}>≈ {eer}</td>
    </tr>
  );
}

function DataRow({ col1, col2, col3 }: { col1: string; col2: string; col3?: string }) {
  return (
    <tr className="border-b last:border-b-0" style={{ borderColor: "var(--page-border)" }}>
      <td className="px-5 py-3 text-ink-soft">{col1}</td>
      <td className="px-5 py-3 font-display">{col2}</td>
      {col3 !== undefined && <td className="px-5 py-3 font-display">{col3}</td>}
    </tr>
  );
}

function WaveIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 12h3l2-7 3 14 3-11 2 8 2-4h5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function FaceIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <circle cx="12" cy="12" r="9" />
      <path d="M8.5 10.5v.01 M15.5 10.5v.01 M8.5 15c1 1 2.2 1.5 3.5 1.5s2.5-.5 3.5-1.5" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function ShiftIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M3 9h12 M3 15h12" strokeLinecap="round" />
      <path d="m15 6 4 3-4 3 M9 12l-4 3 4 3" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}
