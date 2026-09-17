import { useState, type ReactNode } from "react";

type TabKey = "overview" | "audio" | "visual" | "fusion";

const TABS: { key: TabKey; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "audio", label: "Audio Model" },
  { key: "visual", label: "Visual Model" },
  { key: "fusion", label: "Fusion & Synchronization" },
];

export default function Technology() {
  const [activeTab, setActiveTab] = useState<TabKey>("overview");

  return (
    <div className="container-page py-16">
      <p className="text-xs uppercase tracking-[0.2em] opacity-60">Technology</p>
      <h1 className="font-display text-4xl md:text-5xl mt-3 max-w-3xl leading-tight">
        A Multimodal Approach to Media Verification
      </h1>
      <p className="mt-4 max-w-xl opacity-70">
        SyncGuard combines audio and visual analysis to examine synthetic speech and temporal
        audio-visual consistency.
      </p>

      <div className="mt-10 flex flex-wrap gap-x-2 gap-y-1 border-b" style={{ borderColor: "var(--page-border)" }}>
        {TABS.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 sm:px-5 py-3 text-sm transition-colors duration-200 border-b-2 -mb-px whitespace-nowrap ${
              activeTab === tab.key ? "border-sage" : "border-transparent opacity-60 hover:opacity-100"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div key={activeTab} className="mt-12 fade-in">
        {activeTab === "overview" && <OverviewTab />}
        {activeTab === "audio" && <AudioModelTab />}
        {activeTab === "visual" && <VisualModelTab />}
        {activeTab === "fusion" && <FusionTab />}
      </div>
    </div>
  );
}

function OverviewTab() {
  return (
    <div className="space-y-8">
      <div className="max-w-2xl">
        <h2 className="font-display text-2xl">System overview</h2>
        <p className="mt-3 opacity-70 leading-relaxed">
          SyncGuard operates in two modes. An audio-only pipeline evaluates whether a speech
          signal is likely bonafide or spoofed. An audio-visual pipeline additionally aligns the
          audio stream with facial and mouth motion to examine whether the two remain
          synchronized over time.
        </p>
      </div>
      <ArchitectureDiagram />
    </div>
  );
}

function AudioModelTab() {
  return (
    <div className="grid md:grid-cols-2 gap-10">
      <div className="max-w-md">
        <h2 className="font-display text-2xl">Audio model</h2>
        <p className="mt-3 opacity-70 leading-relaxed">
          Raw audio is converted to a time-frequency representation, passed through a
          convolutional front-end, and then encoded by a transformer to produce a fixed
          representation of the speech signal used for spoof detection.
        </p>
        <p className="mt-4 text-sm opacity-60">
          Trained and evaluated on <span className="opacity-100">ASVspoof 2019 LA</span>.
        </p>
      </div>
      <StepList steps={["Mel Spectrogram", "CNN Front-End", "Audio Transformer", "Audio Representation"]} />
    </div>
  );
}

function VisualModelTab() {
  return (
    <div className="grid md:grid-cols-2 gap-10">
      <div className="max-w-md">
        <h2 className="font-display text-2xl">Visual model</h2>
        <p className="mt-3 opacity-70 leading-relaxed">
          The visual branch is landmark-based rather than image-based: MediaPipe extracts face
          and mouth landmark coordinates from each video frame, and a transformer encodes the
          resulting landmark sequence into a visual representation. No convolutional network is
          run over raw face-crop images in this branch.
        </p>
      </div>
      <StepList
        steps={["Video", "MediaPipe Face Landmarks", "Face + Mouth Representation", "Visual Landmark Transformer"]}
      />
    </div>
  );
}

function FusionTab() {
  return (
    <div className="grid md:grid-cols-2 gap-10">
      <div className="max-w-md">
        <h2 className="font-display text-2xl">Fusion &amp; synchronization</h2>
        <p className="mt-3 opacity-70 leading-relaxed">
          Audio and visual representations are aligned in time and related through bidirectional
          cross-attention, allowing each modality to attend to the other before a synchronization
          head produces a score for each analysis window.
        </p>
        <p className="mt-4 text-sm opacity-60 leading-relaxed">
          SyncGuard combines established deep-learning components — encoders, cross-attention,
          and contrastive objectives — into a multimodal analysis pipeline, rather than proposing
          a new architecture.
        </p>
      </div>
      <StepList
        steps={[
          "Temporal Audio-Visual Alignment",
          "Bidirectional Cross-Attention",
          "Synchronization Head",
          "Per-window Sync Scores",
        ]}
      />
    </div>
  );
}

function StepList({ steps }: { steps: string[] }) {
  return (
    <div className="flex flex-col items-stretch max-w-sm">
      {steps.map((step, i) => (
        <div key={step} className="flex flex-col items-center">
          <div
            className="w-full border px-5 py-3 text-sm text-center"
            style={{ borderColor: "var(--page-border)", background: "var(--page-surface)" }}
          >
            {step}
          </div>
          {i < steps.length - 1 && (
            <div className="py-1.5 text-sm opacity-40" aria-hidden="true">
              ↓
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

function NodeBox({
  children,
  accent,
  className = "",
}: {
  children: ReactNode;
  accent?: "sage" | "copper";
  className?: string;
}) {
  return (
    <div
      className={`border px-4 py-2.5 text-center text-sm leading-snug ${
        accent === "sage" ? "border-sage" : accent === "copper" ? "border-copper" : ""
      } ${className}`}
      style={{
        borderColor: accent ? undefined : "var(--page-border)",
        background: "var(--page-surface)",
      }}
    >
      {children}
    </div>
  );
}

function VLine({ h = 24 }: { h?: number }) {
  return <div style={{ height: h, width: 1, background: "var(--page-border)" }} aria-hidden="true" />;
}

function HBar() {
  return (
    <div className="relative w-full" style={{ height: 1 }} aria-hidden="true">
      <div className="absolute" style={{ left: "25%", right: "25%", top: 0, height: 1, background: "var(--page-border)" }} />
    </div>
  );
}

function ArchitectureDiagram() {
  return (
    <div className="overflow-x-auto border" style={{ borderColor: "var(--page-border)" }}>
      <div className="min-w-[640px] flex flex-col items-center py-10 px-6">
        <NodeBox className="uppercase text-xs tracking-[0.2em] font-medium px-6 py-3">SyncGuard</NodeBox>
        <VLine />
        <HBar />
        <div className="w-full grid grid-cols-2">
          {/* Audio-only branch */}
          <div className="flex flex-col items-center px-4 sm:px-8">
            <VLine h={20} />
            <NodeBox className="uppercase text-xs tracking-widest font-medium">Audio-Only</NodeBox>
            <VLine />
            <NodeBox>Audio Encoder</NodeBox>
            <VLine />
            <NodeBox>Spoof Head</NodeBox>
            <VLine h={16} />
            <p className="text-xs uppercase tracking-widest opacity-50 text-center mt-1">
              Bonafide / Spoof
            </p>
          </div>

          {/* Audio-visual branch */}
          <div className="flex flex-col items-center px-4 sm:px-8">
            <VLine h={20} />
            <NodeBox accent="copper" className="uppercase text-xs tracking-widest font-medium">
              Audio-Visual
            </NodeBox>
            <VLine />
            <div className="w-full grid grid-cols-2">
              <div className="flex flex-col items-center px-1 sm:px-2">
                <VLine h={16} />
                <NodeBox className="text-xs sm:text-sm">Audio Encoder</NodeBox>
              </div>
              <div className="flex flex-col items-center px-1 sm:px-2">
                <VLine h={16} />
                <NodeBox className="text-xs sm:text-sm">Visual Encoder</NodeBox>
              </div>
            </div>
            <HBar />
            <VLine />
            <NodeBox>Temporal Alignment</NodeBox>
            <VLine />
            <NodeBox>Bidirectional Cross-Attention</NodeBox>
            <VLine />
            <NodeBox>Sync Head</NodeBox>
            <VLine h={16} />
            <p className="text-xs uppercase tracking-widest text-sage-dark font-medium text-center mt-1">
              Sync / Desync
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
