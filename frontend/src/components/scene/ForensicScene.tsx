import { useMemo, useState } from "react";
import type { CSSProperties, ReactNode } from "react";
import { Link } from "react-router-dom";
import { cueValue, usePinnedProgress, usePrefersReducedMotion } from "../../lib/scroll";
import type { Cue } from "../../lib/scroll";
import { useMagnetic, usePointerInElement, useTilt } from "../../lib/cursor";

// ---------------------------------------------------------------------------
// One continuous scene, six states, driven entirely by how far the reader has
// scrolled through a single pinned stage. No section is its own component —
// every state shares the same frame, the same two signal tracks and the same
// corner HUD, and only what's drawn inside that frame changes. The six beats
// mirror the pipeline the backend actually runs (see Results.tsx's
// ModelPipeline / predictor.py): ingestion -> feature extraction -> audio +
// visual encoding -> cross-attention -> sync scoring -> verdict.
// ---------------------------------------------------------------------------

const TOTAL_STATES = 6;

// [inStart, inEnd, outStart, outEnd] in overall pin progress (0..1).
// State 0 greets (already visible at p=0); state 5 holds (never fades once reached).
const WINDOWS: Cue[] = [
  [0, 0, 0.12, 0.17],
  [0.13, 0.18, 0.29, 0.34],
  [0.3, 0.35, 0.46, 0.5],
  [0.47, 0.52, 0.63, 0.67],
  [0.64, 0.69, 0.8, 0.84],
  [0.81, 0.86, 1, 1],
];

// The [hold-start, hold-end] span of each state, used to derive a 0..1 local
// progress for continuous motion inside a state (drift distance, node stagger).
const HOLD: [number, number][] = [
  [0, 0.12],
  [0.18, 0.29],
  [0.35, 0.46],
  [0.52, 0.63],
  [0.69, 0.8],
  [0.86, 1],
];

function localProgress(p: number, i: number): number {
  const [s, e] = HOLD[i];
  return Math.min(1, Math.max(0, (p - s) / (e - s)));
}

const STATE_LABEL = [
  "Raw Media",
  "Signals Extracted",
  "Temporal Alignment",
  "Sync Drift",
  "Cross-Modal Analysis",
  "Verdict",
];

function wavePath(amplitude: number, phase: number, width = 420, height = 28, points = 48): string {
  let d = "";
  for (let i = 0; i <= points; i++) {
    const x = (i / points) * width;
    const y = height / 2 + Math.sin((i / points) * Math.PI * 4 + phase) * amplitude;
    d += `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${y.toFixed(1)} `;
  }
  return d.trim();
}

const AUDIO_PATH = wavePath(9, 0);
const VIDEO_PATH = wavePath(7, 0.9);

export default function ForensicScene() {
  const reduced = usePrefersReducedMotion();
  if (reduced) return <StaticNarrative />;
  return <PinnedNarrative />;
}

// ---------------------------------------------------------------------------
// Motion-enabled path
// ---------------------------------------------------------------------------

function PinnedNarrative() {
  const [ref, p] = usePinnedProgress<HTMLDivElement>();
  const [hover, setHover] = useState<"audio" | "video" | "sync" | null>(null);

  const opacities = useMemo(() => WINDOWS.map((w) => cueValue(p, w)), [p]);
  const activeIndex = opacities.indexOf(Math.max(...opacities));
  const driftLocal = localProgress(p, 3);
  const analysisLocal = localProgress(p, 4);
  const mismatchDetected = driftLocal > 0.55;
  const [stageRef, cursor] = usePointerInElement<HTMLDivElement>();

  return (
    <section ref={ref} className="relative" style={{ height: "640vh" }} aria-label="SyncGuard analysis walkthrough">
      <div
        ref={stageRef}
        className="sticky top-0 h-screen w-full overflow-hidden bg-grid bg-grain border-b"
        style={{ borderColor: "var(--page-border)" }}
      >
        <RadialGlow activeIndex={activeIndex} />
        <SignalTraces activeIndex={activeIndex} />
        <ScanLine active={activeIndex <= 1} />
        <CursorTracker cursor={cursor} />

        <HudCorners p={p} activeIndex={activeIndex} totalStates={TOTAL_STATES} />

        <div className="relative h-full w-full container-page flex flex-col items-center justify-center">
          {/* ---- State 0: Raw Media ---- */}
          <StateLayer opacity={opacities[0]}>
            <RawMediaState />
          </StateLayer>

          {/* ---- State 1: Signals Extracted ---- */}
          <StateLayer opacity={opacities[1]}>
            <SignalsExtractedState />
          </StateLayer>

          {/* ---- State 2: Temporal Alignment ---- */}
          <StateLayer opacity={opacities[2]}>
            <AlignmentState hover={hover} onHover={setHover} />
          </StateLayer>

          {/* ---- State 3: Sync Drift ---- */}
          <StateLayer opacity={opacities[3]}>
            <DriftState local={driftLocal} mismatchDetected={mismatchDetected} />
          </StateLayer>

          {/* ---- State 4: Cross-Modal Analysis ---- */}
          <StateLayer opacity={opacities[4]}>
            <AnalysisState local={analysisLocal} />
          </StateLayer>

          {/* ---- State 5: Verdict ---- */}
          <StateLayer opacity={opacities[5]}>
            <VerdictState />
          </StateLayer>
        </div>

        <div
          className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-2"
          aria-hidden="true"
        >
          {STATE_LABEL.map((label, i) => (
            <span
              key={label}
              className="h-1.5 w-1.5 rounded-full transition-colors duration-500"
              style={{ backgroundColor: i === activeIndex ? "var(--page-signal)" : "rgba(237,239,243,0.18)" }}
            />
          ))}
        </div>
      </div>
    </section>
  );
}

function StateLayer({ opacity, children }: { opacity: number; children: ReactNode }) {
  const style: CSSProperties = {
    opacity,
    transform: `scale(${0.98 + 0.02 * opacity}) translateY(${(1 - opacity) * 14}px)`,
    pointerEvents: opacity > 0.6 ? "auto" : "none",
  };
  return (
    <div className="absolute inset-0 flex items-center justify-center" style={style}>
      {children}
    </div>
  );
}

function RadialGlow({ activeIndex }: { activeIndex: number }) {
  const intensity = [0.12, 0.16, 0.2, 0.26, 0.32, 0.18][activeIndex] ?? 0.14;
  const hue = activeIndex === 3 ? "245,158,11" : activeIndex === 4 ? "139,124,246" : "34,211,238";
  return (
    <div
      className="absolute inset-0 pointer-events-none transition-[opacity] duration-700"
      style={{
        background: `radial-gradient(60% 50% at 50% 42%, rgba(${hue},${intensity}), transparent 70%)`,
      }}
      aria-hidden="true"
    />
  );
}

function ScanLine({ active }: { active: boolean }) {
  return (
    <div
      className="absolute inset-x-0 h-px bg-signal/30 pointer-events-none transition-opacity duration-500"
      style={{
        opacity: active ? 1 : 0,
        animation: active ? "scanline 5.5s linear infinite" : "none",
      }}
      aria-hidden="true"
    />
  );
}

const TRACE_PATH = wavePath(16, 0, 900, 48, 60);
const DATA_POINTS = [
  [6, 22], [92, 14], [14, 68], [96, 74], [50, 10], [78, 88], [8, 44], [97, 46], [34, 92], [62, 18],
];

// Always-on background telemetry: two faint signal traces drifting at
// different speeds, plus a sparse scatter of pulsing data points. Gives the
// stage a sense of "live instrument" even during a state's quiet moments.
function SignalTraces({ activeIndex }: { activeIndex: number }) {
  const color1 = activeIndex === 3 ? "#f59e0b" : "#22d3ee";
  const color2 = activeIndex === 4 ? "#8b7cf6" : "#22d3ee";
  return (
    <div className="absolute inset-0 pointer-events-none overflow-hidden" aria-hidden="true">
      <div className="absolute left-0 top-[18%] w-[1800px] h-12 opacity-[0.07]" style={{ animation: "trace-drift 26s linear infinite" }}>
        <svg viewBox="0 0 1800 48" className="w-full h-full" preserveAspectRatio="none">
          <path d={TRACE_PATH} fill="none" stroke={color1} strokeWidth="1" />
          <path d={TRACE_PATH} transform="translate(900,0)" fill="none" stroke={color1} strokeWidth="1" />
        </svg>
      </div>
      <div className="absolute left-0 top-[76%] w-[1800px] h-12 opacity-[0.06]" style={{ animation: "trace-drift 34s linear infinite reverse" }}>
        <svg viewBox="0 0 1800 48" className="w-full h-full" preserveAspectRatio="none">
          <path d={wavePath(11, 1.4, 900, 48, 60)} fill="none" stroke={color2} strokeWidth="1" />
          <path d={wavePath(11, 1.4, 900, 48, 60)} transform="translate(900,0)" fill="none" stroke={color2} strokeWidth="1" />
        </svg>
      </div>
      {DATA_POINTS.map(([x, y], i) => (
        <span
          key={i}
          className="absolute h-1 w-1 rounded-full bg-signal"
          style={{
            left: `${x}%`,
            top: `${y}%`,
            animation: `dot-pulse ${3 + (i % 4)}s ease-in-out infinite`,
            animationDelay: `${i * 0.4}s`,
          }}
        />
      ))}
    </div>
  );
}

// Pointer-tracking HUD: a soft scanner spotlight, a viewfinder reticle, and a
// live coordinate readout, all following the cursor. Desktop-only (the
// underlying hook is gated to hover:hover + pointer:fine and disabled under
// reduced motion), purely decorative (pointer-events: none throughout) so it
// never competes with the real hover targets in the alignment state.
function CursorTracker({ cursor }: { cursor: { x: number; y: number } | null }) {
  if (!cursor) return null;
  return (
    <div className="absolute inset-0 pointer-events-none" aria-hidden="true">
      <div
        className="absolute h-72 w-72 -translate-x-1/2 -translate-y-1/2 rounded-full"
        style={{
          left: cursor.x,
          top: cursor.y,
          background: "radial-gradient(circle, rgba(34,211,238,0.07), transparent 70%)",
        }}
      />
      <div
        className="absolute -translate-x-1/2 -translate-y-1/2"
        style={{ left: cursor.x, top: cursor.y }}
      >
        <svg width="28" height="28" viewBox="0 0 28 28" className="opacity-70">
          <path d="M14 0 V8 M14 20 V28 M0 14 H8 M20 14 H28" stroke="var(--page-signal)" strokeWidth="1" />
          <circle cx="14" cy="14" r="5" fill="none" stroke="var(--page-signal)" strokeWidth="1" strokeOpacity="0.6" />
        </svg>
        <p className="absolute top-4 left-4 whitespace-nowrap font-mono text-[10px] tracking-[0.15em] text-signal/80">
          X:{String(Math.round(cursor.x)).padStart(4, "0")} Y:{String(Math.round(cursor.y)).padStart(4, "0")}
        </p>
      </div>
    </div>
  );
}

function HudCorners({ p, activeIndex, totalStates }: { p: number; activeIndex: number; totalStates: number }) {
  const seconds = (p * 13.42).toFixed(2).padStart(5, "0");
  const frame = Math.round(p * 1842)
    .toString()
    .padStart(5, "0");

  const rows: [string, string][] =
    activeIndex === 0
      ? [
          ["AUDIO", "ACTIVE"],
          ["VIDEO", "ACTIVE"],
        ]
      : activeIndex === 1
        ? [
            ["FEATURES", "EXTRACTING"],
            ["MODEL", "READY"],
          ]
        : activeIndex === 2
          ? [
              ["AUDIO", "ALIGNED"],
              ["VIDEO", "ALIGNED"],
            ]
          : activeIndex === 3
            ? [
                ["SYNC", "ANALYZING"],
                ["OFFSET", "DRIFTING"],
              ]
            : activeIndex === 4
              ? [
                  ["CROSS-ATTN", "ACTIVE"],
                  ["SYNC", "SCORING"],
                ]
              : [
                  ["ANALYSIS", "COMPLETE"],
                  ["SYNC", "ANOMALY"],
                ];

  return (
    <div className="absolute inset-0 pointer-events-none select-none" aria-hidden="true">
      <div className="absolute top-20 left-5 md:left-10 font-mono text-[11px] tracking-[0.25em] text-ink-faint">
        SYNCGUARD <span className="hidden sm:inline">// MEDIA FORENSICS</span>
      </div>

      <div key={activeIndex} className="absolute top-20 right-5 md:right-10 text-right fade-in hidden sm:block">
        {rows.map(([label, value]) => (
          <p key={label} className="font-mono text-[11px] tracking-[0.2em] text-ink-faint">
            {label} <span className="text-signal">{value}</span>
          </p>
        ))}
      </div>

      <div className="absolute bottom-6 left-5 md:left-10 font-mono text-[11px] tracking-[0.15em] text-ink-faint hidden sm:block">
        <p>TIMESTAMP 00:{seconds}</p>
        <p className="mt-0.5">FRAME {frame}</p>
      </div>

      <div className="absolute bottom-6 right-5 md:right-10 font-mono text-[11px] tracking-[0.15em] text-ink-faint hidden sm:block">
        STAGE {String(activeIndex + 1).padStart(2, "0")} / {String(totalStates).padStart(2, "0")}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Individual state contents
// ---------------------------------------------------------------------------

function FrameShell({ children }: { children: ReactNode }) {
  const [ref, tiltStyle] = useTilt<HTMLDivElement>(5);
  return (
    <div
      ref={ref}
      className="relative w-full max-w-3xl border bg-canvas-raised/80 backdrop-blur-sm"
      style={{ borderColor: "var(--page-border-strong)", ...tiltStyle }}
    >
      <CornerTicks />
      {children}
    </div>
  );
}

function CornerTicks() {
  const base = "absolute h-3 w-3 border-signal/50";
  return (
    <>
      <span className={`${base} top-0 left-0 border-t border-l`} aria-hidden="true" />
      <span className={`${base} top-0 right-0 border-t border-r`} aria-hidden="true" />
      <span className={`${base} bottom-0 left-0 border-b border-l`} aria-hidden="true" />
      <span className={`${base} bottom-0 right-0 border-b border-r`} aria-hidden="true" />
    </>
  );
}

function TrackSvg({ path, color, offsetPx = 0, jitter = false }: { path: string; color: string; offsetPx?: number; jitter?: boolean }) {
  return (
    <svg viewBox="0 0 420 28" className={`w-full h-7 ${jitter ? "animate-[jitter_0.5s_ease-in-out_infinite]" : ""}`} preserveAspectRatio="none">
      <path
        d={path}
        fill="none"
        stroke={color}
        strokeWidth={1.75}
        strokeOpacity={0.9}
        style={{ transform: `translateX(${offsetPx}px)`, transition: "transform 160ms linear" }}
      />
    </svg>
  );
}

function MagneticCta() {
  // `magnet` device: primary CTA only (devices.md §9 — a page of magnetic
  // elements is unusable). Wrapper tracks the pointer over a slightly larger
  // hit area than the button itself so the pull starts just before the hand
  // arrives; the Link's own hover/active styling is untouched.
  const [ref, style] = useMagnetic<HTMLDivElement>(0.3);
  return (
    <div ref={ref} className="inline-block p-3 -m-3" style={style}>
      <Link
        to="/analyze"
        className="inline-flex items-center gap-2 bg-signal text-[#061012] px-7 py-3.5 text-base tracking-wide hover:bg-signal-light transition-colors duration-300 active:scale-[0.98]"
      >
        Analyze Media
      </Link>
    </div>
  );
}

function RecDot() {
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.2em] text-signal">
      <span className="h-1.5 w-1.5 rounded-full bg-signal animate-[blink_1.6s_ease-in-out_infinite]" />
      Rec
    </span>
  );
}

const BAR_HEIGHTS = [30, 55, 40, 70, 45, 85, 60, 75, 50, 90, 65, 40, 72, 48, 30, 58, 42, 68, 52, 38, 60, 35, 65, 44];

function WaveformBars({ className = "" }: { className?: string }) {
  return (
    <div className={`flex items-end justify-center gap-[2px] ${className}`} aria-hidden="true">
      {BAR_HEIGHTS.map((h, i) => (
        <span
          key={i}
          className="w-[3px] bg-signal origin-bottom"
          style={{
            height: `${h}%`,
            animation: `bar-pulse ${1.1 + (i % 5) * 0.15}s ease-in-out infinite`,
            animationDelay: `${i * 35}ms`,
          }}
        />
      ))}
    </div>
  );
}

function RawMediaState() {
  return (
    <div className="grid lg:grid-cols-[1fr_1.1fr] gap-10 lg:gap-14 items-center w-full">
      <div>
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-signal/80">SyncGuard</p>
        <h1 className="font-display text-[2.875rem] leading-[1.04] sm:text-7xl sm:leading-[1.0] md:text-[4.5rem] md:leading-[0.98] mt-5">
          Verify what you hear
          <br />
          and what you see.
        </h1>
        <p className="mt-6 max-w-md text-base sm:text-lg text-ink-soft leading-relaxed">
          SyncGuard analyzes raw media as it arrives, separating audio and visual
          signal before asking whether they agree. Every verdict traces back to
          signals you can inspect yourself.
        </p>
        <div className="mt-8">
          <MagneticCta />
        </div>
      </div>

      <FrameShell>
        <div className="px-6 py-6 sm:px-8 sm:py-8">
          <div className="flex items-center justify-between mb-4">
            <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-faint">Video Frame</p>
            <RecDot />
          </div>
          <div className="relative">
            <svg viewBox="0 0 200 140" className="w-full h-28 sm:h-32">
              <path
                d="M100 24 C 70 24 54 50 54 80 C 54 108 64 128 100 128 C 136 128 146 108 146 80 C 146 50 130 24 100 24 Z"
                fill="none"
                stroke="var(--page-fg)"
                strokeOpacity="0.2"
                strokeWidth="1.25"
              />
            </svg>
            <WaveformBars className="absolute inset-x-0 bottom-0 h-10 opacity-70" />
          </div>
          <div className="mt-6 space-y-3">
            <Row label="AUDIO" color="var(--page-signal)" path={AUDIO_PATH} />
            <Row label="VIDEO" color="var(--page-fusion)" path={VIDEO_PATH} />
            <Row label="SYNC" color="#edeff3" path={AUDIO_PATH} dim />
          </div>
        </div>
      </FrameShell>
    </div>
  );
}

function Row({ label, color, path, dim = false }: { label: string; color: string; path: string; dim?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-12 shrink-0 font-mono text-[10px] tracking-[0.15em] text-ink-faint">{label}</span>
      <div className="flex-1" style={{ opacity: dim ? 0.3 : 1 }}>
        <TrackSvg path={path} color={color} />
      </div>
    </div>
  );
}

function SignalsExtractedState() {
  return (
    <div className="w-full max-w-2xl text-center">
      <p className="font-mono text-xs uppercase tracking-[0.3em] text-fusion/90">Signals Extracted</p>
      <h2 className="font-display text-3xl sm:text-4xl mt-4">Every clip breaks into separable signals.</h2>
      <p className="mt-4 text-sm sm:text-base text-ink-soft leading-relaxed max-w-md mx-auto">
        Landmarks, frames, waveform and spectrogram are pulled apart independently,
        before anything is compared.
      </p>

      <div className="mt-10 grid sm:grid-cols-2 gap-5 text-left">
        <TagGroup tone="fusion" title="Video" tags={["Face", "Lips", "Frames", "Motion"]} visual={<LandmarkMesh />} />
        <TagGroup tone="signal" title="Audio" tags={["Waveform", "Spectrogram", "Temporal Features"]} visual={<SpectrogramBars />} />
      </div>
    </div>
  );
}

function TagGroup({
  tone,
  title,
  tags,
  visual,
}: {
  tone: "signal" | "fusion";
  title: string;
  tags: string[];
  visual?: ReactNode;
}) {
  const color = tone === "signal" ? "text-signal" : "text-fusion";
  const border = tone === "signal" ? "border-signal/30" : "border-fusion/30";
  return (
    <div className="border bg-canvas-raised/60 px-5 py-5" style={{ borderColor: "var(--page-border)" }}>
      <p className={`font-mono text-[11px] uppercase tracking-[0.2em] ${color}`}>{title}</p>
      {visual && <div className="mt-4 h-16 flex items-center justify-center">{visual}</div>}
      <div className="mt-4 flex flex-wrap gap-2">
        {tags.map((tag) => (
          <span key={tag} className={`font-mono text-[11px] uppercase tracking-[0.1em] border ${border} px-2.5 py-1 text-ink-soft`}>
            {tag}
          </span>
        ))}
      </div>
    </div>
  );
}

// Nine landmark points around an abstract face oval (eyes, nose, mouth,
// jawline) — a cheap visual nod to the real MediaPipe landmark extraction
// the backend runs, not a literal reproduction of its output.
const LANDMARKS: [number, number][] = [
  [30, 24], [48, 18], [66, 24], // brow line
  [26, 40], [70, 40], // cheekbones
  [48, 48], // nose
  [36, 62], [48, 66], [60, 62], // mouth
];
const MESH_EDGES: [number, number][] = [
  [0, 1], [1, 2], [0, 3], [2, 4], [3, 5], [4, 5], [5, 6], [5, 8], [6, 7], [7, 8],
];

function LandmarkMesh() {
  return (
    <svg viewBox="0 0 96 80" className="h-full w-auto opacity-90">
      <ellipse cx="48" cy="42" rx="26" ry="32" fill="none" stroke="var(--page-fusion)" strokeOpacity="0.2" strokeWidth="1" />
      {MESH_EDGES.map(([a, b], i) => {
        const [x1, y1] = LANDMARKS[a];
        const [x2, y2] = LANDMARKS[b];
        return (
          <line
            key={i}
            x1={x1}
            y1={y1}
            x2={x2}
            y2={y2}
            stroke="var(--page-fusion)"
            strokeWidth="0.75"
            style={{ animation: "dot-pulse 2.6s ease-in-out infinite", animationDelay: `${i * 90}ms` }}
          />
        );
      })}
      {LANDMARKS.map(([x, y], i) => (
        <circle
          key={i}
          cx={x}
          cy={y}
          r="1.8"
          fill="var(--page-fusion)"
          style={{ animation: "dot-pulse 2.4s ease-in-out infinite", animationDelay: `${i * 180}ms`, opacity: 0.8 }}
        />
      ))}
    </svg>
  );
}

const SPECTRO_BARS = [20, 60, 35, 80, 45, 90, 55, 70, 30, 85, 50, 65, 40, 75, 25, 95, 60, 45, 80, 35, 70, 50, 30, 65];

function SpectrogramBars() {
  return (
    <div className="flex items-end justify-center gap-[2px] h-full w-full" aria-hidden="true">
      {SPECTRO_BARS.map((h, i) => (
        <span
          key={i}
          className="w-[3px] bg-signal origin-bottom rounded-[0.5px]"
          style={{
            height: `${h}%`,
            opacity: 0.35 + (h / 100) * 0.55,
            animation: `bar-pulse ${0.9 + (i % 6) * 0.12}s ease-in-out infinite`,
            animationDelay: `${i * 45}ms`,
          }}
        />
      ))}
    </div>
  );
}

function AlignmentState({
  hover,
  onHover,
}: {
  hover: "audio" | "video" | "sync" | null;
  onHover: (v: "audio" | "video" | "sync" | null) => void;
}) {
  const tooltip =
    hover === "audio" ? "Audio Feature Stream" : hover === "video" ? "Visual Feature Stream" : hover === "sync" ? "Cross-Modal Alignment" : null;

  return (
    <div className="w-full max-w-2xl">
      <p className="font-mono text-xs uppercase tracking-[0.3em] text-signal/80 text-center">Temporal Alignment</p>
      <h2 className="font-display text-3xl sm:text-4xl mt-4 text-center">Two independent signals, one timeline.</h2>
      <p className="mt-4 text-sm sm:text-base text-ink-soft leading-relaxed text-center max-w-md mx-auto">
        Each stream keeps its own clock until SyncGuard checks whether they still
        agree.
      </p>

      <FrameShell>
        <div className="px-6 py-8 sm:px-10 sm:py-10 space-y-6">
          <HoverRow label="AUDIO" color="var(--page-signal)" path={AUDIO_PATH} active={hover === "audio" || hover === "sync"} speed={5.5} onEnter={() => onHover("audio")} onLeave={() => onHover(null)} />
          <HoverRow label="VIDEO" color="var(--page-fusion)" path={VIDEO_PATH} active={hover === "video" || hover === "sync"} speed={7} onEnter={() => onHover("video")} onLeave={() => onHover(null)} />
          <div
            className="flex items-center gap-3 cursor-default"
            onMouseEnter={() => onHover("sync")}
            onMouseLeave={() => onHover(null)}
          >
            <span className="w-12 shrink-0 font-mono text-[10px] tracking-[0.15em] text-ink-faint">SYNC</span>
            <div className="flex-1 flex items-center">
              <span className="h-px flex-1 bg-ink/15" />
              <span className="h-2.5 w-2.5 rounded-full bg-[#edeff3] mx-1" />
              <span className="h-px flex-1 bg-ink/15" />
            </div>
          </div>
          <p className="h-4 text-center font-mono text-[11px] uppercase tracking-[0.2em] text-signal transition-opacity duration-200" style={{ opacity: tooltip ? 1 : 0 }}>
            {tooltip}
          </p>
        </div>
      </FrameShell>
    </div>
  );
}

function HoverRow({
  label,
  color,
  path,
  active,
  speed = 6,
  onEnter,
  onLeave,
}: {
  label: string;
  color: string;
  path: string;
  active: boolean;
  speed?: number;
  onEnter: () => void;
  onLeave: () => void;
}) {
  return (
    <div className="flex items-center gap-3 cursor-default" onMouseEnter={onEnter} onMouseLeave={onLeave}>
      <span className="w-12 shrink-0 font-mono text-[10px] tracking-[0.15em] text-ink-faint">{label}</span>
      <div className="relative flex-1 transition-opacity duration-200" style={{ opacity: active ? 1 : 0.65 }}>
        <TrackSvg path={path} color={color} />
        <span
          className="absolute top-1/2 h-1.5 w-1.5 -translate-y-1/2 rounded-full"
          style={{ backgroundColor: color, animation: `playhead-slide ${speed}s ease-in-out infinite` }}
          aria-hidden="true"
        />
      </div>
    </div>
  );
}

function DriftState({ local, mismatchDetected }: { local: number; mismatchDetected: boolean }) {
  const offset = local * 34;
  return (
    <div className="w-full max-w-2xl">
      <div className="flex items-center justify-between">
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-anomaly/90">Sync Drift</p>
        <p className="font-mono text-[11px] text-ink-faint">00:13.42</p>
      </div>
      <h2 className="font-display text-3xl sm:text-4xl mt-4">
        A convincing clip can still disagree with itself.
      </h2>

      <FrameShell>
        <div className="px-6 py-8 sm:px-10 sm:py-10 space-y-5">
          <Row label="AUDIO" color="var(--page-signal)" path={AUDIO_PATH} />
          <div className="flex items-center gap-3">
            <span className="w-12 shrink-0 font-mono text-[10px] tracking-[0.15em] text-ink-faint">VIDEO</span>
            <div className="flex-1">
              <TrackSvg path={VIDEO_PATH} color="var(--page-fusion)" offsetPx={offset} jitter={mismatchDetected} />
            </div>
          </div>

          <div className="flex items-center justify-center gap-2 pt-2">
            <span className="font-mono text-[11px] text-ink-faint">OFFSET</span>
            <span
              className="h-px bg-anomaly/60 transition-[width] duration-150"
              style={{ width: `${8 + offset}px` }}
            />
            <span className="font-mono text-[11px] text-anomaly">{(local * 184).toFixed(0)}ms</span>
          </div>

          <p
            className="flex items-center justify-center gap-2 text-center font-mono text-sm uppercase tracking-[0.2em] text-anomaly transition-opacity duration-300"
            style={{ opacity: mismatchDetected ? 1 : 0 }}
          >
            <span className="h-1.5 w-1.5 rounded-full bg-anomaly" style={{ animation: "pulse-signal 1s ease-in-out infinite" }} aria-hidden="true" />
            Temporal mismatch detected
          </p>
        </div>
      </FrameShell>
    </div>
  );
}

const ANALYSIS_TAGS = ["Audio Features", "Visual Features", "Temporal Features", "Sync Representation"];

function AnalysisState({ local }: { local: number }) {
  const [tiltRef, tiltStyle] = useTilt<HTMLDivElement>(8);
  return (
    <div className="w-full max-w-2xl text-center">
      <p className="font-mono text-xs uppercase tracking-[0.3em] text-fusion/90">Cross-Modal Analysis</p>
      <h2 className="font-display text-3xl sm:text-4xl mt-4">
        Audio and visual representations fuse through bidirectional cross-attention.
      </h2>

      <div className="mt-10 flex items-center justify-center gap-4">
        <MiniLabel text="AUDIO" color="var(--page-signal)" />
        <ConnectorDiag />
        <div ref={tiltRef} className="relative">
          <PulseRing />
          <div
            className="relative border px-6 py-5 bg-canvas-raised"
            style={{ borderColor: "var(--page-signal)", animation: "pulse-signal 2.2s ease-in-out infinite", ...tiltStyle }}
          >
            <p className="font-display text-sm tracking-tight">SyncGuard</p>
            <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-faint mt-1">Cross-Attention</p>
          </div>
        </div>
        <ConnectorDiag flip />
        <MiniLabel text="VIDEO" color="var(--page-fusion)" />
      </div>

      <div className="mt-8 flex flex-wrap justify-center gap-2">
        {ANALYSIS_TAGS.map((tag, i) => (
          <span
            key={tag}
            className="font-mono text-[10px] uppercase tracking-[0.1em] border border-ink/15 px-2.5 py-1 text-ink-soft transition-opacity duration-300"
            style={{ opacity: local > (i + 1) / (ANALYSIS_TAGS.length + 1) ? 1 : 0.15 }}
          >
            {tag}
          </span>
        ))}
      </div>
    </div>
  );
}

function MiniLabel({ text, color }: { text: string; color: string }) {
  return (
    <span className="font-mono text-[11px] uppercase tracking-[0.15em]" style={{ color }}>
      {text}
    </span>
  );
}

function ConnectorDiag({ flip = false }: { flip?: boolean }) {
  return (
    <svg width="40" height="2" viewBox="0 0 40 2" className="hidden sm:block" preserveAspectRatio="none">
      <line
        x1={flip ? 40 : 0}
        y1="1"
        x2={flip ? 0 : 40}
        y2="1"
        stroke="var(--page-signal)"
        strokeOpacity="0.5"
        strokeWidth="1.5"
        strokeDasharray="4 4"
        style={{ animation: `dash-flow 1.4s linear infinite ${flip ? "reverse" : "normal"}` }}
      />
    </svg>
  );
}

function PulseRing() {
  return (
    <>
      <span
        className="absolute inset-0 border border-signal/60 pointer-events-none"
        style={{ animation: "ring-pulse 2.4s ease-out infinite" }}
        aria-hidden="true"
      />
      <span
        className="absolute inset-0 border border-signal/60 pointer-events-none"
        style={{ animation: "ring-pulse 2.4s ease-out infinite", animationDelay: "1.2s" }}
        aria-hidden="true"
      />
    </>
  );
}

function VerdictState() {
  return (
    <div className="w-full max-w-md">
      <FrameShell>
        <div className="flex items-center justify-between px-6 py-4 border-b" style={{ borderColor: "var(--page-border)" }}>
          <p className="font-mono text-sm uppercase tracking-[0.2em] text-ink-soft">SyncGuard Analysis</p>
          <span className="font-mono text-[11px] uppercase tracking-[0.2em] text-anomaly border border-anomaly/40 px-2 py-0.5">
            Demo Visualization
          </span>
        </div>
        <div className="px-6 py-7 grid grid-cols-2 gap-6 items-center">
          <Field label="Status" value="Suspicious" />
          <RadialGauge value={62} label="Sync Consistency" />
          <Field label="Temporal Offset" value="+184ms" />
          <Field label="Anomaly" value="Detected" />
        </div>
        <p className="px-6 pb-6 text-sm text-ink-faint leading-relaxed">
          Illustrative only. Run your own media through the real pipeline on the Analyze page.
        </p>
      </FrameShell>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-faint">{label}</p>
      <p className="font-display text-2xl mt-1.5 text-anomaly">{value}</p>
    </div>
  );
}

function RadialGauge({ value, label }: { value: number; label: string }) {
  const r = 20;
  const circumference = 2 * Math.PI * r;
  const offset = circumference * (1 - value / 100);
  return (
    <div className="flex items-center gap-3">
      <svg viewBox="0 0 48 48" className="h-12 w-12 shrink-0 -rotate-90">
        <circle cx="24" cy="24" r={r} fill="none" stroke="var(--page-border-strong)" strokeWidth="3" />
        <circle
          cx="24"
          cy="24"
          r={r}
          fill="none"
          stroke="var(--page-anomaly)"
          strokeWidth="3"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
        />
      </svg>
      <div>
        <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-ink-faint">{label}</p>
        <p className="font-display text-2xl mt-1 text-anomaly">{value}%</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// prefers-reduced-motion fallback — same six beats, same content, laid out
// as plain stacked panels in document flow. No position:sticky, no
// scroll-driven transform: everything is simply visible.
// ---------------------------------------------------------------------------

function StaticNarrative() {
  return (
    <div className="container-page py-20 flex flex-col gap-14">
      <RawMediaState />
      <StaticPanel><SignalsExtractedState /></StaticPanel>
      <StaticPanel><AlignmentState hover={null} onHover={() => {}} /></StaticPanel>
      <StaticPanel><DriftState local={1} mismatchDetected /></StaticPanel>
      <StaticPanel><AnalysisState local={1} /></StaticPanel>
      <StaticPanel><VerdictState /></StaticPanel>
    </div>
  );
}

function StaticPanel({ children }: { children: ReactNode }) {
  return <div className="flex justify-center">{children}</div>;
}
