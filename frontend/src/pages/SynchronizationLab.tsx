import { useEffect, useRef, useState } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import SyncTimeline from "../components/SyncTimeline";
import MediaFrame from "../components/MediaFrame";
import ScoreCard from "../components/ScoreCard";
import RadialGauge from "../components/RadialGauge";
import Reveal from "../components/scroll/Reveal";
import { fetchLabSamples, runLabAnalysis } from "../services/api";
import type { LabSample, SyncLabResult } from "../services/api";

type ShiftResults = Record<number, SyncLabResult | "error" | undefined>;

export default function SynchronizationLab() {
  const [samples, setSamples] = useState<LabSample[]>([]);
  const [shifts, setShifts] = useState<number[]>([]);
  const [selectedSample, setSelectedSample] = useState<string | null>(null);
  const [selectedShift, setSelectedShift] = useState<number>(0);
  const [results, setResults] = useState<ShiftResults>({});
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [samplesLoaded, setSamplesLoaded] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    fetchLabSamples()
      .then((res) => {
        setSamples(res.samples);
        setShifts(res.available_shifts);
        if (res.samples.length > 0) setSelectedSample(res.samples[0].id);
      })
      .catch(() => setLoadError("Unable to reach the SyncGuard backend to load lab samples."))
      .finally(() => setSamplesLoaded(true));
  }, []);

  useEffect(() => {
    if (!selectedSample || shifts.length === 0) return;
    let cancelled = false;
    setLoading(true);
    setResults({});
    setSelectedShift(0);

    Promise.all(
      shifts.map((shift) =>
        runLabAnalysis(selectedSample, shift)
          .then((r) => [shift, r] as const)
          .catch(() => [shift, "error"] as const),
      ),
    ).then((entries) => {
      if (cancelled) return;
      const next: ShiftResults = {};
      for (const [shift, r] of entries) next[shift] = r;
      setResults(next);
      setLoading(false);
    });

    return () => {
      cancelled = true;
    };
  }, [selectedSample, shifts]);

  const activeSampleMeta = samples.find((s) => s.id === selectedSample);
  const activeResult = results[selectedShift];
  const hasActiveResult = activeResult && activeResult !== "error";

  const chartData = shifts
    .map((shift) => {
      const r = results[shift];
      return r && r !== "error" ? { shift, score: r.aggregate_sync_score } : null;
    })
    .filter((d): d is { shift: number; score: number } => d !== null);

  const handleSeek = (t: number) => {
    if (videoRef.current) videoRef.current.currentTime = t;
  };

  return (
    <div className="container-page py-10 md:py-14 bg-grain relative">
      <Reveal>
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-signal/80">SyncGuard // Controlled Experiment</p>
        <h1 className="font-display text-4xl sm:text-5xl md:text-6xl mt-3 max-w-2xl leading-tight">Synchronization Lab</h1>
        <p className="mt-3 max-w-2xl text-sm sm:text-base text-ink-soft leading-relaxed">
          Explore how controlled temporal offsets affect audio-visual synchronization analysis. This is not a
          deepfake generator or simulator — it demonstrates the trained model's sensitivity to timing shifts on a
          fixed set of real clips.
        </p>
      </Reveal>

      <Reveal className="mt-6 max-w-2xl border border-[color:var(--page-anomaly)] px-5 py-4 bg-anomaly/5">
        <DisclaimerCard />
      </Reveal>

      {loadError && (
        <div className="mt-10 max-w-xl border px-5 py-4 bg-anomaly/5" style={{ borderColor: "var(--page-anomaly)" }}>
          <p className="font-mono text-xs uppercase tracking-[0.15em] text-anomaly">Connection Error</p>
          <p className="mt-2 text-sm text-ink-soft">{loadError}</p>
        </div>
      )}

      {!loadError && samplesLoaded && samples.length === 0 && (
        <div className="mt-10 max-w-xl border bg-canvas-raised/30 px-5 py-4" style={{ borderColor: "var(--page-border)", borderStyle: "dashed" }}>
          <p className="font-mono text-xs uppercase tracking-[0.15em] text-ink-faint">No Samples Available</p>
          <p className="mt-2 text-sm text-ink-soft leading-relaxed">
            No curated Synchronization Lab samples are available in this deployment. This feature depends on
            locally cached LAV-DF sample clips, which are intentionally excluded from deployed builds (see
            docs/datasets.md). Use the Analyze page with your own audio-visual upload instead.
          </p>
        </div>
      )}

      {samples.length > 0 && (
        <>
          <Reveal className="mt-12">
            <p className="font-mono text-xs uppercase tracking-[0.25em] text-signal/80 mb-4">Sample</p>
            <div className="flex flex-wrap gap-3">
              {samples.map((s) => {
                const active = selectedSample === s.id;
                return (
                  <button
                    key={s.id}
                    onClick={() => setSelectedSample(s.id)}
                    className={`font-mono text-xs uppercase tracking-[0.1em] px-4 py-2.5 border transition-all duration-200 ${
                      active
                        ? "border-signal text-signal shadow-[0_0_0_1px_rgba(34,211,238,0.3)]"
                        : "text-ink-faint hover:text-ink-soft hover:border-ink-faint/50"
                    }`}
                    style={{ borderColor: active ? undefined : "var(--page-border)" }}
                  >
                    {s.label}
                  </button>
                );
              })}
            </div>
          </Reveal>

          <div className="mt-8 grid lg:grid-cols-2 gap-10 lg:gap-12">
            <div>
              {activeSampleMeta && (
                <MediaFrame
                  key={activeSampleMeta.id}
                  src={activeSampleMeta.video_url}
                  label={`Sample // ${activeSampleMeta.id}`}
                  videoRef={videoRef}
                />
              )}

              <p className="mt-6 font-mono text-xs uppercase tracking-[0.25em] text-signal/80 mb-3">Temporal Shift</p>
              <div className="flex flex-wrap gap-2">
                {shifts.map((shift) => {
                  const r = results[shift];
                  const state = loading && !r ? "pending" : r === "error" ? "error" : r ? "ready" : "pending";
                  const active = selectedShift === shift;
                  return (
                    <button
                      key={shift}
                      disabled={state !== "ready"}
                      onClick={() => setSelectedShift(shift)}
                      className={`inline-flex items-center gap-2 font-mono text-xs px-3.5 py-2 border transition-all duration-200 disabled:cursor-wait ${
                        active
                          ? "border-signal text-signal"
                          : state === "error"
                            ? "border-anomaly/40 text-anomaly/70"
                            : "text-ink-faint hover:text-ink-soft hover:border-ink-faint/50 disabled:opacity-50"
                      }`}
                      style={{ borderColor: active ? undefined : state === "error" ? undefined : "var(--page-border)" }}
                    >
                      <ShiftStatusDot state={state} active={active} />
                      {shift === 0 ? "Original" : `+${shift.toFixed(1)}s`}
                    </button>
                  );
                })}
              </div>

              {loading && (
                <p className="mt-4 flex items-center gap-2 font-mono text-xs text-ink-faint">
                  <span className="h-3 w-3 rounded-full border border-signal border-t-transparent animate-spin" />
                  Running the sync model across all shift values…
                </p>
              )}

              {hasActiveResult && (
                <div className="mt-8 border bg-canvas-raised/40 p-5" style={{ borderColor: "var(--page-border-strong)" }}>
                  <RadialGauge
                    value={activeResult.aggregate_sync_score * 100}
                    label={selectedShift === 0 ? "Original · Sync Score" : `+${selectedShift.toFixed(1)}s · Sync Score`}
                    tone={activeResult.aggregate_sync_score >= 0.5 ? "signal" : "anomaly"}
                    size={104}
                  />
                  <div className="mt-6 grid grid-cols-2 gap-3">
                    <ScoreCard label="Shift" value={selectedShift === 0 ? "Original" : `+${selectedShift.toFixed(1)}s`} />
                    <ScoreCard label="Windows" value={String(activeResult.timing_metadata?.num_windows ?? "—")} />
                  </div>
                </div>
              )}
            </div>

            <div>
              <div className="border bg-canvas-raised/40 p-5" style={{ borderColor: "var(--page-border-strong)" }}>
                <p className="font-mono text-xs uppercase tracking-[0.2em] text-ink-faint mb-4">
                  Temporal Offset vs. Synchronization Score
                </p>
                {chartData.length > 1 ? (
                  <div className="w-full h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--page-border)" vertical={false} />
                        <XAxis
                          dataKey="shift"
                          tickFormatter={(v) => `+${v}s`}
                          stroke="var(--page-muted)"
                          fontSize={11}
                          tickLine={false}
                        />
                        <YAxis domain={[0, 1]} stroke="var(--page-muted)" fontSize={11} tickLine={false} />
                        <Tooltip
                          contentStyle={{
                            background: "var(--page-surface)",
                            border: "1px solid var(--page-border)",
                            borderRadius: 2,
                            fontSize: 12,
                          }}
                          formatter={(value) => [Number(value).toFixed(4), "Aggregate sync score"]}
                          labelFormatter={(label) => (Number(label) === 0 ? "Original (no shift)" : `+${label}s shift`)}
                        />
                        <Line
                          type="monotone"
                          dataKey="score"
                          stroke="#22d3ee"
                          strokeWidth={2}
                          dot={{ r: 4, fill: "#22d3ee", strokeWidth: 0 }}
                          activeDot={{ r: 6, fill: "#67e8f9", strokeWidth: 0 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div
                    className="h-64 border bg-grid flex items-center justify-center font-mono text-xs text-ink-faint"
                    style={{ borderColor: "var(--page-border)", borderStyle: "dashed" }}
                  >
                    Waiting for shift results…
                  </div>
                )}
              </div>

              {hasActiveResult && activeResult.per_window_sync_scores && (
                <div className="mt-6 border bg-canvas-raised/40 p-5" style={{ borderColor: "var(--page-border-strong)" }}>
                  <p className="font-mono text-xs uppercase tracking-[0.2em] text-ink-faint mb-4">
                    Per-Window Scores — {selectedShift === 0 ? "Original" : `+${selectedShift.toFixed(1)}s shift`}
                  </p>
                  <SyncTimeline
                    scores={activeResult.per_window_sync_scores}
                    metadata={activeResult.timing_metadata}
                    onSeek={handleSeek}
                  />
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function DisclaimerCard() {
  return (
    <>
      <p className="font-mono text-xs uppercase tracking-[0.15em] text-anomaly">Scientific Disclaimer</p>
      <p className="mt-2 text-sm text-ink-soft leading-relaxed">
        These results demonstrate sensitivity to temporal offsets. They should not be interpreted as
        real-world deepfake detection accuracy.
      </p>
      <p className="mt-2 text-sm text-ink-soft leading-relaxed">
        Manipulation and desynchronization are distinct properties. A manipulated video can remain
        synchronized.
      </p>
    </>
  );
}

function ShiftStatusDot({ state, active }: { state: "pending" | "ready" | "error"; active: boolean }) {
  if (state === "error") {
    return <span className="h-1.5 w-1.5 rounded-full bg-anomaly" aria-hidden="true" />;
  }
  if (state === "pending") {
    return <span className="h-1.5 w-1.5 rounded-full border border-ink-faint/50" aria-hidden="true" />;
  }
  return (
    <span
      className={`h-1.5 w-1.5 rounded-full ${active ? "bg-signal" : "border border-signal"}`}
      aria-hidden="true"
    />
  );
}
