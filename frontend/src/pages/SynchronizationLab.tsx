import { useEffect, useState } from "react";
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

  useEffect(() => {
    fetchLabSamples()
      .then((res) => {
        setSamples(res.samples);
        setShifts(res.available_shifts);
        if (res.samples.length > 0) setSelectedSample(res.samples[0].id);
      })
      .catch(() => setLoadError("Unable to reach the SyncGuard backend to load lab samples."));
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

  return (
    <div className="container-page py-16">
      <p className="text-xs uppercase tracking-[0.2em] opacity-60">Controlled Experiment</p>
      <h1 className="font-display text-4xl md:text-5xl mt-3 max-w-2xl leading-tight">Synchronization Lab</h1>
      <p className="mt-4 max-w-2xl opacity-70 leading-relaxed">
        Explore how controlled temporal offsets affect audio-visual synchronization analysis. This is not a
        deepfake generator or simulator — it demonstrates the trained model's sensitivity to timing shifts on a
        fixed set of real clips.
      </p>

      <div
        className="mt-8 border-l-2 pl-5 py-1 max-w-2xl text-xs leading-relaxed opacity-75"
        style={{ borderColor: "var(--copper)" }}
      >
        <p className="uppercase tracking-widest text-copper-dark font-medium mb-2">Scientific disclaimer</p>
        <p>
          These results demonstrate sensitivity to temporal offsets. They should not be interpreted as
          real-world deepfake detection accuracy.
        </p>
        <p className="mt-2">
          Manipulation and desynchronization are distinct properties. A manipulated video can remain
          synchronized.
        </p>
      </div>

      {loadError && <p className="mt-10 text-sm text-copper-dark">{loadError}</p>}

      {samples.length > 0 && (
        <>
          <div className="mt-12">
            <p className="text-xs uppercase tracking-widest opacity-60 mb-4">Sample</p>
            <div className="flex flex-wrap gap-3">
              {samples.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setSelectedSample(s.id)}
                  className={`px-4 py-2 text-sm border transition-colors duration-200 ${
                    selectedSample === s.id ? "border-sage text-sage-light" : "opacity-70 hover:opacity-100"
                  }`}
                  style={{ borderColor: selectedSample === s.id ? undefined : "var(--page-border)" }}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-10 grid md:grid-cols-2 gap-10">
            <div>
              {activeSampleMeta && (
                <video
                  key={activeSampleMeta.id}
                  src={activeSampleMeta.video_url}
                  controls
                  className="w-full border"
                  style={{ borderColor: "var(--page-border)" }}
                />
              )}

              <p className="mt-6 text-xs uppercase tracking-widest opacity-60 mb-3">Temporal Shift</p>
              <div className="flex flex-wrap gap-2">
                {shifts.map((shift) => {
                  const r = results[shift];
                  const state = loading && !r ? "pending" : r === "error" ? "error" : r ? "ready" : "pending";
                  return (
                    <button
                      key={shift}
                      disabled={state !== "ready"}
                      onClick={() => setSelectedShift(shift)}
                      className={`px-4 py-2 text-sm border transition-colors duration-200 disabled:opacity-40 disabled:cursor-wait ${
                        selectedShift === shift ? "border-sage text-sage-light" : "opacity-70 hover:opacity-100"
                      }`}
                      style={{ borderColor: selectedShift === shift ? undefined : "var(--page-border)" }}
                    >
                      {shift === 0 ? "Original" : `+${shift.toFixed(1)}s`}
                    </button>
                  );
                })}
              </div>

              {loading && <p className="mt-4 text-xs opacity-50">Running the sync model across all shift values…</p>}

              {hasActiveResult && (
                <div className="mt-8 grid grid-cols-3 gap-3">
                  <StatCard label="Shift" value={selectedShift === 0 ? "Original" : `+${selectedShift.toFixed(1)}s`} />
                  <StatCard label="Sync Score" value={activeResult.aggregate_sync_score.toFixed(4)} />
                  <StatCard label="Windows" value={String(activeResult.timing_metadata?.num_windows ?? "—")} />
                </div>
              )}
            </div>

            <div>
              <p className="text-xs uppercase tracking-widest opacity-60 mb-3">Temporal Offset vs. Synchronization Score</p>
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
                      <Line type="monotone" dataKey="score" stroke="#7c9478" strokeWidth={2} dot={{ r: 4 }} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div
                  className="h-64 border border-dashed flex items-center justify-center text-sm opacity-50"
                  style={{ borderColor: "var(--page-border)" }}
                >
                  Waiting for shift results…
                </div>
              )}

              {hasActiveResult && activeResult.per_window_sync_scores && (
                <div className="mt-8">
                  <p className="text-xs uppercase tracking-widest opacity-60 mb-3">
                    Per-Window Scores — {selectedShift === 0 ? "Original" : `+${selectedShift.toFixed(1)}s shift`}
                  </p>
                  <SyncTimeline scores={activeResult.per_window_sync_scores} metadata={activeResult.timing_metadata} />
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="border px-4 py-3" style={{ borderColor: "var(--page-border)" }}>
      <p className="text-xs uppercase tracking-widest opacity-60">{label}</p>
      <p className="mt-1 font-display text-lg">{value}</p>
    </div>
  );
}
