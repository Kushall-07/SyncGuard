import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AVTimingMetadata } from "../services/api";
import { SYNC_THRESHOLD as THRESHOLD, buildTimelineData, findLowestPoint, formatTimestamp, perWindowSeconds } from "../lib/timeline";

interface SyncTimelineProps {
  scores: number[];
  metadata: AVTimingMetadata | null | undefined;
  onSeek?: (timeSeconds: number) => void;
  selectedTime?: number | null;
}

export default function SyncTimeline({ scores, metadata, onSeek, selectedTime }: SyncTimelineProps) {
  if (!scores || scores.length === 0) {
    return (
      <div
        className="h-64 border border-dashed flex items-center justify-center text-sm opacity-50 text-center px-8"
        style={{ borderColor: "var(--page-border)" }}
      >
        No per-window synchronization data was returned for this analysis.
      </div>
    );
  }

  const fps = metadata?.fps && metadata.fps > 0 ? metadata.fps : null;
  const stepSeconds = perWindowSeconds(scores, metadata);
  const data = buildTimelineData(scores, metadata);
  const lowest = findLowestPoint(data)!;
  const hasLowRegion = lowest.score < THRESHOLD;

  return (
    <div>
      {!fps && (
        <p className="mb-2 text-xs text-copper-dark">
          Frame rate metadata was unavailable — timestamps are estimated from window count.
        </p>
      )}
      <div className="w-full h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 8, right: 8, left: -20, bottom: 0 }}
            onClick={(state) => {
              if (onSeek && state && typeof state.activeLabel === "number") {
                onSeek(state.activeLabel);
              }
            }}
            className={onSeek ? "cursor-pointer" : undefined}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="var(--page-border)" vertical={false} />
            <XAxis
              dataKey="time"
              tickFormatter={(t) => `${t}s`}
              stroke="var(--page-muted)"
              fontSize={11}
              tickLine={false}
              label={{ value: "Time (s)", position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--page-muted)" }}
            />
            <YAxis
              domain={[0, 1]}
              stroke="var(--page-muted)"
              fontSize={11}
              tickLine={false}
              label={{ value: "Sync score", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--page-muted)" }}
            />
            <ReferenceLine y={THRESHOLD} stroke="var(--page-muted)" strokeDasharray="4 4" />
            <Tooltip
              cursor={{ fill: "var(--page-border)" }}
              contentStyle={{
                background: "var(--page-surface)",
                border: "1px solid var(--page-border)",
                borderRadius: 2,
                fontSize: 12,
              }}
              content={({ active, payload }) => {
                if (!active || !payload || payload.length === 0) return null;
                const point = payload[0].payload as { time: number; score: number; index: number };
                const status = point.score >= THRESHOLD ? "Higher synchronization" : "Lower synchronization";
                return (
                  <div
                    style={{
                      background: "var(--page-surface)",
                      border: "1px solid var(--page-border)",
                      borderRadius: 2,
                      padding: "8px 12px",
                      fontSize: 12,
                    }}
                  >
                    <p className="font-display text-sm">{formatTimestamp(point.time)}</p>
                    <p className="opacity-70 mt-1">Window {point.index + 1}</p>
                    <p className="opacity-70">Sync score: {point.score.toFixed(3)}</p>
                    <p className={point.score >= THRESHOLD ? "text-sage-dark" : "text-copper-dark"}>{status}</p>
                  </div>
                );
              }}
            />
            <Bar dataKey="score" radius={[2, 2, 0, 0]}>
              {data.map((entry) => (
                <Cell
                  key={entry.index}
                  fill={entry.score >= THRESHOLD ? "#7c9478" : "#ab7440"}
                  fillOpacity={selectedTime != null && Math.abs(entry.time - selectedTime) < stepSeconds ? 1 : 0.85}
                  stroke={
                    selectedTime != null && Math.abs(entry.time - selectedTime) < stepSeconds
                      ? "var(--page-fg)"
                      : "none"
                  }
                  strokeWidth={1}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2 text-xs opacity-60">
        <LegendDot color="#7c9478" label={`At/above threshold (${THRESHOLD})`} />
        <LegendDot color="#ab7440" label="Below threshold" />
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3 border-t border-dashed" style={{ borderColor: "var(--page-muted)" }} />
          Reference threshold
        </span>
        {onSeek && <span>Click a bar to seek the video preview.</span>}
      </div>

      {hasLowRegion && (
        <p className="mt-4 text-xs leading-relaxed border-l-2 pl-4" style={{ borderColor: "var(--copper)" }}>
          <span className="uppercase tracking-widest text-copper-dark font-medium">Lower synchronization region</span>
          <br />
          Window {lowest.index + 1} at {formatTimestamp(lowest.time)} scored {lowest.score.toFixed(3)}, the lowest in
          this clip. This indicates weaker temporal correspondence at that point — not confirmed manipulation.
        </p>
      )}
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-sm" style={{ background: color }} aria-hidden="true" />
      {label}
    </span>
  );
}
