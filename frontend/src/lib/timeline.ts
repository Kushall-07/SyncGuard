// Pure timeline math shared by SyncTimeline and its tests. Kept separate from the
// component so window-time derivation can be unit-tested without rendering Recharts
// (which needs real layout dimensions jsdom doesn't provide).

import type { AVTimingMetadata } from "../services/api";

export const SYNC_THRESHOLD = 0.5;

export interface TimelinePoint {
  time: number;
  score: number;
  index: number;
}

// `fps` gives per-frame time directly. `window_seconds` from the predictor is the
// total aligned clip duration (T_v / fps), not a per-window duration — dividing by
// the window count recovers seconds-per-window as a fallback when fps is missing.
export function perWindowSeconds(scores: number[], metadata: AVTimingMetadata | null | undefined): number {
  if (metadata?.fps && metadata.fps > 0) return 1 / metadata.fps;
  if (metadata?.window_seconds && scores.length > 0) return metadata.window_seconds / scores.length;
  return 1;
}

export function buildTimelineData(scores: number[], metadata: AVTimingMetadata | null | undefined): TimelinePoint[] {
  const step = perWindowSeconds(scores, metadata);
  return scores.map((score, i) => ({
    time: Number((i * step).toFixed(3)),
    score,
    index: i,
  }));
}

export function findLowestPoint(data: TimelinePoint[]): TimelinePoint | null {
  if (data.length === 0) return null;
  return data.reduce((min, d) => (d.score < min.score ? d : min), data[0]);
}

export function formatTimestamp(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds - m * 60;
  return `${m.toString().padStart(2, "0")}:${s.toFixed(2).padStart(5, "0")}`;
}
