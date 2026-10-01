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

// In "windowed" mode each score covers `window_seconds` (= window_frames / fps)
// of the clip, not one frame — `fps` alone would understate the spacing. In
// "legacy_full_clip" mode (or when `av_inference_mode` is absent, for
// backward compatibility with older responses) each score is one video frame,
// so `fps` gives per-frame time directly; `window_seconds` divided by the score
// count is a last-resort fallback when fps is missing entirely.
export function perWindowSeconds(scores: number[], metadata: AVTimingMetadata | null | undefined): number {
  if (metadata?.av_inference_mode === "windowed" && metadata.window_seconds && metadata.window_seconds > 0) {
    return metadata.window_seconds;
  }
  if (metadata?.fps && metadata.fps > 0) return 1 / metadata.fps;
  if (metadata?.window_seconds && scores.length > 0) return metadata.window_seconds / scores.length;
  return 1;
}

export function buildTimelineData(scores: number[], metadata: AVTimingMetadata | null | undefined): TimelinePoint[] {
  // Windowed responses carry each window's real start time (the final window can
  // be shorter than the rest), which is more precise than a uniform step.
  const windows = metadata?.windows;
  if (windows && windows.length === scores.length) {
    return scores.map((score, i) => ({ time: Number(windows[i].start_time.toFixed(3)), score, index: i }));
  }
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
