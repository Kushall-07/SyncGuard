import { describe, expect, it } from "vitest";
import { buildTimelineData, findLowestPoint, formatTimestamp, perWindowSeconds } from "./timeline";

describe("perWindowSeconds", () => {
  it("uses 1/fps when fps is present", () => {
    expect(perWindowSeconds([0.1, 0.2, 0.3], { fps: 25 })).toBeCloseTo(0.04);
  });

  it("falls back to window_seconds / count when fps is missing", () => {
    // This is the exact bug the master spec calls out: window_seconds is the total
    // clip duration (T_v / fps), not a per-window duration, so it must be divided
    // by the window count rather than used directly.
    expect(perWindowSeconds([0, 0, 0, 0], { window_seconds: 5.08 })).toBeCloseTo(1.27);
  });

  it("falls back to 1 second when neither fps nor window_seconds is available", () => {
    expect(perWindowSeconds([0, 0], {})).toBe(1);
    expect(perWindowSeconds([0, 0], null)).toBe(1);
  });

  it("does not divide by zero for an empty score list", () => {
    expect(perWindowSeconds([], { window_seconds: 5 })).toBe(1);
  });
});

describe("buildTimelineData", () => {
  it("derives timestamps from fps, not from total clip duration", () => {
    const data = buildTimelineData([0.9, 0.8, 0.7], { fps: 25, num_frames: 3 });
    expect(data).toEqual([
      { time: 0, score: 0.9, index: 0 },
      { time: 0.04, score: 0.8, index: 1 },
      { time: 0.08, score: 0.7, index: 2 },
    ]);
  });

  it("handles a single window without error", () => {
    const data = buildTimelineData([0.5], { fps: 25 });
    expect(data).toEqual([{ time: 0, score: 0.5, index: 0 }]);
  });
});

describe("findLowestPoint", () => {
  it("returns the minimum-scoring point", () => {
    const data = buildTimelineData([0.9, 0.2, 0.7], { fps: 10 });
    expect(findLowestPoint(data)?.index).toBe(1);
  });

  it("returns null for an empty timeline", () => {
    expect(findLowestPoint([])).toBeNull();
  });
});

describe("formatTimestamp", () => {
  it("formats sub-minute timestamps as mm:ss.cc", () => {
    expect(formatTimestamp(1.24)).toBe("00:01.24");
  });

  it("formats timestamps past one minute", () => {
    expect(formatTimestamp(65.5)).toBe("01:05.50");
  });

  it("formats zero", () => {
    expect(formatTimestamp(0)).toBe("00:00.00");
  });
});
