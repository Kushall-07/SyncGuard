import { beforeEach, describe, expect, it } from "vitest";
import {
  buildReport,
  cacheSessionResult,
  clearHistory,
  getSessionResult,
  loadHistory,
  pushHistoryEntry,
} from "./report";
import type { AudioOnlyResult, AudioVisualResult } from "../services/api";

const audioResult: AudioOnlyResult = {
  mode: "audio_only",
  predicted_label: "spoof",
  spoof_probability: 0.98,
  bonafide_probability: 0.02,
  confidence: 0.98,
  audio_encoder_checkpoint: "None",
  spoof_head_checkpoint: "ckpt.pt",
  cnn_checkpoint: "cnn.pt",
  use_ensemble: true,
};

const avResult: AudioVisualResult = {
  mode: "audio_visual",
  predicted_label: "sync",
  sync_probability: 0.72,
  desync_probability: 0.28,
  aggregate_sync_score: 0.72,
  per_window_sync_scores: [0.9, 0.6, 0.3],
  timing_metadata: { fps: 25, num_frames: 127, num_windows: 127 },
  audio_encoder_checkpoint: "a.pt",
  visual_encoder_checkpoint: "v.pt",
  sync_model_checkpoint: "s.pt",
};

describe("buildReport", () => {
  it("builds an audio report with real scores, not fabricated ones", () => {
    const report = buildReport({ mode: "audio_only", result: audioResult, filename: "clip.flac" });
    expect(report.mode).toBe("audio_only");
    expect((report.result as Record<string, unknown>).spoof_score).toBe(0.98);
    expect((report.input as Record<string, unknown>).filename).toBe("clip.flac");
  });

  it("builds an AV report with derived temporal statistics", () => {
    const report = buildReport({ mode: "audio_visual", result: avResult, filename: "clip.mp4" });
    const temporal = report.temporal_analysis as Record<string, number>;
    expect(temporal.min_sync_score).toBeCloseTo(0.3);
    expect(temporal.max_sync_score).toBeCloseTo(0.9);
    expect(temporal.mean_sync_score).toBeCloseTo(0.6);
  });

  it("includes the model pipeline and a scientific disclaimer", () => {
    const report = buildReport({ mode: "audio_visual", result: avResult, filename: "clip.mp4" });
    expect(report.disclaimer).toMatch(/does not directly detect content manipulation/i);
    expect((report.model as Record<string, unknown>).fusion).toBe("Bidirectional Cross-Attention");
  });
});

describe("history", () => {
  beforeEach(() => {
    clearHistory();
  });

  it("stores entries most-recent-first", () => {
    pushHistoryEntry({ filename: "a.wav", mode: "audio_only", resultLabel: "Bonafide", score: 0.9 });
    pushHistoryEntry({ filename: "b.wav", mode: "audio_only", resultLabel: "Spoof", score: 0.1 });
    const history = loadHistory();
    expect(history).toHaveLength(2);
    expect(history[0].filename).toBe("b.wav");
  });

  it("caches the full result in-memory for the session and returns it by id", () => {
    const entry = pushHistoryEntry({ filename: "clip.mp4", mode: "audio_visual", resultLabel: "Synchronized", score: 0.72 });
    cacheSessionResult(entry.id, avResult);
    expect(getSessionResult(entry.id)).toBe(avResult);
  });

  it("returns undefined for an id that was never cached", () => {
    expect(getSessionResult("unknown-id")).toBeUndefined();
  });
});
