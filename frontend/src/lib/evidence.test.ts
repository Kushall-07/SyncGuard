import { describe, expect, it } from "vitest";
import { buildAudioEvidence, buildAvEvidence } from "./evidence";
import type { AudioOnlyResult, AudioVisualResult } from "../services/api";

function audioResult(overrides: Partial<AudioOnlyResult> = {}): AudioOnlyResult {
  return {
    mode: "audio_only",
    predicted_label: "bonafide",
    spoof_probability: 0.02,
    bonafide_probability: 0.98,
    confidence: 0.98,
    audio_encoder_checkpoint: "None",
    spoof_head_checkpoint: "ckpt.pt",
    cnn_checkpoint: "cnn.pt",
    use_ensemble: true,
    ...overrides,
  };
}

function avResult(overrides: Partial<AudioVisualResult> = {}): AudioVisualResult {
  return {
    mode: "audio_visual",
    predicted_label: "sync",
    sync_probability: 0.72,
    desync_probability: 0.28,
    aggregate_sync_score: 0.72,
    per_window_sync_scores: [0.9, 0.8, 0.3, 0.6],
    timing_metadata: { fps: 25, num_windows: 4 },
    audio_encoder_checkpoint: "a.pt",
    visual_encoder_checkpoint: "v.pt",
    sync_model_checkpoint: "s.pt",
    ...overrides,
  };
}

describe("buildAudioEvidence", () => {
  it("observes the raw scores without relabeling them as confidence", () => {
    const evidence = buildAudioEvidence(audioResult());
    const labels = evidence.observed.map((i) => i.label);
    expect(labels).toContain("Bonafide score");
    expect(labels).toContain("Spoof score");
    expect(labels.join(" ")).not.toMatch(/confidence/i);
  });

  it("derives the winning class from whichever score is larger", () => {
    const evidence = buildAudioEvidence(audioResult({ bonafide_probability: 0.3, spoof_probability: 0.7 }));
    expect(evidence.derived[0]).toMatch(/"spoof"/);
  });

  it("never claims to know the manipulation technique", () => {
    const evidence = buildAudioEvidence(audioResult());
    expect(evidence.notDetermined.join(" ")).toMatch(/manipulation technique/i);
  });
});

describe("buildAvEvidence", () => {
  it("counts windows above and below the synchronization threshold", () => {
    const evidence = buildAvEvidence(avResult());
    // scores: [0.9, 0.8, 0.3, 0.6] -> 3 at/above 0.5, 1 below
    expect(evidence.derived[0]).toMatch(/3 of 4/);
  });

  it("reports the observed windows-analyzed count from timing metadata", () => {
    const evidence = buildAvEvidence(avResult());
    const windows = evidence.observed.find((i) => i.label === "Windows analyzed");
    expect(windows?.value).toBe("4");
  });

  it("never claims to determine whether media was manipulated", () => {
    const evidence = buildAvEvidence(avResult());
    expect(evidence.notDetermined.some((line) => /manipulated/i.test(line))).toBe(true);
  });

  it("handles an empty per-window score list gracefully", () => {
    const evidence = buildAvEvidence(avResult({ per_window_sync_scores: [] }));
    expect(evidence.derived.length).toBeGreaterThan(0);
  });
});
