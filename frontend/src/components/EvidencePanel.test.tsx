import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import EvidencePanel from "./EvidencePanel";
import { buildAvEvidence } from "../lib/evidence";
import type { AudioVisualResult } from "../services/api";

const result: AudioVisualResult = {
  mode: "audio_visual",
  predicted_label: "sync",
  sync_probability: 0.72,
  desync_probability: 0.28,
  aggregate_sync_score: 0.72,
  per_window_sync_scores: [0.9, 0.8, 0.3],
  timing_metadata: { fps: 25, num_windows: 3 },
  audio_encoder_checkpoint: "a.pt",
  visual_encoder_checkpoint: "v.pt",
  sync_model_checkpoint: "s.pt",
};

describe("EvidencePanel", () => {
  it("renders all three evidence categories", () => {
    render(<EvidencePanel evidence={buildAvEvidence(result)} />);
    expect(screen.getByText("Observed")).toBeInTheDocument();
    expect(screen.getByText("Derived")).toBeInTheDocument();
    expect(screen.getByText("Not Determined")).toBeInTheDocument();
  });

  it("renders observed values from the actual result, not placeholders", () => {
    render(<EvidencePanel evidence={buildAvEvidence(result)} />);
    expect(screen.getByText("0.2800")).toBeInTheDocument(); // desync score, unique to this result
    expect(screen.getAllByText("0.7200").length).toBeGreaterThan(0); // sync + aggregate scores
  });
});
