// Deterministic "why this result?" evidence, derived only from actual model outputs.
// No LLM, no causal explanations — every line here is either a raw field from the
// backend response or a simple statistic computed from it.

import type { AudioOnlyResult, AudioVisualResult } from "../services/api";

export interface EvidenceItem {
  label: string;
  value: string;
}

export interface Evidence {
  observed: EvidenceItem[];
  derived: string[];
  notDetermined: string[];
}

export function buildAudioEvidence(result: AudioOnlyResult): Evidence {
  const larger = result.bonafide_probability >= result.spoof_probability ? "bonafide" : "spoof";
  return {
    observed: [
      { label: "Bonafide score", value: result.bonafide_probability.toFixed(4) },
      { label: "Spoof score", value: result.spoof_probability.toFixed(4) },
      { label: "Model", value: result.use_ensemble ? "CNN + Transformer ensemble" : "Transformer" },
    ],
    derived: [
      `The model assigned the larger score to the "${larger}" class.`,
      result.use_ensemble
        ? "This score is the mean of two independently trained audio models (CNN, Transformer)."
        : "This score comes from a single Transformer audio model.",
    ],
    notDetermined: [
      "The exact manipulation technique or provenance of the recording.",
      "Whether this score would generalize to audio recorded under different conditions than ASVspoof 2019 LA.",
    ],
  };
}

export function buildAvEvidence(result: AudioVisualResult): Evidence {
  const scores = result.per_window_sync_scores ?? [];
  const threshold = 0.5;
  const aboveCount = scores.filter((s) => s >= threshold).length;
  const belowCount = scores.length - aboveCount;
  const majorityStatus = scores.length > 0 && aboveCount >= belowCount ? "above" : "below";

  const derived: string[] = [];
  if (scores.length > 0) {
    derived.push(
      `${aboveCount} of ${scores.length} analyzed windows (${((aboveCount / scores.length) * 100).toFixed(0)}%) remained at or above the ${threshold.toFixed(1)} synchronization threshold; the majority sit ${majorityStatus} it.`,
    );
    const min = Math.min(...scores);
    const max = Math.max(...scores);
    derived.push(`Per-window sync scores ranged from ${min.toFixed(3)} to ${max.toFixed(3)} across the clip.`);
  }
  derived.push(
    `The aggregate score is the mean of per-window scores over valid (landmark-detected) frames, producing the "${result.predicted_label}" label.`,
  );

  return {
    observed: [
      { label: "Sync score", value: result.sync_probability.toFixed(4) },
      { label: "Desync score", value: result.desync_probability.toFixed(4) },
      { label: "Aggregate sync score", value: result.aggregate_sync_score.toFixed(4) },
      { label: "Windows analyzed", value: String(result.timing_metadata?.num_windows ?? scores.length) },
    ],
    derived,
    notDetermined: [
      "Whether the underlying media was intentionally manipulated.",
      "The location or nature of any manipulation, if present.",
      "Whether a synchronized clip is authentic — a manipulated video can remain synchronized.",
    ],
  };
}
