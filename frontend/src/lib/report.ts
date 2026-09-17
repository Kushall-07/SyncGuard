// Structured JSON report generation and lightweight client-side analysis history.
// Everything here is derived from actual analysis results — nothing is fabricated.

import type { AudioOnlyResult, AudioVisualResult } from "../services/api";

export interface AudioReportInput {
  mode: "audio_only";
  result: AudioOnlyResult;
  filename: string;
}

export interface AvReportInput {
  mode: "audio_visual";
  result: AudioVisualResult;
  filename: string;
}

export type ReportInput = AudioReportInput | AvReportInput;

export function buildReport(input: ReportInput): Record<string, unknown> {
  const generatedAt = new Date().toISOString();

  if (input.mode === "audio_only") {
    const { result, filename } = input;
    return {
      report_type: "syncguard_audio_analysis_report",
      generated_at: generatedAt,
      input: { filename },
      mode: "audio_only",
      result: {
        predicted_label: result.predicted_label,
        bonafide_score: result.bonafide_probability,
        spoof_score: result.spoof_probability,
        model_score: result.confidence,
      },
      model: {
        pipeline: result.use_ensemble ? "CNN + Transformer ensemble" : "Transformer",
        spoof_head_checkpoint: result.spoof_head_checkpoint,
        cnn_checkpoint: result.cnn_checkpoint || null,
      },
      disclaimer:
        "Scores are raw model outputs, not calibrated probabilities. This is a research system, not a forensic tool.",
    };
  }

  const { result, filename } = input;
  const scores = result.per_window_sync_scores ?? [];
  return {
    report_type: "syncguard_av_analysis_report",
    generated_at: generatedAt,
    input: {
      filename,
      duration_seconds:
        result.timing_metadata?.num_frames && result.timing_metadata?.fps
          ? result.timing_metadata.num_frames / result.timing_metadata.fps
          : null,
      fps: result.timing_metadata?.fps ?? null,
      frames: result.timing_metadata?.num_frames ?? null,
      windows_analyzed: result.timing_metadata?.num_windows ?? scores.length,
    },
    mode: "audio_visual",
    result: {
      predicted_label: result.predicted_label,
      sync_score: result.sync_probability,
      desync_score: result.desync_probability,
      aggregate_sync_score: result.aggregate_sync_score,
    },
    temporal_analysis:
      scores.length > 0
        ? {
            mean_sync_score: scores.reduce((a, b) => a + b, 0) / scores.length,
            min_sync_score: Math.min(...scores),
            max_sync_score: Math.max(...scores),
            lowest_scoring_window_index: scores.indexOf(Math.min(...scores)),
            highest_scoring_window_index: scores.indexOf(Math.max(...scores)),
          }
        : null,
    model: {
      audio: "CNN + Transformer",
      visual: "MediaPipe Face/Mouth Landmarks + Visual Landmark Transformer",
      fusion: "Bidirectional Cross-Attention",
      synchronization: "Sync Head",
      audio_encoder_checkpoint: result.audio_encoder_checkpoint,
      visual_encoder_checkpoint: result.visual_encoder_checkpoint,
      sync_model_checkpoint: result.sync_model_checkpoint,
    },
    disclaimer:
      "AV mode detects temporal synchronization inconsistencies between audio and visual streams. It does not directly detect content manipulation. A synchronized manipulated video can still be a deepfake. Raw model scores are not calibrated probabilities.",
  };
}

export function downloadJson(filename: string, data: unknown) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// --- Lightweight client-side analysis history (localStorage only, never uploaded) ---

export interface HistoryEntry {
  id: string;
  filename: string;
  mode: "audio_only" | "audio_visual";
  resultLabel: string;
  score: number;
  timestamp: string;
}

const HISTORY_KEY = "syncguard.history.v1";
const HISTORY_LIMIT = 12;

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function pushHistoryEntry(entry: Omit<HistoryEntry, "id" | "timestamp">): HistoryEntry {
  const created: HistoryEntry = { ...entry, id: crypto.randomUUID(), timestamp: new Date().toISOString() };
  try {
    const existing = loadHistory();
    const next = [created, ...existing].slice(0, HISTORY_LIMIT);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
  } catch {
    // localStorage unavailable (private browsing, quota) — history just won't persist.
  }
  return created;
}

export function clearHistory() {
  try {
    localStorage.removeItem(HISTORY_KEY);
  } catch {
    // ignore
  }
}

// In-memory only (never persisted): lets a history entry reopen its full result
// object within the same browser session, without uploading or storing media.
const sessionResultCache = new Map<string, AudioOnlyResult | AudioVisualResult>();

export function cacheSessionResult(id: string, result: AudioOnlyResult | AudioVisualResult) {
  sessionResultCache.set(id, result);
}

export function getSessionResult(id: string): AudioOnlyResult | AudioVisualResult | undefined {
  return sessionResultCache.get(id);
}
