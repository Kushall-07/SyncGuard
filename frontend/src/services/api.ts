// Thin client for the SyncGuard FastAPI backend (backend/main.py).
// No scoring/labelling logic lives here — only the shapes the predictor's
// dataclasses actually return (see src/inference/predictor.py).

export interface AudioOnlyResult {
  mode: "audio_only";
  predicted_label: "bonafide" | "spoof";
  spoof_probability: number;
  bonafide_probability: number;
  confidence: number;
  audio_encoder_checkpoint: string;
  spoof_head_checkpoint: string;
  cnn_checkpoint: string;
  use_ensemble: boolean;
}

export interface AVTimingMetadata {
  fps?: number;
  num_frames?: number;
  window_seconds?: number;
  audio_token_seconds?: number;
  num_valid_landmark_frames?: number;
  num_windows?: number;
}

export interface AudioVisualResult {
  mode: "audio_visual";
  predicted_label: "sync" | "desync";
  sync_probability: number;
  desync_probability: number;
  aggregate_sync_score: number;
  per_window_sync_scores: number[] | null;
  timing_metadata: AVTimingMetadata | null;
  audio_encoder_checkpoint: string;
  visual_encoder_checkpoint: string;
  sync_model_checkpoint: string;
}

export interface HealthStatus {
  status: "ready" | "unavailable";
  gpu_ready: boolean;
  device: string;
}

export type AnalysisResult = AudioOnlyResult | AudioVisualResult;

export interface SyncLabTimingMetadata extends AVTimingMetadata {
  shift_seconds?: number;
}

export interface SyncLabResult {
  mode: "sync_lab";
  sample_id: string;
  shift_seconds: number;
  aggregate_sync_score: number;
  per_window_sync_scores: number[] | null;
  timing_metadata: SyncLabTimingMetadata | null;
}

export interface LabSample {
  id: string;
  label: string;
  video_url: string;
}

export interface LabSamplesResponse {
  samples: LabSample[];
  available_shifts: number[];
}

async function parseErrorDetail(res: Response): Promise<string> {
  try {
    const body = await res.json();
    return body.detail ?? res.statusText;
  } catch {
    return res.statusText;
  }
}

export async function checkHealth(): Promise<HealthStatus> {
  const res = await fetch("/api/health");
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export async function analyzeAudio(file: File): Promise<AudioOnlyResult> {
  const form = new FormData();
  form.append("audio", file);
  const res = await fetch("/api/analyze/audio", { method: "POST", body: form });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export async function analyzeAV(
  video: File,
  audio?: File | null,
  landmarks?: File | null,
): Promise<AudioVisualResult> {
  const form = new FormData();
  form.append("video", video);
  if (audio) form.append("audio", audio);
  if (landmarks) form.append("landmarks", landmarks);
  const res = await fetch("/api/analyze/av", { method: "POST", body: form });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export async function fetchLabSamples(): Promise<LabSamplesResponse> {
  const res = await fetch("/api/lab/samples");
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export async function runLabAnalysis(sampleId: string, shiftSeconds: number): Promise<SyncLabResult> {
  const res = await fetch("/api/lab/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_id: sampleId, shift_seconds: shiftSeconds }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}
