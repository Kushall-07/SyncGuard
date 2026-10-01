// Thin client for the SyncGuard FastAPI backend (backend/main.py).
// No scoring/labelling logic lives here — only the shapes the predictor's
// dataclasses actually return (see src/inference/predictor.py).

// In local development this is left unset, so requests stay relative
// ("/api/...") and are handled by the Vite dev server proxy configured in
// vite.config.ts (which forwards them to http://127.0.0.1:8000). In a
// production build where the frontend and backend are deployed separately
// (e.g. Vercel + a Docker-hosted API), set VITE_API_BASE_URL at build time to
// the deployed backend's origin, e.g. https://syncguard-api.example.com.
const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function apiUrl(path: string): string {
  return `${API_BASE_URL}${path}`;
}

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

export interface AVWindowMetadata {
  index: number;
  start_time: number;
  end_time: number;
  n_valid_frames: number;
  valid_fraction: number;
  sync_score: number;
  desync_score: number;
}

export interface AVTimingMetadata {
  fps?: number;
  num_frames?: number;
  window_seconds?: number;
  audio_token_seconds?: number;
  num_valid_landmark_frames?: number;
  num_windows?: number;
  // Present once the backend runs windowed inference (see
  // src/inference/predictor.py); absent (or "legacy_full_clip") for the
  // pre-windowing full-clip path, kept as a fallback/reference mode.
  av_inference_mode?: "windowed" | "legacy_full_clip";
  window_frames?: number;
  stride_frames?: number;
  total_duration_seconds?: number;
  windows?: AVWindowMetadata[];
  aggregate_method?: string;
  median_window_score?: number;
  valid_weighted_window_score?: number;
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
  const res = await fetch(apiUrl("/api/health"));
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export async function analyzeAudio(file: File): Promise<AudioOnlyResult> {
  const form = new FormData();
  form.append("audio", file);
  const res = await fetch(apiUrl("/api/analyze/audio"), { method: "POST", body: form });
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
  const res = await fetch(apiUrl("/api/analyze/av"), { method: "POST", body: form });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}

export async function fetchLabSamples(): Promise<LabSamplesResponse> {
  const res = await fetch(apiUrl("/api/lab/samples"));
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  const data: LabSamplesResponse = await res.json();
  // video_url comes back as a path relative to the API origin (e.g.
  // "/api/lab/samples/004073/video"); resolve it against the same configured
  // API base so it still loads when the frontend and backend are on different
  // origins in production.
  return { ...data, samples: data.samples.map((s) => ({ ...s, video_url: apiUrl(s.video_url) })) };
}

export async function runLabAnalysis(sampleId: string, shiftSeconds: number): Promise<SyncLabResult> {
  const res = await fetch(apiUrl("/api/lab/analyze"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sample_id: sampleId, shift_seconds: shiftSeconds }),
  });
  if (!res.ok) throw new Error(await parseErrorDetail(res));
  return res.json();
}
