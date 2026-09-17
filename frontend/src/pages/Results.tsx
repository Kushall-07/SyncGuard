import { useMemo, useRef } from "react";
import type { RefObject } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import ScoreCard from "../components/ScoreCard";
import SyncTimeline from "../components/SyncTimeline";
import EvidencePanel from "../components/EvidencePanel";
import Button from "../components/Button";
import { buildAudioEvidence, buildAvEvidence } from "../lib/evidence";
import { buildReport, downloadJson } from "../lib/report";
import type { AudioOnlyResult, AudioVisualResult } from "../services/api";

type LocationState = {
  result: AudioOnlyResult | AudioVisualResult;
  mode: "audio_only" | "audio_visual";
  filename: string;
  videoUrl?: string | null;
} | null;

function basename(path: string): string {
  if (!path || path === "None") return "Not used in this mode";
  return path.split(/[\\/]/).pop() ?? path;
}

export default function Results() {
  const location = useLocation();
  const navigate = useNavigate();
  const state = location.state as LocationState;

  const generatedAt = useMemo(() => new Date(), []);
  const analysisId = useMemo(() => crypto.randomUUID().slice(0, 8).toUpperCase(), []);
  const videoRef = useRef<HTMLVideoElement>(null);

  if (!state?.result) {
    return (
      <div className="container-page py-24 text-center">
        <p className="text-xs uppercase tracking-widest opacity-60">Analysis Result</p>
        <h1 className="font-display text-3xl mt-3">No report loaded</h1>
        <p className="mt-4 opacity-70">Run an analysis first to generate a shareable report.</p>
        <Link to="/analyze" className="mt-8 inline-block underline text-sage-light">
          Go to Analyze →
        </Link>
      </div>
    );
  }

  const { result, mode, filename, videoUrl } = state;
  const isAv = mode === "audio_visual";

  const handleDownload = () => {
    const report = isAv
      ? buildReport({ mode: "audio_visual", result: result as AudioVisualResult, filename })
      : buildReport({ mode: "audio_only", result: result as AudioOnlyResult, filename });
    downloadJson(`syncguard-report-${analysisId}.json`, report);
  };

  const handleSeek = (t: number) => {
    if (videoRef.current) videoRef.current.currentTime = t;
  };

  return (
    <div className="container-page py-16">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-widest opacity-60">Analysis Result</p>
          <h1 className="font-display text-4xl mt-2">{isAv ? "Synchronization Report" : "Authenticity Report"}</h1>
        </div>
        <div className="text-right text-xs opacity-60">
          <p>Analysis ID: {analysisId}</p>
          <p>{generatedAt.toLocaleString()}</p>
        </div>
      </div>

      {isAv ? (
        <AvReport result={result as AudioVisualResult} videoUrl={videoUrl} videoRef={videoRef} onSeek={handleSeek} />
      ) : (
        <AudioReport result={result as AudioOnlyResult} />
      )}

      <ModelPipeline mode={mode} />
      <Provenance result={result} mode={mode} filename={filename} generatedAt={generatedAt} />

      <div className="mt-16 border-t pt-10 flex flex-wrap gap-4" style={{ borderColor: "var(--page-border)" }}>
        <Button variant="primary" onClick={handleDownload}>
          Download JSON Report ↓
        </Button>
        <Button variant="ghost" onClick={() => navigate("/analyze")}>
          Run New Analysis
        </Button>
        <Button variant="ghost" onClick={() => navigate("/analyze")}>
          Back to Analyze
        </Button>
      </div>
    </div>
  );
}

function AvReport({
  result,
  videoUrl,
  videoRef,
  onSeek,
}: {
  result: AudioVisualResult;
  videoUrl?: string | null;
  videoRef: RefObject<HTMLVideoElement | null>;
  onSeek: (t: number) => void;
}) {
  const isSync = result.predicted_label === "sync";
  const meta = result.timing_metadata;
  const duration = meta?.num_frames && meta?.fps ? meta.num_frames / meta.fps : null;
  const evidence = buildAvEvidence(result);

  return (
    <>
      <div className="mt-12 grid lg:grid-cols-[1fr_1.3fr] gap-12">
        <div>
          {videoUrl && (
            <video ref={videoRef} src={videoUrl} controls className="w-full border mb-6" style={{ borderColor: "var(--page-border)" }} />
          )}
          <div
            className={`inline-block px-3 py-1 text-xs uppercase tracking-widest border ${isSync ? "border-sage text-sage-light" : "border-copper text-copper-light"}`}
          >
            {isSync ? "Synchronized" : "Desynchronized"}
          </div>
          <p className="mt-6 font-display text-6xl">{(result.aggregate_sync_score * 100).toFixed(2)}%</p>
          <p className="text-sm opacity-60">Aggregate Sync Score</p>

          <div className="mt-8 grid grid-cols-2 gap-3">
            <ScoreCard label="Sync Score" value={`${(result.sync_probability * 100).toFixed(2)}%`} />
            <ScoreCard label="Desync Score" value={`${(result.desync_probability * 100).toFixed(2)}%`} />
          </div>
        </div>

        <div id="timeline">
          {result.per_window_sync_scores && (
            <>
              <p className="text-xs uppercase tracking-widest opacity-60 mb-3">Synchronization Timeline</p>
              <SyncTimeline scores={result.per_window_sync_scores} metadata={meta} onSeek={videoUrl ? onSeek : undefined} />
            </>
          )}
        </div>
      </div>

      <div className="mt-16 border-t pt-10 grid sm:grid-cols-2 md:grid-cols-4 gap-8 text-sm" style={{ borderColor: "var(--page-border)" }}>
        <Field label="Input Duration" value={duration ? `${duration.toFixed(1)}s` : "—"} />
        <Field label="Windows Analyzed" value={meta?.num_windows?.toString() ?? "—"} />
        <Field label="Frame Rate" value={meta?.fps ? `${meta.fps.toFixed(1)} fps` : "—"} />
        <Field label="Mode" value="Audio-Visual" />
      </div>

      <p className="mt-12 text-xs leading-relaxed opacity-60 max-w-2xl border-l-2 pl-4" style={{ borderColor: "var(--sage)" }}>
        AV mode detects temporal synchronization inconsistencies between audio and visual streams. It does not
        directly detect content manipulation. A synchronized manipulated video can still be a deepfake. Raw
        model scores are not calibrated probabilities.
      </p>

      <div className="mt-16">
        <EvidencePanel evidence={evidence} />
      </div>
    </>
  );
}

function AudioReport({ result }: { result: AudioOnlyResult }) {
  const isBonafide = result.predicted_label === "bonafide";
  const evidence = buildAudioEvidence(result);

  return (
    <>
      <div className="mt-12">
        <div
          className={`inline-block px-3 py-1 text-xs uppercase tracking-widest border ${isBonafide ? "border-sage text-sage-light" : "border-copper text-copper-light"}`}
        >
          {isBonafide ? "Bonafide / Real" : "Spoof / Synthetic"}
        </div>

        <div className="mt-8 grid grid-cols-2 sm:grid-cols-3 gap-3 max-w-xl">
          <ScoreCard label="Bonafide Score" value={result.bonafide_probability.toFixed(4)} emphasis />
          <ScoreCard label="Spoof Score" value={result.spoof_probability.toFixed(4)} />
          <ScoreCard label="Model Score" value={result.confidence.toFixed(4)} />
        </div>
      </div>

      <div className="mt-12 border-t pt-10 grid sm:grid-cols-2 md:grid-cols-4 gap-8 text-sm" style={{ borderColor: "var(--page-border)" }}>
        <Field label="Mode" value="Audio Only" />
        <Field label="Model" value={result.use_ensemble ? "CNN + Transformer Ensemble" : "Transformer"} />
      </div>

      <p className="mt-12 text-xs leading-relaxed opacity-60 max-w-2xl border-l-2 pl-4" style={{ borderColor: "var(--sage)" }}>
        Scores are raw model outputs, not calibrated probabilities. Audio-only mode evaluates whether the speech
        signal is likely bonafide or spoofed, based on acoustic features learned from ASVspoof 2019 LA.
      </p>

      <div className="mt-16">
        <EvidencePanel evidence={evidence} />
      </div>
    </>
  );
}

function ModelPipeline({ mode }: { mode: "audio_only" | "audio_visual" }) {
  return (
    <div className="mt-16 border-t pt-10" style={{ borderColor: "var(--page-border)" }}>
      <p className="text-xs uppercase tracking-widest opacity-60 mb-5">Model Pipeline</p>
      <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-6 text-sm">
        <PipelineStep label="Audio" value="CNN + Transformer" />
        {mode === "audio_visual" && (
          <>
            <PipelineStep label="Visual" value="MediaPipe Landmarks + Landmark Transformer" />
            <PipelineStep label="Fusion" value="Bidirectional Cross-Attention" />
            <PipelineStep label="Synchronization" value="Sync Head" />
          </>
        )}
        {mode === "audio_only" && <PipelineStep label="Classification" value="Spoof Head" />}
      </div>
    </div>
  );
}

function PipelineStep({ label, value }: { label: string; value: string }) {
  return (
    <div className="border-l-2 pl-4" style={{ borderColor: "var(--page-border)" }}>
      <p className="text-xs uppercase tracking-widest opacity-50">{label}</p>
      <p className="mt-1 text-sm">{value}</p>
    </div>
  );
}

function Provenance({
  result,
  mode,
  filename,
  generatedAt,
}: {
  result: AudioOnlyResult | AudioVisualResult;
  mode: "audio_only" | "audio_visual";
  filename: string;
  generatedAt: Date;
}) {
  const isAv = mode === "audio_visual";
  const av = isAv ? (result as AudioVisualResult) : null;
  const meta = av?.timing_metadata;

  return (
    <div className="mt-16 border-t pt-10" style={{ borderColor: "var(--page-border)" }}>
      <p className="text-xs uppercase tracking-widest opacity-60 mb-5">Provenance</p>
      <div className="grid sm:grid-cols-2 md:grid-cols-4 gap-6 text-sm">
        <Field label="Analysis Timestamp" value={generatedAt.toISOString()} />
        <Field label="Input Filename" value={filename} />
        <Field label="Mode" value={isAv ? "Audio-Visual" : "Audio Only"} />
        {isAv && <Field label="FPS" value={meta?.fps ? meta.fps.toFixed(1) : "—"} />}
        {isAv && <Field label="Window Count" value={meta?.num_windows?.toString() ?? "—"} />}
        {isAv && av && (
          <Field label="Audio Encoder" value={basename(av.audio_encoder_checkpoint)} title={av.audio_encoder_checkpoint} />
        )}
        {isAv && av && (
          <Field label="Visual Encoder" value={basename(av.visual_encoder_checkpoint)} title={av.visual_encoder_checkpoint} />
        )}
        {isAv && av && (
          <Field label="Sync Model" value={basename(av.sync_model_checkpoint)} title={av.sync_model_checkpoint} />
        )}
        {!isAv && (
          <Field
            label="Spoof Head Checkpoint"
            value={basename((result as AudioOnlyResult).spoof_head_checkpoint)}
            title={(result as AudioOnlyResult).spoof_head_checkpoint}
          />
        )}
        {!isAv && (result as AudioOnlyResult).use_ensemble && (
          <Field
            label="CNN Checkpoint"
            value={basename((result as AudioOnlyResult).cnn_checkpoint)}
            title={(result as AudioOnlyResult).cnn_checkpoint}
          />
        )}
      </div>
    </div>
  );
}

function Field({ label, value, title }: { label: string; value: string; title?: string }) {
  return (
    <div title={title}>
      <p className="text-xs uppercase tracking-widest opacity-50">{label}</p>
      <p className="mt-1 font-display text-lg truncate">{value}</p>
    </div>
  );
}
