import { useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Button from "../components/Button";
import UploadDropzone from "../components/UploadDropzone";
import ScoreCard from "../components/ScoreCard";
import RadialGauge from "../components/RadialGauge";
import MediaFrame from "../components/MediaFrame";
import SyncTimeline from "../components/SyncTimeline";
import AnalyzingState from "../components/AnalyzingState";
import RecentAnalyses from "../components/RecentAnalyses";
import Reveal from "../components/scroll/Reveal";
import { analyzeAudio, analyzeAV } from "../services/api";
import type { AudioOnlyResult, AudioVisualResult } from "../services/api";
import { cacheSessionResult, pushHistoryEntry } from "../lib/report";

type Mode = "av" | "audio";

const AV_STAGES = [
  "Preparing media",
  "Extracting landmarks",
  "Processing audio",
  "Aligning temporal streams",
  "Running multimodal analysis",
  "Generating synchronization timeline",
];

const AUDIO_STAGES = ["Preparing audio", "Extracting representation", "Running ensemble", "Generating result"];

export default function Analyze() {
  const navigate = useNavigate();
  const [mode, setMode] = useState<Mode>("av");

  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [avAudioFile, setAvAudioFile] = useState<File | null>(null);
  const [landmarksFile, setLandmarksFile] = useState<File | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const [audioFile, setAudioFile] = useState<File | null>(null);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [avResult, setAvResult] = useState<AudioVisualResult | null>(null);
  const [audioResult, setAudioResult] = useState<AudioOnlyResult | null>(null);
  const [selectedTime, setSelectedTime] = useState<number | null>(null);
  const [historyTick, setHistoryTick] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const videoUrl = useMemo(() => (videoFile ? URL.createObjectURL(videoFile) : null), [videoFile]);

  const runAv = async () => {
    if (!videoFile) return;
    setLoading(true);
    setError(null);
    setAvResult(null);
    setSelectedTime(null);
    try {
      const result = await analyzeAV(videoFile, avAudioFile, landmarksFile);
      setAvResult(result);
      const entry = pushHistoryEntry({
        filename: videoFile.name,
        mode: "audio_visual",
        resultLabel: result.predicted_label === "sync" ? "Synchronized" : "Desynchronized",
        score: result.aggregate_sync_score,
      });
      cacheSessionResult(entry.id, result);
      setHistoryTick((t) => t + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to analyze this file.");
    } finally {
      setLoading(false);
    }
  };

  const runAudio = async () => {
    if (!audioFile) return;
    setLoading(true);
    setError(null);
    setAudioResult(null);
    try {
      const result = await analyzeAudio(audioFile);
      setAudioResult(result);
      const entry = pushHistoryEntry({
        filename: audioFile.name,
        mode: "audio_only",
        resultLabel: result.predicted_label === "bonafide" ? "Bonafide" : "Spoof",
        score: result.confidence,
      });
      cacheSessionResult(entry.id, result);
      setHistoryTick((t) => t + 1);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unable to analyze this file.");
    } finally {
      setLoading(false);
    }
  };

  const switchMode = (next: Mode) => {
    setMode(next);
    setError(null);
  };

  const openAvReport = () => {
    if (!avResult) return;
    navigate("/results", {
      state: { result: avResult, mode: "audio_visual", filename: videoFile?.name ?? "video", videoUrl },
    });
  };

  const openAudioReport = () => {
    if (!audioResult) return;
    navigate("/results", {
      state: { result: audioResult, mode: "audio_only", filename: audioFile?.name ?? "audio" },
    });
  };

  const handleSeek = (t: number) => {
    setSelectedTime(t);
    if (videoRef.current) {
      videoRef.current.currentTime = t;
    }
  };

  return (
    <div className="container-page py-8 md:py-12 bg-grain relative">
      <Reveal>
        <p className="font-mono text-xs uppercase tracking-[0.3em] text-signal/80">SyncGuard // Workspace</p>
        <h1 className="font-display text-4xl sm:text-5xl md:text-6xl mt-3">Analyze media</h1>
        <p className="mt-3 max-w-xl text-sm sm:text-base text-ink-soft leading-relaxed">
          Detect synthetic speech, or examine audio-visual synchronization with the real inference
          pipeline — the same one described in the technology pages.
        </p>
      </Reveal>

      <Reveal className="mt-6 flex gap-1 border-b border-[color:var(--page-border)]">
        {(
          [
            { key: "av", label: "Video / Audio + Visual" },
            { key: "audio", label: "Audio Only" },
          ] as const
        ).map((tab) => (
          <button
            key={tab.key}
            onClick={() => switchMode(tab.key)}
            className={`px-5 py-3 font-mono text-xs sm:text-sm uppercase tracking-[0.1em] transition-colors duration-200 border-b-2 -mb-px ${
              mode === tab.key ? "border-signal text-signal" : "border-transparent text-ink-faint hover:text-ink-soft"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </Reveal>

      {mode === "av" ? (
        <div className="mt-6 grid lg:grid-cols-2 gap-10 lg:gap-12">
          <div>
            {videoUrl ? (
              <MediaFrame src={videoUrl} label="Video Source" videoRef={videoRef} />
            ) : (
              <UploadDropzone
                label="Drop your video here"
                hint="MP4 · MOV · AVI · MKV"
                accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
                file={videoFile}
                onSelect={setVideoFile}
              />
            )}
            <div className="mt-3 flex flex-wrap items-center gap-x-6 gap-y-2">
              {videoFile && (
                <button
                  className="font-mono text-xs uppercase tracking-wide text-ink-faint hover:text-signal underline underline-offset-4"
                  onClick={() => {
                    setVideoFile(null);
                    setAvResult(null);
                  }}
                >
                  Remove file
                </button>
              )}

              <button
                className="font-mono text-xs uppercase tracking-[0.15em] text-ink-faint hover:text-signal"
                onClick={() => setShowAdvanced((v) => !v)}
              >
                {showAdvanced ? "Hide" : "Show"} advanced options
              </button>
            </div>
            {showAdvanced && (
              <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-4 fade-in">
                <UploadDropzone
                  label="Separate audio"
                  hint="Optional · WAV / FLAC"
                  accept="audio/wav,audio/flac,audio/x-flac"
                  file={avAudioFile}
                  onSelect={setAvAudioFile}
                />
                <UploadDropzone
                  label="Precomputed landmarks"
                  hint="Optional · .npz"
                  accept=".npz"
                  file={landmarksFile}
                  onSelect={setLandmarksFile}
                />
              </div>
            )}

            <Button className="mt-4 w-full sm:w-auto" disabled={!videoFile || loading} onClick={runAv}>
              {loading ? "Analyzing…" : "Run AV Analysis →"}
            </Button>
            {error && (
              <ErrorState
                message={error}
                onRetry={() => {
                  setError(null);
                  runAv();
                }}
                onChooseAnother={() => {
                  setError(null);
                  setVideoFile(null);
                }}
              />
            )}
          </div>

          <div>
            {loading && <AnalyzingState stages={AV_STAGES} />}
            {!loading && avResult && (
              <AvResultPanel
                result={avResult}
                onOpenReport={openAvReport}
                onSeek={handleSeek}
                selectedTime={selectedTime}
              />
            )}
            {!loading && !avResult && !error && (
              <EmptyState text="Your synchronization analysis will appear here." />
            )}
          </div>
        </div>
      ) : (
        <div className="mt-6 grid lg:grid-cols-2 gap-10 lg:gap-12">
          <div>
            <UploadDropzone
              label="Drop audio file here"
              hint="WAV · FLAC · MP3"
              accept="audio/wav,audio/flac,audio/mpeg,audio/mp3"
              file={audioFile}
              onSelect={setAudioFile}
            />
            {audioFile && (
              <div className="mt-3">
                <button
                  className="font-mono text-xs uppercase tracking-wide text-ink-faint hover:text-signal underline underline-offset-4"
                  onClick={() => {
                    setAudioFile(null);
                    setAudioResult(null);
                  }}
                >
                  Remove file
                </button>
              </div>
            )}
            <Button className="mt-4 w-full sm:w-auto" disabled={!audioFile || loading} onClick={runAudio}>
              {loading ? "Analyzing…" : "Run Audio Analysis →"}
            </Button>
            {error && (
              <ErrorState
                message={error}
                onRetry={() => {
                  setError(null);
                  runAudio();
                }}
                onChooseAnother={() => {
                  setError(null);
                  setAudioFile(null);
                }}
              />
            )}
          </div>

          <div>
            {loading && <AnalyzingState stages={AUDIO_STAGES} />}
            {!loading && audioResult && <AudioResultPanel result={audioResult} onOpenReport={openAudioReport} />}
            {!loading && !audioResult && !error && (
              <EmptyState text="Your authenticity result will appear here." />
            )}
          </div>
        </div>
      )}

      <RecentAnalyses refreshKey={historyTick} />
    </div>
  );
}

function EmptyState({ text }: { text: string }) {
  return (
    <div
      className="h-full min-h-[360px] border bg-canvas-raised/30 flex flex-col items-center justify-center gap-4 text-center px-8"
      style={{ borderColor: "var(--page-border)", borderStyle: "dashed" }}
    >
      <svg width="32" height="32" viewBox="0 0 32 32" className="opacity-30">
        <circle cx="16" cy="16" r="11" fill="none" stroke="currentColor" strokeWidth="1.25" />
        <path d="M16 11v6 M16 20.5v.01" stroke="currentColor" strokeWidth="1.25" strokeLinecap="round" />
      </svg>
      <p className="font-mono text-sm text-ink-faint max-w-xs">{text}</p>
    </div>
  );
}

function ErrorState({
  message,
  onRetry,
  onChooseAnother,
}: {
  message: string;
  onRetry: () => void;
  onChooseAnother: () => void;
}) {
  return (
    <div className="mt-5 border px-5 py-4 bg-anomaly/5" style={{ borderColor: "var(--page-anomaly)" }} role="alert">
      <p className="font-mono text-xs uppercase tracking-[0.15em] text-anomaly">Unable to analyze this file</p>
      <p className="mt-2 text-sm text-ink-soft">{message}</p>
      <div className="mt-4 flex gap-5 text-xs">
        <button onClick={onRetry} className="font-mono uppercase tracking-wide underline underline-offset-4 text-ink-soft hover:text-signal">
          Try again
        </button>
        <button onClick={onChooseAnother} className="font-mono uppercase tracking-wide underline underline-offset-4 text-ink-soft hover:text-signal">
          Choose another file
        </button>
      </div>
    </div>
  );
}

function AvResultPanel({
  result,
  onOpenReport,
  onSeek,
  selectedTime,
}: {
  result: AudioVisualResult;
  onOpenReport: () => void;
  onSeek: (t: number) => void;
  selectedTime: number | null;
}) {
  const isSync = result.predicted_label === "sync";
  const meta = result.timing_metadata;
  const duration = meta?.num_frames && meta?.fps ? meta.num_frames / meta.fps : null;

  return (
    <div className="fade-in border bg-canvas-raised/40 p-6 sm:p-8" style={{ borderColor: "var(--page-border-strong)" }}>
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-ink-faint">Audio-Visual Synchronization</p>

      <div className="mt-5 flex flex-wrap items-center gap-6">
        <RadialGauge
          value={result.aggregate_sync_score * 100}
          label={isSync ? "Synchronized" : "Desynchronized"}
          tone={isSync ? "signal" : "anomaly"}
          size={120}
        />
      </div>

      <div className="mt-8 grid grid-cols-2 gap-3">
        <ScoreCard label="Sync Score" value={`${(result.sync_probability * 100).toFixed(2)}%`} />
        <ScoreCard label="Desync Score" value={`${(result.desync_probability * 100).toFixed(2)}%`} />
      </div>

      {result.per_window_sync_scores && result.per_window_sync_scores.length > 0 && (
        <div className="mt-8">
          <p className="font-mono text-xs uppercase tracking-[0.2em] text-ink-faint mb-3">Synchronization Timeline</p>
          <SyncTimeline scores={result.per_window_sync_scores} metadata={meta} onSeek={onSeek} selectedTime={selectedTime} />
        </div>
      )}

      <div className="mt-8 grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm">
        <Meta label="Duration" value={duration ? `${duration.toFixed(1)}s` : "—"} />
        <Meta label="Windows Analyzed" value={meta?.num_windows?.toString() ?? "—"} />
        <Meta label="Frame Rate" value={meta?.fps ? `${meta.fps.toFixed(1)} fps` : "—"} />
        <Meta
          label="Temporal Analysis"
          value={
            meta?.av_inference_mode === "windowed" && meta.window_frames
              ? `Windowed (${meta.window_frames}f)`
              : "Full clip"
          }
        />
      </div>

      <p className="mt-8 text-sm leading-relaxed text-ink-soft border-l-2 pl-4" style={{ borderColor: "var(--page-signal)" }}>
        AV mode detects temporal synchronization inconsistencies between audio and visual streams. It does not
        directly detect content manipulation. A synchronized manipulated video can still be a deepfake.
      </p>

      <button onClick={onOpenReport} className="mt-6 font-mono text-xs uppercase tracking-wide underline underline-offset-4 text-ink-soft hover:text-signal">
        Open full report →
      </button>
    </div>
  );
}

function AudioResultPanel({ result, onOpenReport }: { result: AudioOnlyResult; onOpenReport: () => void }) {
  const isBonafide = result.predicted_label === "bonafide";
  const primaryScore = isBonafide ? result.bonafide_probability : result.spoof_probability;

  return (
    <div className="fade-in border bg-canvas-raised/40 p-6 sm:p-8" style={{ borderColor: "var(--page-border-strong)" }}>
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-ink-faint">Audio Authenticity</p>

      <div className="mt-5">
        <RadialGauge
          value={primaryScore * 100}
          label={isBonafide ? "Bonafide / Real" : "Spoof / Synthetic"}
          tone={isBonafide ? "signal" : "anomaly"}
          size={120}
        />
      </div>

      <div className="mt-8 grid grid-cols-2 gap-3">
        <ScoreCard label="Bonafide Score" value={result.bonafide_probability.toFixed(4)} />
        <ScoreCard label="Spoof Score" value={result.spoof_probability.toFixed(4)} />
      </div>

      <div className="mt-8 text-sm">
        <Meta label="Model" value={result.use_ensemble ? "CNN + Transformer Ensemble" : "Transformer"} />
      </div>

      <p className="mt-8 text-sm leading-relaxed text-ink-soft border-l-2 pl-4" style={{ borderColor: "var(--page-signal)" }}>
        Scores are raw model outputs, not calibrated probabilities. Audio-only mode evaluates whether the
        speech signal is likely bonafide or spoofed, based on acoustic features learned from ASVspoof 2019 LA.
      </p>

      <button onClick={onOpenReport} className="mt-6 font-mono text-xs uppercase tracking-wide underline underline-offset-4 text-ink-soft hover:text-signal">
        Open full report →
      </button>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="font-mono text-[11px] uppercase tracking-[0.15em] text-ink-faint">{label}</p>
      <p className="mt-1 font-display text-xl">{value}</p>
    </div>
  );
}
