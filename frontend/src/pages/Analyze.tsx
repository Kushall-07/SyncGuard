import { useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import Button from "../components/Button";
import UploadDropzone from "../components/UploadDropzone";
import ScoreCard from "../components/ScoreCard";
import SyncTimeline from "../components/SyncTimeline";
import AnalyzingState from "../components/AnalyzingState";
import RecentAnalyses from "../components/RecentAnalyses";
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
    <div className="container-page py-16">
      <p className="text-xs uppercase tracking-[0.2em] opacity-60">Analyze Media</p>
      <h1 className="font-display text-4xl md:text-5xl mt-3">Upload and analyze</h1>
      <p className="mt-4 max-w-xl opacity-70">
        Detect synthetic speech or examine audio-visual synchronization.
      </p>

      <div className="mt-10 flex gap-2 border-b" style={{ borderColor: "var(--page-border)" }}>
        {(
          [
            { key: "av", label: "Video / Audio + Visual" },
            { key: "audio", label: "Audio Only" },
          ] as const
        ).map((tab) => (
          <button
            key={tab.key}
            onClick={() => switchMode(tab.key)}
            className={`px-5 py-3 text-sm transition-colors duration-200 border-b-2 -mb-px ${
              mode === tab.key ? "border-sage" : "border-transparent opacity-60 hover:opacity-100"
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {mode === "av" ? (
        <div className="mt-10 grid md:grid-cols-2 gap-10">
          <div>
            {videoUrl ? (
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                className="w-full border"
                style={{ borderColor: "var(--page-border)" }}
              />
            ) : (
              <UploadDropzone
                label="Drop your video here"
                hint="MP4 · MOV · AVI · MKV"
                accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
                file={videoFile}
                onSelect={setVideoFile}
              />
            )}
            <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2">
              {videoFile && (
                <button
                  className="text-xs opacity-60 hover:opacity-100 underline"
                  onClick={() => {
                    setVideoFile(null);
                    setAvResult(null);
                  }}
                >
                  Remove file
                </button>
              )}

              <button
                className="text-xs uppercase tracking-widest opacity-60 hover:opacity-100"
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

            <Button className="mt-8 w-full sm:w-auto" disabled={!videoFile || loading} onClick={runAv}>
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
        <div className="mt-10 grid md:grid-cols-2 gap-10">
          <div>
            <UploadDropzone
              label="Drop audio file here"
              hint="WAV · FLAC · MP3"
              accept="audio/wav,audio/flac,audio/mpeg,audio/mp3"
              file={audioFile}
              onSelect={setAudioFile}
            />
            {audioFile && (
              <button
                className="mt-3 text-xs opacity-60 hover:opacity-100 underline"
                onClick={() => {
                  setAudioFile(null);
                  setAudioResult(null);
                }}
              >
                Remove file
              </button>
            )}
            <Button className="mt-8 w-full sm:w-auto" disabled={!audioFile || loading} onClick={runAudio}>
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
      className="h-full min-h-[280px] border border-dashed flex items-center justify-center text-sm opacity-50 text-center px-8"
      style={{ borderColor: "var(--page-border)" }}
    >
      {text}
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
    <div className="mt-4 border-l-2 pl-4 py-1" style={{ borderColor: "var(--copper)" }} role="alert">
      <p className="text-sm text-copper-dark">Unable to analyze this file.</p>
      <p className="mt-1 text-xs opacity-60">{message}</p>
      <div className="mt-3 flex gap-4 text-xs">
        <button onClick={onRetry} className="underline opacity-80 hover:opacity-100">
          Try again
        </button>
        <button onClick={onChooseAnother} className="underline opacity-80 hover:opacity-100">
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
    <div className="fade-in">
      <p className="text-xs uppercase tracking-widest opacity-60">Audio-Visual Synchronization</p>
      <div className="mt-3 flex items-baseline gap-4">
        <h2 className={`font-display text-3xl ${isSync ? "text-sage-dark" : "text-copper-dark"}`}>
          {isSync ? "Synchronized" : "Desynchronized"}
        </h2>
        <span className="font-display text-2xl opacity-70">
          {(result.aggregate_sync_score * 100).toFixed(2)}%
        </span>
      </div>
      <p className="text-xs opacity-50">Sync Score</p>

      <div className="mt-6 grid grid-cols-3 gap-3">
        <ScoreCard label="Sync Score" value={`${(result.sync_probability * 100).toFixed(2)}%`} />
        <ScoreCard label="Desync Score" value={`${(result.desync_probability * 100).toFixed(2)}%`} />
        <ScoreCard label="Aggregate Score" value={`${(result.aggregate_sync_score * 100).toFixed(2)}%`} />
      </div>

      {result.per_window_sync_scores && result.per_window_sync_scores.length > 0 && (
        <div className="mt-8">
          <p className="text-xs uppercase tracking-widest opacity-60 mb-3">Synchronization Timeline</p>
          <SyncTimeline scores={result.per_window_sync_scores} metadata={meta} onSeek={onSeek} selectedTime={selectedTime} />
        </div>
      )}

      <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-3 text-sm">
        <Meta label="Duration" value={duration ? `${duration.toFixed(1)}s` : "—"} />
        <Meta label="Windows Analyzed" value={meta?.num_windows?.toString() ?? "—"} />
        <Meta label="Frame Rate" value={meta?.fps ? `${meta.fps.toFixed(1)} fps` : "—"} />
        <Meta label="Analysis Mode" value="Audio-Visual" />
      </div>

      <p className="mt-8 text-xs leading-relaxed opacity-60 border-l-2 pl-4" style={{ borderColor: "var(--sage)" }}>
        AV mode detects temporal synchronization inconsistencies between audio and visual streams. It does not
        directly detect content manipulation. A synchronized manipulated video can still be a deepfake.
      </p>

      <button onClick={onOpenReport} className="mt-6 text-xs underline opacity-70 hover:opacity-100">
        Open full report →
      </button>
    </div>
  );
}

function AudioResultPanel({ result, onOpenReport }: { result: AudioOnlyResult; onOpenReport: () => void }) {
  const isBonafide = result.predicted_label === "bonafide";
  return (
    <div className="fade-in">
      <p className="text-xs uppercase tracking-widest opacity-60">Audio Authenticity</p>
      <h2 className={`mt-3 font-display text-3xl ${isBonafide ? "text-sage-dark" : "text-copper-dark"}`}>
        {isBonafide ? "Bonafide / Real" : "Spoof / Synthetic"}
      </h2>

      <div className="mt-6 grid grid-cols-2 gap-3">
        <ScoreCard label="Bonafide Score" value={result.bonafide_probability.toFixed(4)} />
        <ScoreCard label="Spoof Score" value={result.spoof_probability.toFixed(4)} />
      </div>

      <div className="mt-6 text-sm">
        <Meta label="Model" value={result.use_ensemble ? "CNN + Transformer Ensemble" : "Transformer"} />
      </div>

      <p className="mt-8 text-xs leading-relaxed opacity-60 border-l-2 pl-4" style={{ borderColor: "var(--sage)" }}>
        Scores are raw model outputs, not calibrated probabilities. Audio-only mode evaluates whether the
        speech signal is likely bonafide or spoofed, based on acoustic features learned from ASVspoof 2019 LA.
      </p>

      <button onClick={onOpenReport} className="mt-6 text-xs underline opacity-70 hover:opacity-100">
        Open full report →
      </button>
    </div>
  );
}

function Meta({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-widest opacity-50">{label}</p>
      <p className="mt-1 font-display text-lg">{value}</p>
    </div>
  );
}
