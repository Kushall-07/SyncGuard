"""Core LAV-DF preprocessing module for audio and landmark extraction (Phase 11A).

Implements:
1. Audio extraction directly from MP4 files (downmix to mono, 16kHz resample, normalize).
2. Landmark extraction using MediaPipe FaceLandmarker and PyAV video stream decoding.
3. Temporal consistency tracking (frame/sample timestamps, duration comparison).
4. Cache validation and resumable sample processing.
5. Missing/corrupted audio handling (marks audio_invalid, fails cache validation, excludes from training success counts).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from src.config import AudioConfig
from src.preprocessing.audio import preprocess_audio
from src.preprocessing.landmarks import N_FACEMESH_POINTS

logger = logging.getLogger(__name__)


def extract_audio_from_mp4(
    mp4_path: str | Path,
    target_sr: int = 16000,
) -> tuple[torch.Tensor, int]:
    """Decode audio stream directly from MP4 using PyAV.

    Parameters
    ----------
    mp4_path:
        Path to the source MP4 file.
    target_sr:
        Target sample rate for resampling.

    Returns
    -------
    tuple[torch.Tensor, int]
        Waveform tensor shaped [1, N] and the sample rate (target_sr).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If no audio stream is present, stream is unreadable, or contains 0 frames.
    """
    import av

    mp4_path = Path(mp4_path)
    if not mp4_path.is_file():
        raise FileNotFoundError(f"Video file not found: {mp4_path}")

    with av.open(str(mp4_path)) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream found in {mp4_path}")

        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sr)
        chunks: list[np.ndarray] = []

        for frame in container.decode(stream):
            resampled_frames = resampler.resample(frame)
            if resampled_frames:
                for rf in resampled_frames:
                    chunks.append(rf.to_ndarray())

        rf = resampler.resample(None)
        if rf:
            for f in rf:
                chunks.append(f.to_ndarray())

        if not chunks:
            raise ValueError(f"Empty or undecodable audio stream in {mp4_path}")

        audio_np = np.concatenate(chunks, axis=1)  # [1, N]
        waveform = torch.from_numpy(audio_np).to(torch.float32)
        return waveform, target_sr


def extract_landmarks_from_mp4(
    mp4_path: str | Path,
    landmarker: Any,
) -> dict[str, Any]:
    """Decode video frames from MP4 using PyAV and extract 478-point MediaPipe landmarks.

    Parameters
    ----------
    mp4_path:
        Path to the source MP4 file.
    landmarker:
        Initialized MediaPipe vision.FaceLandmarker instance.

    Returns
    -------
    dict
        Contains 'points', 'valid', 'fps', 'frame_idx', 'n_frames', 'duration', 'valid_count'.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.
    ValueError
        If no video stream or 0 frames are found.
    """
    import av
    import mediapipe as mp

    mp4_path = Path(mp4_path)
    if not mp4_path.is_file():
        raise FileNotFoundError(f"Video file not found: {mp4_path}")

    with av.open(str(mp4_path)) as container:
        if not container.streams.video:
            raise ValueError(f"No video stream found in {mp4_path}")

        stream = container.streams.video[0]
        fps = None
        if stream.average_rate and float(stream.average_rate) > 0:
            fps = float(stream.average_rate)
        elif stream.r_frame_rate and float(stream.r_frame_rate) > 0:
            fps = float(stream.r_frame_rate)

        if fps is None or fps <= 0:
            raise ValueError(f"Could not determine valid video stream FPS in {mp4_path}")

        frames: list[np.ndarray] = []
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))

    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError(f"No video frames decoded from {mp4_path}")

    points = np.zeros((n_frames, N_FACEMESH_POINTS, 3), dtype=np.float32)
    valid = np.zeros(n_frames, dtype=bool)
    frame_idx = np.arange(n_frames, dtype=np.int64)

    for i, img in enumerate(frames):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img))
        result = landmarker.detect(mp_image)
        if result.face_landmarks:
            lms = result.face_landmarks[0]
            points[i] = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            valid[i] = True

    duration = float(n_frames) / fps
    return {
        "points": points,
        "valid": valid,
        "fps": np.float32(fps),
        "frame_idx": frame_idx,
        "n_frames": n_frames,
        "duration": duration,
        "valid_count": int(valid.sum()),
    }


def validate_audio_cache(wav_path: str | Path, meta_path: str | Path | None = None) -> bool:
    """Check if cached audio output is present, non-empty, readable, and valid."""
    wav_path = Path(wav_path)
    if not wav_path.is_file() or wav_path.stat().st_size == 0:
        return False

    if meta_path is not None:
        meta_path = Path(meta_path)
        if meta_path.is_file():
            try:
                with meta_path.open("r", encoding="utf-8") as f:
                    meta = json.load(f)
                if not meta.get("audio_valid", True):
                    return False
            except Exception:
                return False

    try:
        info = sf.info(str(wav_path))
        if info.channels != 1 or info.samplerate != 16000 or info.frames == 0:
            return False
        return True
    except Exception:
        return False


def validate_landmarks_cache(npz_path: str | Path) -> bool:
    """Check if cached landmark NPZ is present, non-empty, readable, and valid."""
    npz_path = Path(npz_path)
    if not npz_path.is_file() or npz_path.stat().st_size == 0:
        return False

    try:
        with np.load(npz_path) as data:
            if not all(k in data for k in ("points", "valid", "fps", "frame_idx")):
                return False
            points = data["points"]
            valid = data["valid"]
            fps = float(data["fps"])

            if points.ndim != 3 or points.shape[1] != N_FACEMESH_POINTS or points.shape[2] != 3:
                return False
            if valid.ndim != 1 or valid.shape[0] != points.shape[0]:
                return False
            if fps <= 0:
                return False
            return True
    except Exception:
        return False


def process_single_sample(
    sample_id: str,
    mp4_path: str | Path,
    out_audio_dir: str | Path,
    out_landmarks_dir: str | Path,
    landmarker: Any | None = None,
    audio_cfg: AudioConfig | None = None,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Process a single LAV-DF sample, updating audio, landmarks, and sidecar metadata.

    If audio extraction fails/corrupts:
    - Marks audio_valid=False in metadata.
    - Removes any existing invalid audio file.
    - Excludes sample from audio success cache validation.
    - Allows landmark extraction to proceed if possible.
    """
    mp4_path = Path(mp4_path)
    out_audio_dir = Path(out_audio_dir)
    out_landmarks_dir = Path(out_landmarks_dir)

    out_audio_dir.mkdir(parents=True, exist_ok=True)
    out_landmarks_dir.mkdir(parents=True, exist_ok=True)

    audio_path = out_audio_dir / f"{sample_id}.wav"
    landmarks_path = out_landmarks_dir / f"{sample_id}.npz"
    meta_path = out_landmarks_dir / f"{sample_id}.meta.json"

    if audio_cfg is None:
        audio_cfg = AudioConfig()

    audio_valid_cached = validate_audio_cache(audio_path, meta_path)
    landmarks_valid_cached = validate_landmarks_cache(landmarks_path)

    if not force and audio_valid_cached and landmarks_valid_cached and meta_path.is_file():
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["skipped"] = True
            return meta
        except Exception:
            pass

    meta: dict[str, Any] = {
        "sample_id": sample_id,
        "source_path": str(mp4_path),
        "audio_valid": False,
        "landmarks_valid": False,
        "skipped": False,
    }

    # If meta already exists, load existing flags
    if meta_path.is_file() and not force:
        try:
            with meta_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            meta["skipped"] = False
        except Exception:
            pass

    # 1. Process Audio
    if force or not audio_valid_cached:
        try:
            raw_waveform, sr = extract_audio_from_mp4(mp4_path, target_sr=audio_cfg.sample_rate)
            processed_waveform = preprocess_audio(raw_waveform, audio_cfg, source_sr=sr, device="cpu")

            # Write WAV file
            sf.write(
                str(audio_path),
                processed_waveform.squeeze(0).numpy(),
                samplerate=audio_cfg.sample_rate,
                subtype="PCM_16",
            )
            meta["audio_valid"] = True
            meta["audio_samples"] = processed_waveform.shape[-1]
            meta["audio_sample_rate"] = audio_cfg.sample_rate
            meta["audio_duration"] = float(processed_waveform.shape[-1]) / float(audio_cfg.sample_rate)
            meta.pop("audio_error", None)
        except Exception as e:
            logger.warning(f"Audio extraction failed for {sample_id} ({mp4_path}): {e}")
            meta["audio_valid"] = False
            meta["audio_error"] = str(e)
            if audio_path.is_file():
                try:
                    audio_path.unlink()
                except Exception:
                    pass

    # 2. Process Landmarks
    if force or not landmarks_valid_cached:
        if landmarker is None:
            from scripts.extract_celebdf_landmarks import DEFAULT_MODEL, ensure_model, make_landmarker
            model_path = ensure_model(DEFAULT_MODEL)
            landmarker = make_landmarker(model_path)

        try:
            lm_res = extract_landmarks_from_mp4(mp4_path, landmarker)
            np.savez(
                landmarks_path,
                points=lm_res["points"],
                valid=lm_res["valid"],
                fps=lm_res["fps"],
                frame_idx=lm_res["frame_idx"],
            )
            meta["landmarks_valid"] = True
            meta["fps"] = float(lm_res["fps"])
            meta["video_frames"] = lm_res["n_frames"]
            meta["video_duration"] = lm_res["duration"]
            meta["valid_landmark_frames"] = lm_res["valid_count"]
            meta["has_usable_landmarks"] = lm_res["valid_count"] > 0
            meta.pop("landmarks_error", None)
        except Exception as e:
            logger.warning(f"Landmark extraction failed for {sample_id} ({mp4_path}): {e}")
            meta["landmarks_valid"] = False
            meta["landmarks_error"] = str(e)
            if landmarks_path.is_file():
                try:
                    landmarks_path.unlink()
                except Exception:
                    pass

    # Temporal consistency calculations
    if meta.get("audio_valid") and meta.get("landmarks_valid"):
        meta["duration_diff"] = abs(meta["video_duration"] - meta["audio_duration"])
        meta["window_seconds_32"] = 32.0 / meta["fps"]

    # Write sidecar metadata
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta
