"""SyncGuard API (thin FastAPI wrapper around the existing inference pipeline).

This module contains NO model logic, preprocessing, or scoring code. It only:
- accepts uploaded media over HTTP,
- writes it to a temp file (SyncGuardPredictor's public API takes file paths),
- calls the existing `SyncGuardPredictor`,
- serializes the returned dataclasses to JSON.

The checkpoint paths and predictor construction mirror app/app.py exactly so both
front ends stay backed by the identical model weights and config.
"""

from __future__ import annotations

import logging
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.inference import SyncGuardPredictor  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Checkpoint paths (identical to app/app.py — do not diverge from the working demo).
CHECKPOINT_DIR = _repo_root / "outputs" / "runs"
SPOOF_HEAD_PATH = CHECKPOINT_DIR / "spoof-transformer-20260906-123646" / "checkpoints" / "best.pt"
VISUAL_ENCODER_PATH = CHECKPOINT_DIR / "deepfake-transformer-final-20260908-210034" / "checkpoints" / "visual_encoder.pt"
SYNC_MODEL_PATH = CHECKPOINT_DIR / "sync-phase12-lambda01-20260913-115101" / "checkpoints" / "best.pt"
SYNC_CONFIG_PATH = _repo_root / "configs" / "av_align_lambda01.yaml"
CNN_CHECKPOINT_PATH = CHECKPOINT_DIR / "spoof-cnn-baseline-20260906-104508" / "checkpoints" / "best.pt"

# Synchronization Lab: a small curated set of LAV-DF clips that have precomputed
# audio + MediaPipe landmarks on disk, so repeated shift experiments are fast and
# don't require re-running audio extraction / MediaPipe per shift value.
LAVDF_DIR = _repo_root / "data" / "lavdf"
LAB_SAMPLES: dict[str, dict[str, Path]] = {
    "004073": {
        "video": LAVDF_DIR / "extracted" / "dev" / "004073.mp4",
        "audio": LAVDF_DIR / "processed" / "audio" / "004073.wav",
        "landmarks": LAVDF_DIR / "processed" / "landmarks" / "004073.npz",
    },
    "004078": {
        "video": LAVDF_DIR / "extracted" / "dev" / "004078.mp4",
        "audio": LAVDF_DIR / "processed" / "audio" / "004078.wav",
        "landmarks": LAVDF_DIR / "processed" / "landmarks" / "004078.npz",
    },
    "004100": {
        "video": LAVDF_DIR / "extracted" / "dev" / "004100.mp4",
        "audio": LAVDF_DIR / "processed" / "audio" / "004100.wav",
        "landmarks": LAVDF_DIR / "processed" / "landmarks" / "004100.npz",
    },
}
# Only non-negative shifts are exposed: this alignment scheme produces zero valid
# overlapping windows for negative shifts (verified against the Phase 11 controlled
# shift-sensitivity evaluation), so negative shifts are not a usable configuration.
LAB_SHIFTS: list[float] = [0.0, 0.5, 1.0, 2.0]

_predictor: SyncGuardPredictor | None = None
_landmarker: Any = None

# Generic-exception detail shown to clients: the actual exception (which can include
# server-side temp file paths or third-party library internals) is logged via
# logger.exception() above each raise site, never returned in the HTTP response.
UNEXPECTED_ERROR_DETAIL = "Unable to analyze this file. Please try again or choose another file."

app = FastAPI(title="SyncGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_predictor() -> SyncGuardPredictor:
    global _predictor
    if _predictor is None:
        logger.info("Initializing SyncGuardPredictor...")
        _predictor = SyncGuardPredictor(
            visual_encoder_path=VISUAL_ENCODER_PATH,
            sync_model_path=SYNC_MODEL_PATH,
            sync_config_path=SYNC_CONFIG_PATH,
            spoof_head_checkpoint=SPOOF_HEAD_PATH,
            cnn_checkpoint=CNN_CHECKPOINT_PATH,
            device="auto",
        )
        logger.info(f"Predictor initialized on device: {_predictor.device}")
    return _predictor


def get_landmarker():
    global _landmarker
    if _landmarker is None:
        from scripts.extract_celebdf_landmarks import DEFAULT_MODEL, ensure_model, make_landmarker

        model_path = ensure_model(DEFAULT_MODEL)
        _landmarker = make_landmarker(model_path)
        logger.info("MediaPipe landmarker initialized")
    return _landmarker


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy scalar/array types to plain Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _save_upload(upload: UploadFile, suffix: str) -> Path:
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    tmp_path = Path(tmp_name)
    with open(fd, "wb") as f:
        f.write(upload.file.read())
    return tmp_path


@app.get("/api/health")
def health() -> dict:
    try:
        predictor = get_predictor()
        return {
            "status": "ready",
            "gpu_ready": predictor.device.type == "cuda",
            "device": str(predictor.device),
        }
    except Exception as e:  # noqa: BLE001 - surface init failure without crashing app
        logger.error(f"Predictor not ready: {e}")
        return {"status": "unavailable", "gpu_ready": False, "device": "unknown"}


@app.post("/api/analyze/audio")
async def analyze_audio(audio: UploadFile = File(...)) -> dict:
    suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
    tmp_path = _save_upload(audio, suffix)
    try:
        predictor = get_predictor()
        result = predictor.predict_audio(tmp_path)
        return _to_jsonable(asdict(result))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"File not found: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {e}") from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected error during audio analysis")
        raise HTTPException(status_code=500, detail=UNEXPECTED_ERROR_DETAIL) from e
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/analyze/av")
async def analyze_av(
    video: UploadFile = File(...),
    audio: UploadFile | None = None,
    landmarks: UploadFile | None = None,
) -> dict:
    video_suffix = Path(video.filename or "video.mp4").suffix or ".mp4"
    video_path = _save_upload(video, video_suffix)
    audio_path = _save_upload(audio, Path(audio.filename or "audio.wav").suffix or ".wav") if audio else None
    landmarks_path = _save_upload(landmarks, ".npz") if landmarks else None
    try:
        predictor = get_predictor()
        landmarker = get_landmarker() if landmarks_path is None else None
        result = predictor.predict_audio_visual(
            video_path=video_path,
            audio_path=audio_path,
            landmarks_path=landmarks_path,
            landmarker=landmarker,
        )
        return _to_jsonable(asdict(result))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"File not found: {e}") from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Processing error: {e}") from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected error during AV analysis")
        raise HTTPException(status_code=500, detail=UNEXPECTED_ERROR_DETAIL) from e
    finally:
        video_path.unlink(missing_ok=True)
        if audio_path:
            audio_path.unlink(missing_ok=True)
        if landmarks_path:
            landmarks_path.unlink(missing_ok=True)


class LabAnalyzeRequest(BaseModel):
    sample_id: str
    shift_seconds: float


@app.get("/api/lab/samples")
def lab_samples() -> dict:
    """List curated Synchronization Lab samples that are actually present on disk."""
    samples = [
        {"id": sample_id, "label": f"LAV-DF Clip {sample_id}", "video_url": f"/api/lab/samples/{sample_id}/video"}
        for sample_id, paths in LAB_SAMPLES.items()
        if all(p.is_file() for p in paths.values())
    ]
    return {"samples": samples, "available_shifts": LAB_SHIFTS}


@app.get("/api/lab/samples/{sample_id}/video")
def lab_sample_video(sample_id: str) -> FileResponse:
    paths = LAB_SAMPLES.get(sample_id)
    if paths is None or not paths["video"].is_file():
        raise HTTPException(status_code=404, detail=f"Unknown sample id: {sample_id}")
    return FileResponse(paths["video"], media_type="video/mp4")


@app.post("/api/lab/analyze")
def lab_analyze(request: LabAnalyzeRequest) -> dict:
    """Run the controlled temporal-shift synchronization experiment for the Lab.

    This is a thin wrapper around `SyncGuardPredictor.predict_sync_lab`, which
    reuses the existing frozen encoders, cross-attention, and sync head — no
    scoring logic lives in this route handler.
    """
    paths = LAB_SAMPLES.get(request.sample_id)
    if paths is None or not all(p.is_file() for p in paths.values()):
        raise HTTPException(status_code=400, detail=f"Unknown sample id: {request.sample_id}")
    if request.shift_seconds not in LAB_SHIFTS:
        raise HTTPException(status_code=400, detail=f"shift_seconds must be one of {LAB_SHIFTS}")
    try:
        predictor = get_predictor()
        result = predictor.predict_sync_lab(
            audio_path=paths["audio"],
            landmarks_path=paths["landmarks"],
            shift_seconds=request.shift_seconds,
            sample_id=request.sample_id,
        )
        return _to_jsonable(asdict(result))
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=f"File not found: {e}") from e
    except Exception as e:  # noqa: BLE001
        logger.exception("Unexpected error during sync lab analysis")
        raise HTTPException(status_code=500, detail=UNEXPECTED_ERROR_DETAIL) from e
