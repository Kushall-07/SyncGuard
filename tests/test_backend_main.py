"""Tests for the thin FastAPI backend (backend/main.py).

These tests mock `get_predictor()` throughout, so no checkpoints are loaded and no
real inference runs — they only verify that the route handlers call the predictor
correctly, serialize its dataclasses to JSON, and translate exceptions into the
right HTTP status codes. Route handlers must stay free of scoring logic; these
tests would fail to cover anything meaningful if that logic leaked into the routes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import backend.main as backend_main
from src.inference.predictor import AudioOnlyResult, AudioVisualResult, SyncLabResult


@pytest.fixture()
def client() -> TestClient:
    return TestClient(backend_main.app)


def _audio_result(**overrides) -> AudioOnlyResult:
    base = dict(
        mode="audio_only",
        predicted_label="bonafide",
        spoof_probability=0.02,
        bonafide_probability=0.98,
        confidence=0.98,
        audio_encoder_checkpoint="None",
        spoof_head_checkpoint="ckpt.pt",
        cnn_checkpoint="cnn.pt",
        use_ensemble=True,
    )
    base.update(overrides)
    return AudioOnlyResult(**base)


def _av_result(**overrides) -> AudioVisualResult:
    base = dict(
        mode="audio_visual",
        predicted_label="sync",
        sync_probability=0.72,
        desync_probability=0.28,
        aggregate_sync_score=0.72,
        per_window_sync_scores=[0.7, 0.8, np.float32(0.65)],
        timing_metadata={"fps": np.float64(25.0), "num_frames": np.int64(3), "num_windows": 3},
        audio_encoder_checkpoint="a.pt",
        visual_encoder_checkpoint="v.pt",
        sync_model_checkpoint="s.pt",
    )
    base.update(overrides)
    return AudioVisualResult(**base)


class TestHealth:
    def test_health_ready(self, client: TestClient) -> None:
        mock_predictor = Mock()
        mock_predictor.device = Mock(type="cpu")
        with patch.object(backend_main, "get_predictor", return_value=mock_predictor):
            res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ready"
        assert body["gpu_ready"] is False

    def test_health_unavailable_when_predictor_fails(self, client: TestClient) -> None:
        with patch.object(backend_main, "get_predictor", side_effect=RuntimeError("no checkpoint")):
            res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "unavailable"
        assert body["gpu_ready"] is False


class TestAnalyzeAudio:
    def test_returns_jsonable_result(self, client: TestClient) -> None:
        mock_predictor = Mock()
        mock_predictor.predict_audio.return_value = _audio_result()
        with patch.object(backend_main, "get_predictor", return_value=mock_predictor):
            res = client.post(
                "/api/analyze/audio",
                files={"audio": ("clip.wav", b"fake-bytes", "audio/wav")},
            )
        assert res.status_code == 200
        body = res.json()
        assert body["predicted_label"] == "bonafide"
        assert body["bonafide_probability"] == pytest.approx(0.98)
        mock_predictor.predict_audio.assert_called_once()

    def test_file_not_found_becomes_400(self, client: TestClient) -> None:
        mock_predictor = Mock()
        mock_predictor.predict_audio.side_effect = FileNotFoundError("missing")
        with patch.object(backend_main, "get_predictor", return_value=mock_predictor):
            res = client.post(
                "/api/analyze/audio",
                files={"audio": ("clip.wav", b"x", "audio/wav")},
            )
        assert res.status_code == 400

    def test_unexpected_error_becomes_500_without_leaking_traceback(self, client: TestClient) -> None:
        mock_predictor = Mock()
        mock_predictor.predict_audio.side_effect = RuntimeError("boom")
        with patch.object(backend_main, "get_predictor", return_value=mock_predictor):
            res = client.post(
                "/api/analyze/audio",
                files={"audio": ("clip.wav", b"x", "audio/wav")},
            )
        assert res.status_code == 500
        assert "Traceback" not in res.text


class TestAnalyzeAV:
    def test_returns_jsonable_result_with_numpy_scalars(self, client: TestClient) -> None:
        """Regression test: numpy scalars/arrays in timing_metadata and
        per_window_sync_scores must serialize cleanly (no NumPy JSON errors)."""
        mock_predictor = Mock()
        mock_predictor.predict_audio_visual.return_value = _av_result()
        with patch.object(backend_main, "get_predictor", return_value=mock_predictor), patch.object(
            backend_main, "get_landmarker", return_value=Mock()
        ):
            res = client.post(
                "/api/analyze/av",
                files={"video": ("clip.mp4", b"fake-bytes", "video/mp4")},
            )
        assert res.status_code == 200
        body = res.json()
        assert body["predicted_label"] == "sync"
        assert body["timing_metadata"]["num_windows"] == 3
        assert isinstance(body["per_window_sync_scores"][2], float)


class TestSyncLab:
    @pytest.fixture()
    def fake_sample(self, tmp_path: Path) -> dict[str, Path]:
        video = tmp_path / "sample.mp4"
        audio = tmp_path / "sample.wav"
        landmarks = tmp_path / "sample.npz"
        for f in (video, audio, landmarks):
            f.write_bytes(b"placeholder")
        return {"video": video, "audio": audio, "landmarks": landmarks}

    def test_lists_only_samples_present_on_disk(self, client: TestClient, fake_sample: dict[str, Path], tmp_path: Path) -> None:
        missing_sample = {
            "video": tmp_path / "missing.mp4",
            "audio": tmp_path / "missing.wav",
            "landmarks": tmp_path / "missing.npz",
        }
        with patch.object(backend_main, "LAB_SAMPLES", {"ok": fake_sample, "missing": missing_sample}):
            res = client.get("/api/lab/samples")
        assert res.status_code == 200
        body = res.json()
        ids = [s["id"] for s in body["samples"]]
        assert ids == ["ok"]
        assert body["available_shifts"] == backend_main.LAB_SHIFTS

    def test_analyze_runs_predictor_with_sample_paths(self, client: TestClient, fake_sample: dict[str, Path]) -> None:
        mock_predictor = Mock()
        mock_predictor.predict_sync_lab.return_value = SyncLabResult(
            mode="sync_lab",
            sample_id="ok",
            shift_seconds=0.5,
            aggregate_sync_score=0.71,
            per_window_sync_scores=[0.7, 0.72],
            timing_metadata={"fps": 25.0, "num_windows": 2, "shift_seconds": 0.5},
        )
        with patch.object(backend_main, "LAB_SAMPLES", {"ok": fake_sample}), patch.object(
            backend_main, "get_predictor", return_value=mock_predictor
        ):
            res = client.post("/api/lab/analyze", json={"sample_id": "ok", "shift_seconds": 0.5})
        assert res.status_code == 200
        body = res.json()
        assert body["shift_seconds"] == pytest.approx(0.5)
        assert body["aggregate_sync_score"] == pytest.approx(0.71)
        mock_predictor.predict_sync_lab.assert_called_once_with(
            audio_path=fake_sample["audio"],
            landmarks_path=fake_sample["landmarks"],
            shift_seconds=0.5,
            sample_id="ok",
        )

    def test_unknown_sample_id_rejected(self, client: TestClient) -> None:
        res = client.post("/api/lab/analyze", json={"sample_id": "does-not-exist", "shift_seconds": 0.0})
        assert res.status_code == 400

    def test_negative_shift_rejected(self, client: TestClient, fake_sample: dict[str, Path]) -> None:
        """Negative shifts produce zero valid overlapping windows with this alignment
        scheme, so the Lab only exposes non-negative shifts (see LAB_SHIFTS)."""
        with patch.object(backend_main, "LAB_SAMPLES", {"ok": fake_sample}):
            res = client.post("/api/lab/analyze", json={"sample_id": "ok", "shift_seconds": -0.5})
        assert res.status_code == 400

    def test_video_endpoint_serves_known_sample(self, client: TestClient, fake_sample: dict[str, Path]) -> None:
        with patch.object(backend_main, "LAB_SAMPLES", {"ok": fake_sample}):
            res = client.get("/api/lab/samples/ok/video")
        assert res.status_code == 200
        assert res.content == b"placeholder"

    def test_video_endpoint_404s_for_unknown_id(self, client: TestClient) -> None:
        res = client.get("/api/lab/samples/not-a-real-id/video")
        assert res.status_code == 404
