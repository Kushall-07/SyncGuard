"""Real production AV end-to-end regression fixture (Phase 6 of the production
AV-inference correctness audit).

`tests/test_backend_main.py` mocks `get_predictor()`/`get_landmarker()`
throughout, so it never actually exercises: real checkpoint loading, real
MediaPipe landmark extraction, real audio decode, or the real frozen
encoders -> alignment -> cross-attention -> SyncHead pipeline through the
actual FastAPI route. This module closes that gap using
`data/lavdf/test_fixture.mp4` - a tiny (128x128, 10-frame, 25fps, ~0.19s
embedded-audio) repository-safe fixture already on disk, not a copyrighted
LAV-DF sample - to genuinely exercise: video + audio + landmarks + the real
`SyncGuardPredictor` + `/api/analyze/av`, end to end.

This is a regression/plumbing test, not a claim of scientific accuracy: the
fixture has no real face, so MediaPipe finds zero valid landmark frames and
the resulting sync/desync label carries no evidentiary meaning. What it does
prove is that the full production code path - upload -> temp file -> audio
extraction -> on-the-fly landmark extraction -> windowing -> frozen
encoders -> cross-attention -> SyncHead -> JSON response - runs without
raising, produces a well-formed, schema-valid result, and leaves the backend
healthy for a subsequent request.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import backend.main as backend_main

_FIXTURE_VIDEO = _repo_root / "data" / "lavdf" / "test_fixture.mp4"

_REQUIRED_PATHS = [
    backend_main.VISUAL_ENCODER_PATH,
    backend_main.SYNC_MODEL_PATH,
    backend_main.SYNC_CONFIG_PATH,
    backend_main.SPOOF_HEAD_PATH,
    _FIXTURE_VIDEO,
]
_missing = [str(p) for p in _REQUIRED_PATHS if not Path(p).exists()]

pytestmark = pytest.mark.skipif(
    bool(_missing),
    reason=f"production checkpoints/fixture not present on disk: {_missing}",
)


@pytest.fixture(scope="module")
def real_predictor() -> "backend_main.SyncGuardPredictor":
    """A genuine `SyncGuardPredictor` built from the exact checkpoint paths
    `backend/main.py` uses in production, loaded once per test module (real
    checkpoint loading is not free) - no ensemble CNN, since only the AV path
    is under test here."""
    from src.inference.predictor import SyncGuardPredictor

    return SyncGuardPredictor(
        visual_encoder_path=backend_main.VISUAL_ENCODER_PATH,
        sync_model_path=backend_main.SYNC_MODEL_PATH,
        sync_config_path=backend_main.SYNC_CONFIG_PATH,
        spoof_head_checkpoint=backend_main.SPOOF_HEAD_PATH,
        device="cpu",
        av_inference_mode=backend_main.AV_INFERENCE_MODE,
        window_frames=backend_main.AV_WINDOW_FRAMES,
        stride_frames=backend_main.AV_STRIDE_FRAMES,
    )


@pytest.fixture(scope="module")
def real_landmarker():
    """The exact MediaPipe FaceLandmarker `backend/main.py`'s `get_landmarker()`
    constructs.

    Explicitly `.close()`d at module teardown: MediaPipe's Tasks Python API
    holds its own native thread pool for the underlying TFLite runtime, and
    leaving it open let it outlive this module in the same pytest process -
    which manifested as later, unrelated `TestClient`-based tests
    (`starlette.testclient`'s own AnyIO portal thread) hanging indefinitely
    when the full suite ran end to end, even though every test passes when
    run in isolation. Closing it here releases that thread pool before the
    next test module runs.
    """
    from scripts.extract_celebdf_landmarks import DEFAULT_MODEL, ensure_model, make_landmarker

    model_path = ensure_model(DEFAULT_MODEL)
    landmarker = make_landmarker(model_path)
    yield landmarker
    landmarker.close()


@pytest.fixture()
def client(real_predictor, real_landmarker) -> TestClient:
    with patch.object(backend_main, "get_predictor", return_value=real_predictor), patch.object(
        backend_main, "get_landmarker", return_value=real_landmarker
    ):
        yield TestClient(backend_main.app)


def _post_fixture_video(client: TestClient):
    with _FIXTURE_VIDEO.open("rb") as f:
        return client.post(
            "/api/analyze/av",
            files={"video": ("test_fixture.mp4", f, "video/mp4")},
        )


def test_real_av_analyze_returns_well_formed_result(client: TestClient) -> None:
    res = _post_fixture_video(client)
    assert res.status_code == 200
    body = res.json()

    assert body["predicted_label"] in ("sync", "desync")
    assert 0.0 <= body["sync_probability"] <= 1.0
    assert 0.0 <= body["desync_probability"] <= 1.0
    assert 0.0 <= body["aggregate_sync_score"] <= 1.0
    assert isinstance(body["per_window_sync_scores"], list) and len(body["per_window_sync_scores"]) >= 1
    assert all(0.0 <= s <= 1.0 for s in body["per_window_sync_scores"])

    timing = body["timing_metadata"]
    assert timing is not None
    for key in ("fps", "num_frames", "window_seconds", "audio_token_seconds", "num_windows", "av_inference_mode"):
        assert key in timing, f"missing timing_metadata.{key}"
    # This is the exact quantity Phase 1 fixed: must reflect the real encoder
    # architecture (0.08s for the production [32, 64, 128]-channel CNN), never
    # the old hardcoded 0.01s.
    assert timing["audio_token_seconds"] == pytest.approx(0.08)


def test_backend_stays_healthy_after_two_consecutive_av_requests(client: TestClient) -> None:
    res1 = _post_fixture_video(client)
    assert res1.status_code == 200

    res2 = _post_fixture_video(client)
    assert res2.status_code == 200

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ready"
