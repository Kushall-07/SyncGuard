"""Tests for the SyncGuard launcher (app/app.py).

`app/app.py` is no longer the Gradio demo UI - it is a thin one-command
launcher that starts the existing FastAPI backend (backend/main.py) and the
existing React/Vite frontend (frontend/) as child processes, waits for both to
become reachable, and shuts them down on Ctrl+C. It contains no ML/inference
logic and no API routes of its own.

This replaces the old Gradio-era test_app.py, whose 11 of 12 tests
unconditionally skipped (they referenced `app.app.get_predictor`,
`app.app.create_ui`, `app.app.process_audio_only`, etc. - none of which exist
in the current launcher), which made the suite misleadingly report a large
skip count instead of real coverage. FastAPI route coverage (backend health,
audio/AV endpoints) lives in `tests/test_backend_main.py` (mocked predictor)
and `tests/test_backend_av_e2e.py` (real end-to-end); this file covers only
what's actually in `app/app.py`: process orchestration and readiness checks.
"""

from __future__ import annotations

import socket
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import URLError

import pytest

_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import app.app as launcher_module


def test_launcher_imports_successfully() -> None:
    assert launcher_module is not None
    assert hasattr(launcher_module, "SyncGuardLauncher")


def test_launcher_has_no_ml_or_route_logic() -> None:
    """The launcher must stay a pure process-orchestration script - it should
    never define its own inference/scoring functions or FastAPI routes, which
    belong exclusively to backend/main.py."""
    forbidden = ("get_predictor", "create_ui", "process_audio_only", "process_audio_visual")
    for name in forbidden:
        assert not hasattr(launcher_module, name), (
            f"app.app should not define {name!r} - inference logic belongs in backend/main.py"
        )


def test_launcher_urls_are_internally_consistent() -> None:
    assert launcher_module.BACKEND_URL == f"http://{launcher_module.BACKEND_HOST}:{launcher_module.BACKEND_PORT}"
    assert launcher_module.FRONTEND_URL == f"http://{launcher_module.FRONTEND_HOST}:{launcher_module.FRONTEND_PORT}"
    assert launcher_module.BACKEND_HEALTH_URL == f"{launcher_module.BACKEND_URL}/api/health"
    assert launcher_module.BACKEND_PORT != launcher_module.FRONTEND_PORT


def test_venv_python_prefers_project_venv_when_present() -> None:
    """This repo's own .venv exists, so `_venv_python()` must resolve to it,
    not fall back to whatever interpreter happens to be running pytest."""
    venv_python = launcher_module.PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.is_file():
        pytest.skip(".venv not present in this environment")
    assert launcher_module._venv_python() == str(venv_python)


def test_venv_python_falls_back_to_current_interpreter_when_missing(tmp_path: Path) -> None:
    with patch.object(launcher_module, "PROJECT_ROOT", tmp_path):
        assert launcher_module._venv_python() == sys.executable


# --------------------------------------------------------------------------- readiness probes


def test_port_is_listening_true_for_a_real_open_socket() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    try:
        port = sock.getsockname()[1]
        assert launcher_module._port_is_listening("127.0.0.1", port) is True
    finally:
        sock.close()


def test_port_is_listening_false_for_a_closed_port() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # now definitely closed
    assert launcher_module._port_is_listening("127.0.0.1", port) is False


def test_backend_is_ready_true_when_health_reports_ready() -> None:
    fake_response = Mock()
    fake_response.status = 200
    fake_response.read.return_value = b'{"status": "ready"}'
    fake_response.__enter__ = Mock(return_value=fake_response)
    fake_response.__exit__ = Mock(return_value=False)
    with patch.object(launcher_module.urllib.request, "urlopen", return_value=fake_response):
        assert launcher_module._backend_is_ready() is True


def test_backend_is_ready_false_when_status_is_unavailable() -> None:
    fake_response = Mock()
    fake_response.status = 200
    fake_response.read.return_value = b'{"status": "unavailable"}'
    fake_response.__enter__ = Mock(return_value=fake_response)
    fake_response.__exit__ = Mock(return_value=False)
    with patch.object(launcher_module.urllib.request, "urlopen", return_value=fake_response):
        assert launcher_module._backend_is_ready() is False


def test_backend_is_ready_false_on_connection_error() -> None:
    with patch.object(launcher_module.urllib.request, "urlopen", side_effect=URLError("refused")):
        assert launcher_module._backend_is_ready() is False


def test_frontend_is_ready_true_for_2xx_3xx() -> None:
    fake_response = Mock()
    fake_response.status = 200
    fake_response.__enter__ = Mock(return_value=fake_response)
    fake_response.__exit__ = Mock(return_value=False)
    with patch.object(launcher_module.urllib.request, "urlopen", return_value=fake_response):
        assert launcher_module._frontend_is_ready() is True


def test_frontend_is_ready_false_on_connection_error() -> None:
    with patch.object(launcher_module.urllib.request, "urlopen", side_effect=URLError("refused")):
        assert launcher_module._frontend_is_ready() is False


def test_wait_until_returns_true_as_soon_as_check_succeeds() -> None:
    calls = {"n": 0}

    def check() -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    assert launcher_module._wait_until(check, timeout_seconds=5.0) is True
    assert calls["n"] == 3


def test_wait_until_returns_false_on_timeout() -> None:
    assert launcher_module._wait_until(lambda: False, timeout_seconds=0.2) is False


# --------------------------------------------------------------------------- SyncGuardLauncher


def test_start_backend_reuses_already_ready_instance() -> None:
    launcher = launcher_module.SyncGuardLauncher()
    with patch.object(launcher_module, "_port_is_listening", return_value=True), patch.object(
        launcher_module, "_backend_is_ready", return_value=True
    ):
        launcher.start_backend()
    assert launcher.processes == []  # nothing spawned - existing instance reused


def test_start_backend_raises_if_port_busy_with_non_syncguard_process() -> None:
    launcher = launcher_module.SyncGuardLauncher()
    with patch.object(launcher_module, "_port_is_listening", return_value=True), patch.object(
        launcher_module, "_backend_is_ready", return_value=False
    ):
        with pytest.raises(launcher_module.LauncherError, match="already in use"):
            launcher.start_backend()


def test_start_backend_raises_if_process_exits_early() -> None:
    launcher = launcher_module.SyncGuardLauncher()
    fake_proc = Mock()
    fake_proc.poll.return_value = 1  # already exited
    fake_proc.returncode = 1
    with patch.object(launcher_module, "_port_is_listening", return_value=False), patch.object(
        launcher_module.SyncGuardLauncher, "_spawn", return_value=fake_proc
    ):
        with pytest.raises(launcher_module.LauncherError, match="exited early"):
            launcher.start_backend()


def test_start_frontend_raises_without_node_on_path() -> None:
    launcher = launcher_module.SyncGuardLauncher()
    with patch.object(launcher_module, "_port_is_listening", return_value=False), patch.object(
        launcher_module.shutil, "which", return_value=None
    ):
        with pytest.raises(launcher_module.LauncherError, match="Node.js"):
            launcher.start_frontend()


def test_shutdown_terminates_and_waits_for_all_owned_processes() -> None:
    launcher = launcher_module.SyncGuardLauncher()
    proc1, proc2 = Mock(), Mock()
    proc1.poll.return_value = None  # still running
    proc2.poll.return_value = None
    launcher.processes = [proc1, proc2]

    launcher.shutdown()

    proc1.terminate.assert_called_once()
    proc2.terminate.assert_called_once()
    proc1.wait.assert_called_once()
    proc2.wait.assert_called_once()


def test_shutdown_is_a_noop_when_nothing_was_spawned() -> None:
    launcher = launcher_module.SyncGuardLauncher()
    launcher.shutdown()  # must not raise
    assert launcher.processes == []


def test_wait_forever_raises_if_an_owned_process_dies() -> None:
    launcher = launcher_module.SyncGuardLauncher()
    dead_proc = Mock()
    dead_proc.poll.return_value = 137
    launcher.processes = [dead_proc]

    with patch.object(time, "sleep"):
        with pytest.raises(launcher_module.LauncherError, match="exited unexpectedly"):
            launcher.wait_forever()
