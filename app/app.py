"""SyncGuard one-command launcher.

Starts the existing FastAPI backend (backend/main.py) and the existing React/Vite
frontend (frontend/) as child processes, waits for both to become reachable,
prints their URLs, optionally opens a browser, and shuts both down cleanly on
Ctrl+C.

This script contains no ML/inference logic and no API routes of its own — it only
orchestrates the existing backend and frontend dev servers described in the
project README.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)  # status lines show up immediately, even when piped
except (AttributeError, ValueError):
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = PROJECT_ROOT / "frontend"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 5173

BACKEND_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}"
BACKEND_HEALTH_URL = f"{BACKEND_URL}/api/health"
FRONTEND_URL = f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"

STARTUP_TIMEOUT_SECONDS = 120.0
POLL_INTERVAL_SECONDS = 0.5

RULE = "=" * 60
THIN_RULE = "-" * 60


class LauncherError(RuntimeError):
    """Raised for startup failures the user needs to act on."""


def _venv_python() -> str:
    """Prefer the project's virtualenv interpreter; fall back to the current one."""
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def _port_is_listening(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _backend_is_ready() -> bool:
    try:
        with urllib.request.urlopen(BACKEND_HEALTH_URL, timeout=2.0) as res:
            if res.status != 200:
                return False
            body = json.loads(res.read().decode("utf-8"))
            return body.get("status") == "ready"
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _frontend_is_ready() -> bool:
    try:
        with urllib.request.urlopen(FRONTEND_URL, timeout=2.0) as res:
            return 200 <= res.status < 400
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _wait_until(check, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(POLL_INTERVAL_SECONDS)
    return False


class SyncGuardLauncher:
    def __init__(self) -> None:
        self.processes: list[subprocess.Popen] = []

    def _spawn(self, args: list[str], cwd: Path) -> subprocess.Popen:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0
        proc = subprocess.Popen(args, cwd=str(cwd), creationflags=creationflags)
        self.processes.append(proc)
        return proc

    def start_backend(self) -> None:
        print("[1/2] Starting backend...")

        if _port_is_listening(BACKEND_HOST, BACKEND_PORT):
            if _backend_is_ready():
                print(f"Backend already running, reusing existing instance:\n{BACKEND_URL}\n")
                return
            raise LauncherError(
                f"Port {BACKEND_PORT} is already in use by another process that is not "
                "responding as the SyncGuard backend. Free the port or stop that process, "
                "then try again."
            )

        python_exe = _venv_python()
        args = [
            python_exe,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            BACKEND_HOST,
            "--port",
            str(BACKEND_PORT),
        ]
        proc = self._spawn(args, cwd=PROJECT_ROOT)

        def check() -> bool:
            if proc.poll() is not None:
                raise LauncherError(
                    f"Backend process exited early (code {proc.returncode}). "
                    "See the output above for the underlying error."
                )
            return _backend_is_ready()

        if not _wait_until(check, STARTUP_TIMEOUT_SECONDS):
            raise LauncherError(
                f"Backend did not become ready at {BACKEND_HEALTH_URL} within "
                f"{STARTUP_TIMEOUT_SECONDS:.0f}s."
            )
        print(f"Backend ready:\n{BACKEND_URL}\n")

    def start_frontend(self) -> None:
        print("[2/2] Starting frontend...")

        if _port_is_listening(FRONTEND_HOST, FRONTEND_PORT):
            if _frontend_is_ready():
                print(f"Frontend already running, reusing existing instance:\n{FRONTEND_URL}\n")
                return
            raise LauncherError(
                f"Port {FRONTEND_PORT} is already in use by another process that is not "
                "responding as the SyncGuard frontend. Free the port or stop that process, "
                "then try again."
            )

        node_exe = shutil.which("node")
        vite_entry = FRONTEND_DIR / "node_modules" / "vite" / "bin" / "vite.js"
        if not node_exe:
            raise LauncherError("Node.js was not found on PATH. Install Node.js to run the frontend.")
        if not vite_entry.is_file():
            raise LauncherError(
                f"Frontend dependencies are not installed ({vite_entry} is missing). "
                "Run 'npm install' inside the frontend/ directory, then try again."
            )

        args = [node_exe, str(vite_entry), "--host", FRONTEND_HOST, "--port", str(FRONTEND_PORT)]
        proc = self._spawn(args, cwd=FRONTEND_DIR)

        def check() -> bool:
            if proc.poll() is not None:
                raise LauncherError(
                    f"Frontend process exited early (code {proc.returncode}). "
                    "See the output above for the underlying error."
                )
            return _frontend_is_ready()

        if not _wait_until(check, STARTUP_TIMEOUT_SECONDS):
            raise LauncherError(
                f"Frontend did not become reachable at {FRONTEND_URL} within "
                f"{STARTUP_TIMEOUT_SECONDS:.0f}s."
            )
        print(f"Frontend ready:\n{FRONTEND_URL}\n")

    def wait_forever(self) -> None:
        """Block until Ctrl+C, or exit early if a launcher-owned process dies."""
        if not self.processes:
            print("Both services were already running - nothing for this launcher to keep alive.")
            print("Press Ctrl+C to exit.\n")
        while True:
            for proc in self.processes:
                code = proc.poll()
                if code is not None:
                    raise LauncherError(f"A SyncGuard process exited unexpectedly (code {code}).")
            time.sleep(1.0)

    def shutdown(self) -> None:
        if not self.processes:
            return
        print("\nStopping SyncGuard...")
        for proc in self.processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except OSError:
                    pass

        deadline = time.monotonic() + 10.0
        for proc in self.processes:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=5.0)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        print("SyncGuard stopped.")


def main() -> int:
    print(RULE)
    print("SYNCGUARD")
    print("Multimodal Deepfake Detection")
    print(RULE)
    print()

    launcher = SyncGuardLauncher()
    try:
        launcher.start_backend()
        launcher.start_frontend()

        print(THIN_RULE)
        print("SYNCGUARD IS READY")
        print(THIN_RULE)
        print()
        print("Open the application:")
        print()
        print(FRONTEND_URL)
        print()
        print("API:")
        print()
        print(BACKEND_URL)
        print()
        print("Health:")
        print()
        print(BACKEND_HEALTH_URL)
        print()
        print("Press Ctrl+C to stop SyncGuard.")
        print(THIN_RULE)

        try:
            webbrowser.open(FRONTEND_URL)
        except Exception:
            pass  # Browser auto-open is best-effort; the URL above still works.

        launcher.wait_forever()
        return 0
    except LauncherError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    finally:
        launcher.shutdown()


if __name__ == "__main__":
    sys.exit(main())
