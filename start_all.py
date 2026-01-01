#!/usr/bin/env python3
"""
start_all.py — Cross-platform UniBuddy Launcher

Starts both backend and frontend in parallel:
- Backend: python backend/run.py (on port 9000)
- Frontend: npm run dev (Vite dev server)

Features:
- Automatic npm install if node_modules is missing
- Streams logs with [BACKEND] / [FRONTEND] prefixes
- Clean shutdown of both processes on Ctrl+C
- Works on Windows, macOS, and Linux
"""

import os
import sys
import subprocess
import threading
import shutil
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT  # Frontend is in project root (Vite app)

PY_CMD = sys.executable
NPM_CMD = shutil.which("npm") or "npm"


def stream_output(prefix: str, proc: subprocess.Popen):
    """Stream process output with prefix. Use robust decoding on Windows."""
    # read binary to avoid encoding errors, decode per-line with replacement
    with proc.stdout:
        for raw in proc.stdout:
            try:
                # proc.stdout is text-mode when encoding provided; ensure it's str
                line = raw.rstrip('\n')
            except Exception:
                try:
                    line = raw.decode('utf-8', errors='replace').rstrip('\n')
                except Exception:
                    line = '<unreadable line>'
            print(f"{prefix} {line}")


def ensure_frontend_deps():
    """Run npm install if node_modules doesn't exist"""
    node_modules = FRONTEND_DIR / "node_modules"
    if not node_modules.exists():
        print("[LAUNCHER] node_modules not found — running 'npm install'...")
        result = subprocess.run([NPM_CMD, "install"], cwd=str(FRONTEND_DIR))
        if result.returncode != 0:
            print("[LAUNCHER] ERROR: npm install failed!")
            sys.exit(1)
        print("[LAUNCHER] npm install completed.")


def start_backend():
    return subprocess.Popen(
        [PY_CMD, "run.py"],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding='utf-8',
        errors='replace'
    )


def start_frontend():
    return subprocess.Popen(
        [NPM_CMD, "run", "dev"],
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
        encoding='utf-8',
        errors='replace'
    )


def main():
    # Basic validation
    if not BACKEND_DIR.exists() or not (BACKEND_DIR / "run.py").exists():
        print(f"[LAUNCHER] ERROR: Backend not found at {BACKEND_DIR}")
        sys.exit(1)

    if shutil.which("npm") is None:
        print("[LAUNCHER] ERROR: npm not found in PATH. Please install Node.js.")
        sys.exit(1)

    ensure_frontend_deps()

    print("[LAUNCHER] Starting UniBuddy — backend + frontend...\n")

    backend_proc = start_backend()
    frontend_proc = start_frontend()

    # Stream logs in background threads
    threading.Thread(target=stream_output, args=("[BACKEND]", backend_proc), daemon=True).start()
    threading.Thread(target=stream_output, args=("[FRONTEND]", frontend_proc), daemon=True).start()

    def shutdown(signum=None, frame=None):
        print("\n[LAUNCHER] Shutting down UniBuddy...")
        for proc, name in [(frontend_proc, "Frontend"), (backend_proc, "Backend")]:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                print(f"[LAUNCHER] {name} stopped.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, shutdown)

    # Keep main thread alive and monitor
    try:
        while True:
            time.sleep(0.5)
            if backend_proc.poll() is not None:
                print(f"[LAUNCHER] Backend exited. Shutting down frontend...")
                shutdown()
            if frontend_proc.poll() is not None:
                print(f"[LAUNCHER] Frontend exited. Shutting down backend...")
                shutdown()
    except KeyboardInterrupt:
        shutdown()

if __name__ == "__main__":
    main()
