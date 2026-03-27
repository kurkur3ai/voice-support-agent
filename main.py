import subprocess
import sys
import time
import socket
import webbrowser
import threading
import os
import argparse
import shutil
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))

OLLAMA_MODEL = "llama3.2"
OLLAMA_URL   = "http://localhost:11434"
OLLAMA_EXE   = os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe")


# ── Ollama helpers ────────────────────────────────────────────────────────────

def _ollama_running() -> bool:
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=2)
        return True
    except Exception:
        return False


def ensure_ollama():
    # 1. Check installed
    exe = shutil.which("ollama") or (OLLAMA_EXE if os.path.exists(OLLAMA_EXE) else None)
    if not exe:
        print("\n[Ollama] Not found. Please install Ollama from https://ollama.com/download")
        print("[Ollama] After installing, re-run main.py.")
        sys.exit(1)

    # 2. Start if not running
    if not _ollama_running():
        print("[Ollama] Starting ollama serve...")
        subprocess.Popen(
            [exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(15):
            time.sleep(1)
            if _ollama_running():
                break
        else:
            raise RuntimeError("Ollama did not start in time.")
        print("[Ollama] Running.")
    else:
        print("[Ollama] Already running.")

    # 3. Pull model if not present
    result = subprocess.run([exe, "list"], capture_output=True, text=True)
    if OLLAMA_MODEL not in result.stdout:
        print(f"[Ollama] Pulling model {OLLAMA_MODEL} (first time only)...")
        subprocess.run([exe, "pull", OLLAMA_MODEL], check=True)
        print(f"[Ollama] Model {OLLAMA_MODEL} ready.")
    else:
        print(f"[Ollama] Model {OLLAMA_MODEL} already present.")


def _wait_for_backend(host: str = "127.0.0.1", port: int = 8000, timeout: float = 120.0) -> bool:
    """Poll until the backend TCP port accepts connections or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def run_backend(keep: bool):
    env = os.environ.copy()
    env["KEEP_RECORDINGS"] = "1" if keep else "0"
    subprocess.run(
        [sys.executable, "-m", "uvicorn", "backend.app:app", "--port", "8000"],
        cwd=BASE, env=env
    )


def run_frontend():
    subprocess.run(
        [sys.executable, "-m", "http.server", "3000"],
        cwd=os.path.join(BASE, "frontend")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Agent launcher")
    parser.add_argument(
        "--delete-recordings", action="store_true",
        help="Delete audio files after transcription (default: keep them in recordings/)"
    )
    args = parser.parse_args()
    keep = not args.delete_recordings

    ensure_ollama()

    recordings_dir = os.path.join(BASE, "recordings")
    if keep:
        print(f"[Config] Recordings saved to: {recordings_dir}")
    else:
        print("[Config] Recordings will be deleted after transcription.")

    threading.Thread(target=run_backend, args=(keep,), daemon=True).start()
    threading.Thread(target=run_frontend, daemon=True).start()

    print("[Setup] Waiting for backend to be ready (STT + embeddings + TTS warm-up)...")
    if _wait_for_backend():
        print("[Setup] Backend ready.")
    else:
        print("[Setup] Warning: backend did not become ready in time.")
    webbrowser.open("http://localhost:3000")

    print("Running. Press Ctrl+C to stop.")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print("\nStopped.")
