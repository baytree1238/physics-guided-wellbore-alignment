#!/usr/bin/env python3
"""Create the pinned environment and hand off to the requested command."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def find_uv() -> str | None:
    """Find uv on PATH or in the repository-local tool directory."""

    candidates = [shutil.which("uv"), str(ROOT / ".uv-bin" / "uv")]
    return next((value for value in candidates if value and Path(value).exists()), None)


def ensure_environment() -> None:
    """Create the pinned environment when its dependency hash is stale."""

    marker = VENV / ".requirements.sha256"
    import hashlib

    expected = hashlib.sha256(
        (ROOT / "requirements.txt").read_bytes() + (ROOT / "pyproject.toml").read_bytes()
    ).hexdigest()
    if PYTHON.exists() and marker.exists() and marker.read_text().strip() == expected:
        return
    uv = find_uv()
    if uv:
        uv_cache = Path(tempfile.gettempdir()) / "rogii-portfolio-uv-cache"
        uv_cache.mkdir(parents=True, exist_ok=True)
        uv_env = {**os.environ, "UV_CACHE_DIR": str(uv_cache)}
        if not PYTHON.exists():
            subprocess.run(
                [uv, "venv", str(VENV), "--python", sys.executable],
                check=True,
                env=uv_env,
            )
        subprocess.run(
            [uv, "pip", "install", "--python", str(PYTHON), "-r", str(ROOT / "requirements.txt")],
            check=True,
            env=uv_env,
        )
    else:
        if not PYTHON.exists():
            subprocess.run([sys.executable, "-m", "venv", str(VENV)], check=True)
        subprocess.run([str(PYTHON), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")], check=True)
    marker.write_text(expected + "\n", encoding="utf-8")


def main() -> int:
    """Prepare the environment and run the requested project command."""

    ensure_environment()
    command = sys.argv[1:] or [str(ROOT / "scripts" / "run_reproduce.py")]
    runtime = Path(tempfile.gettempdir()) / "rogii-portfolio-runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    child_env = {
        **os.environ,
        "MPLCONFIGDIR": str(runtime / "matplotlib"),
        "IPYTHONDIR": str(runtime / "ipython"),
        "JUPYTER_RUNTIME_DIR": str(runtime / "jupyter"),
    }
    for directory in (runtime / "matplotlib", runtime / "ipython", runtime / "jupyter"):
        directory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run([str(PYTHON), *command], cwd=ROOT, env=child_env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
