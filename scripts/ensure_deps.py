#!/usr/bin/env python3
"""Ensure Ultatron runtime dependencies are installed for analysis scripts.

Jobs and one-off scripts often set PYTHONPATH without installing the package,
so declared deps (pydicom, h5py, etc.) are missing until pip-installed ad hoc.
Only the missing I/O packages are installed — not a full editable install, so
preloaded container pins (numpy, cupy, ultr-ai, etc.) are left untouched.
"""
from __future__ import annotations

import fcntl
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Imports that must work for DICOM/HDF5/volume I/O across analysis + training.
_RUNTIME_PACKAGES: dict[str, str] = {
    "pydicom": "pydicom>=2.4.0",
    "h5py": "h5py>=3.11.0,<3.13",
    "SimpleITK": "SimpleITK>=2.3.0",
}

# Optional on GH200 (aarch64) — video decode fallback in dataset.load_video_frames.
if sys.platform != "win32":
    import platform
    if platform.machine() in ("aarch64", "arm64"):
        _RUNTIME_PACKAGES["cv2"] = "opencv-python-headless>=4.8.0,<4.12"


def missing_imports() -> list[str]:
    missing: list[str] = []
    for name in _RUNTIME_PACKAGES:
        try:
            __import__(name)
        except ImportError:
            missing.append(name)
    return missing


def check_torch_numpy() -> None:
    """Fail fast when numpy was upgraded past what container PyTorch supports."""
    try:
        import numpy as np
        import torch
    except ImportError:
        return
    try:
        torch.from_numpy(np.zeros(1, dtype=np.float32))
    except RuntimeError as exc:
        if "numpy is not available" not in str(exc).lower():
            raise
        raise RuntimeError(
            f"PyTorch cannot use the installed numpy ({np.__version__}). "
            "This usually means an earlier `pip install -e .` upgraded numpy "
            "in the CSCS ultr-ai container. Repair with:\n"
            "  pip install 'numpy>=1.24,<1.27' 'h5py>=3.11,<3.13'\n"
            "Then re-run the script."
        ) from exc


def _pip_install(specs: list[str], *, attempts: int = 3) -> None:
    """Install packages with retries and a repo lock for multi-node Slurm jobs."""
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-warn-conflicts",
        "--prefer-binary",
        *specs,
    ]
    lock_path = REPO_ROOT / ".ensure_deps.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        last_exc: subprocess.CalledProcessError | None = None
        for attempt in range(1, attempts + 1):
            try:
                subprocess.check_call(cmd)
                return
            except subprocess.CalledProcessError as exc:
                last_exc = exc
                if attempt < attempts:
                    delay = 5 * attempt
                    print(
                        f"[ensure_deps] pip install failed (attempt {attempt}/{attempts}); "
                        f"retrying in {delay}s …",
                        flush=True,
                    )
                    time.sleep(delay)
        assert last_exc is not None
        raise RuntimeError(
            "pip could not reach PyPI to install: "
            + ", ".join(specs)
            + "\nThis often happens when compute nodes cannot reach pypi.org "
            "(Content-Type: Unknown / no matching distribution). "
            "Re-submit the job, or from an interactive srun session run:\n"
            f"  python3 -m pip install --prefer-binary {' '.join(specs)}"
        ) from last_exc


def ensure_deps(*, force: bool = False) -> None:
    """Install only missing runtime I/O packages (never the full project)."""
    missing = list(_RUNTIME_PACKAGES) if force else missing_imports()
    if missing:
        specs = [_RUNTIME_PACKAGES[name] for name in missing]
        print(
            f"[ensure_deps] Installing missing packages: {', '.join(specs)}",
            flush=True,
        )
        _pip_install(specs)
        still_missing = missing_imports()
        if still_missing:
            raise RuntimeError(
                "Failed to install required packages: "
                + ", ".join(still_missing)
                + f". Try manually: pip install {' '.join(_RUNTIME_PACKAGES[n] for n in still_missing)}"
            )
    check_torch_numpy()


if __name__ == "__main__":
    ensure_deps(force="--force" in sys.argv)
