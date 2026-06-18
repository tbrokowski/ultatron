"""Shared helpers for TUS-REC train and validation adapters."""
from __future__ import annotations

import re
from pathlib import Path

_FNAME_RE = re.compile(
    r"^(?P<side>RH|LH)_(?P<motion>\w+)$",
    re.IGNORECASE,
)
_ZENODO_RE = re.compile(
    r"^(?P<side>RH|LH)_(?P<protocol>.+)$",
    re.IGNORECASE,
)

_SIDE_MAP = {"rh": "right", "lh": "left"}
_MOTION_MAP = {"rotating": "rotating", "fanning": "fanning", "rocking": "rocking"}


def parse_h5_stem(stem: str) -> tuple[str | None, str | None, str]:
    """
    Return (side, motion_or_protocol, scan_name).

    Handles challenge names (RH_rotating) and Zenodo names (LH_Par_C_DtP).
    """
    m = _FNAME_RE.match(stem)
    if m:
        side = _SIDE_MAP.get(m.group("side").lower())
        motion = _MOTION_MAP.get(m.group("motion").lower(), m.group("motion").lower())
        return side, motion, stem

    m = _ZENODO_RE.match(stem)
    if m:
        side = _SIDE_MAP.get(m.group("side").lower())
        return side, m.group("protocol"), stem

    return None, None, stem


def count_h5_frames(h5_path: Path, prefer_key: str = "frames") -> int:
    """Return number of frames in an HDF5 scan file."""
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            if prefer_key in f:
                return int(f[prefer_key].shape[0])
            for key in ("frames", "tforms"):
                if key in f:
                    return int(f[key].shape[0])
            keys = list(f.keys())
            if keys:
                return int(f[keys[0]].shape[0])
    except Exception:
        pass
    return 0


def resolve_landmark_path(meta_root: Path, subject_id: str) -> tuple[Path | None, bool]:
    """Resolve landmark HDF5 for a subject across naming variants."""
    candidates = [
        meta_root / "landmarks" / f"{subject_id}.h5",
        meta_root / "landmark" / f"landmark_{subject_id}.h5",
        meta_root / "landmark" / f"{subject_id}.h5",
    ]
    for path in candidates:
        if path.exists():
            return path, True
    return None, False
