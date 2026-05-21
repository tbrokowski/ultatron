"""
data/adapters/muscle/tus_rec.py  ·  TUS-REC adapter
=====================================================

TUS-REC — "Trackerless 3D Freehand Ultrasound Reconstruction Challenge 2025"
Li et al., MICCAI 2025 Challenge.

  100 freehand US scans of arm/forearm (50 subjects × 2 scans each).
  ~1,600 frames per scan @ 20 fps, image size 480×640.
  Probe: curvilinear (4DC7-3/40), 6 MHz, depth 9 cm.
  Sides: left (LH) and right (RH) forearm.
  Motion types: rotating, fanning, rocking.

Task: trackerless 3D US reconstruction — predict frame-to-frame
      spatial transformations without an external tracker.
      → modality_type = "video", task_type = "reconstruction_3d"
      → no segmentation masks; transformation ground truth in .h5 files.

DOI     : https://doi.org/10.5281/zenodo.15224704
SonoDQS : gold (tracker-ground-truth, standardized protocol, multi-subject)
Probe   : curvilinear

Dataset layout
--------------
  {root}/
    frames_transfs/
      subject_001/
        RH_rotating.h5      # frames [N,H,W] + tforms [N,4,4]
        LH_rotating.h5
      subject_002/
        ...
    landmarks/
      subject_001.h5        # [scan, 100, 3] landmark coords (optional)
    calib_matrix.csv        # pixel-to-mm scale + spatial calibration

Each .h5 file = one scan = one USManifestEntry (video modality).
Frame-level entries can be optionally requested via `frame_level=True`.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

# Regex to parse side and motion type from filename
# e.g. "RH_rotating" → side=RH, motion=rotating
_FNAME_RE = re.compile(
    r"^(?P<side>RH|LH)_(?P<motion>\w+)$",
    re.IGNORECASE,
)

_MOTION_MAP = {
    "rotating": "rotating",
    "fanning":  "fanning",
    "rocking":  "rocking",
}

_SIDE_MAP = {
    "rh": "right",
    "lh": "left",
}


def _parse_h5_stem(stem: str) -> tuple[str | None, str | None]:
    """Return (side, motion) from h5 filename stem."""
    m = _FNAME_RE.match(stem)
    if m:
        side   = _SIDE_MAP.get(m.group("side").lower())
        motion = _MOTION_MAP.get(m.group("motion").lower(), m.group("motion").lower())
        return side, motion
    return None, None


def _count_frames(h5_path: Path) -> int:
    """Return number of frames in .h5 file if h5py is available, else 0."""
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            return int(f["frames"].shape[0])
    except Exception:
        return 0


class TUSRECAdapter(BaseAdapter):
    """
    Adapter for the TUS-REC freehand 3D ultrasound reconstruction dataset.

    Yields one USManifestEntry per scan (.h5 file).
    Each entry represents a full US sequence (video modality) with:
      - image_paths = [path_to_h5]
      - task_type   = "reconstruction_3d"
      - source_meta with subject_id, side, motion, num_frames

    Parameters
    ----------
    root : str | Path
        Root directory containing frames_transfs/, landmarks/, calib_matrix.csv.
    split_override : str, optional
        Force all entries to a single split.
    """

    DATASET_ID     = "TUS-REC"
    ANATOMY_FAMILY = "muscle"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.5281/zenodo.15224704"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        frames_dir = self.root / "frames_transfs"

        if not frames_dir.is_dir():
            # Fallback: h5 files directly at root
            frames_dir = self.root

        # Collect all .h5 scan files
        h5_files: list[tuple[Path, str]] = []  # (path, subject_id)
        for item in sorted(frames_dir.iterdir()):
            if item.is_dir():
                subject_id = item.name
                for h5 in sorted(item.glob("*.h5")):
                    h5_files.append((h5, subject_id))
            elif item.suffix == ".h5":
                h5_files.append((item, item.stem))

        n = len(h5_files)

        for i, (h5_path, subject_id) in enumerate(h5_files):
            split      = self._infer_split(subject_id + h5_path.stem, i, n)
            side, motion = _parse_h5_stem(h5_path.stem)
            num_frames = _count_frames(h5_path)

            # Check if landmark file exists for this subject
            landmark_path = self.root / "landmarks" / f"{subject_id}.h5"
            has_landmarks = landmark_path.exists()

            yield self._make_entry(
                str(h5_path),
                split,
                modality        = "video",
                instances       = [],
                has_mask        = False,
                task_type       = "reconstruction_3d",
                ssl_stream      = "video",
                is_promptable   = False,
                probe_type      = "curvilinear",
                has_temporal_order = True,
                num_frames = num_frames if num_frames > 0 else 1600,  # ~1600 frames per scan
                source_meta     = {
                    "subject_id":    subject_id,
                    "side":          side,        # "left" | "right" | None
                    "motion":        motion,      # "rotating" | "fanning" | "rocking"
                    "has_landmarks": has_landmarks,
                    "calib_csv":     str(self.root / "calib_matrix.csv"),
                    "doi":           self.DOI,
                },
            )
