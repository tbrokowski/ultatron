"""
data/adapters/muscle/tus_rec.py  ·  TUS-REC adapter
=====================================================

TUS-REC — "Trackerless 3D Freehand Ultrasound Reconstruction Challenge 2025"
Li et al., MICCAI 2025 Challenge.

  Two related releases share this adapter:

  TUS-REC2025 (Zenodo 15224704): 100 scans — 50 subjects × 2 rotating scans.
  ~1,600 frames per scan @ 20 fps, image size 480×640.

  TUS-REC2024 supplementary (Zenodo 11178508 + 11180794): 1,200 scans —
  50 subjects × 24 scans (Par/Per × L/C/S × DtP/PtD per forearm).
  ~200–850 frames per scan.
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
  Challenge release (frames_transfs/):
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

  Zenodo 2024 download (train_part1/ + train_part2/):
  {root}/
    train_part1/            # subjects 000–024 (600 scans)
      000/
        LH_Par_C_DtP.h5     # 24 scans per subject, numeric subject IDs
        RH_Per_S_PtD.h5
        ...
    train_part2/            # subjects 025–049 (600 scans)
      025/
        ...
    landmarks/
      landmark_000.h5
    calib_matrix.csv

Each .h5 file = one scan = one USManifestEntry (video modality).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.adapters.muscle.tus_rec_common import (
    count_h5_frames,
    parse_h5_stem,
    resolve_landmark_path,
)
from data.schema.manifest import USManifestEntry


def _resolve_scans_dirs(root: Path) -> list[Path]:
    """Return directories containing per-subject scan subfolders."""
    dirs: list[Path] = []
    frames = root / "frames_transfs"
    if frames.is_dir():
        dirs.append(frames)
    for name in ("train_part1", "train_part2", "train_part3"):
        candidate = root / name
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs or [root]


def _resolve_dataset_meta_root(scans_dirs: list[Path], dataset_root: Path) -> Path:
    """Return root for calib_matrix.csv and landmarks/ (may differ from scans dirs)."""
    if any(d != dataset_root for d in scans_dirs):
        return dataset_root
    return scans_dirs[0]


class TUSRECAdapter(BaseAdapter):
    """
    Adapter for the TUS-REC freehand 3D ultrasound reconstruction dataset.

    Yields one USManifestEntry per scan (.h5 file).
    """

    DATASET_ID     = "TUS-REC"
    ANATOMY_FAMILY = "muscle"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.5281/zenodo.15224704"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        scans_dirs = _resolve_scans_dirs(self.root)
        meta_root  = _resolve_dataset_meta_root(scans_dirs, self.root)
        calib_path = meta_root / "calib_matrix.csv"

        h5_files: list[tuple[Path, str]] = []
        for scans_dir in scans_dirs:
            for item in sorted(scans_dir.iterdir()):
                if item.is_dir():
                    subject_id = item.name
                    for h5 in sorted(item.glob("*.h5")):
                        h5_files.append((h5, subject_id))
                elif item.suffix == ".h5":
                    h5_files.append((item, item.stem))

        n = len(h5_files)

        for i, (h5_path, subject_id) in enumerate(h5_files):
            split = self._infer_split(subject_id + h5_path.stem, i, n)
            side, motion, scan_name = parse_h5_stem(h5_path.stem)
            num_frames = count_h5_frames(h5_path)

            landmark_path, has_landmarks = resolve_landmark_path(meta_root, subject_id)

            yield self._make_entry(
                str(h5_path),
                split,
                modality="video",
                instances=[],
                has_mask=False,
                task_type="reconstruction_3d",
                ssl_stream="video",
                is_promptable=False,
                probe_type="curvilinear",
                has_temporal_order=True,
                num_frames=num_frames if num_frames > 0 else 1600,
                source_meta={
                    "subject_id": subject_id,
                    "side": side,
                    "motion": motion,
                    "scan_name": scan_name,
                    "has_landmarks": has_landmarks,
                    "landmark_path": str(landmark_path) if landmark_path else None,
                    "calib_csv": str(calib_path),
                    "doi": self.DOI,
                },
            )
