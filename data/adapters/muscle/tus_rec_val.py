"""
data/adapters/muscle/tus_rec_val.py  ·  TUS-REC Validation adapter
===================================================================

TUS-REC2025 Validation Dataset — "Reconstructing 2D to 3D US (Forearms)"
Li et al., MICCAI 2025 Challenge.

  6 freehand US scans (3 subjects × 2 scans each, LH + RH forearm).
  ~1,500 frames per scan @ 20 fps, image size 480×640.
  Same probe/protocol as TUS-REC train set.

Key structural difference from the train set:
  Train: frames + tforms stored TOGETHER in frames_transfs/{subj}/*.h5
  Val  : frames stored in frames/{subj}/*.h5
         tforms stored in transfs/{subj}/*.h5   (separate folder)
         landmarks in landmarks/{subj}.h5

DOI     : https://doi.org/10.5281/zenodo.15699958
SonoDQS : gold
Probe   : curvilinear

Dataset layout
--------------
  {root}/
    frames/
      subject_001/
        RH_rotating.h5    # frames only: [N,H,W]
        LH_rotating.h5
      subject_002/
        ...
    transfs/              # transformations only: [N,4,4]
      subject_001/
        RH_rotating.h5
        LH_rotating.h5
      ...
    landmarks/
      subject_001.h5
      ...
    calib_matrix.csv
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_FNAME_RE = re.compile(r"^(?P<side>RH|LH)_(?P<motion>\w+)$", re.IGNORECASE)
_SIDE_MAP   = {"rh": "right", "lh": "left"}
_MOTION_MAP = {"rotating": "rotating", "fanning": "fanning", "rocking": "rocking"}


def _parse_h5_stem(stem: str) -> tuple[str | None, str | None]:
    m = _FNAME_RE.match(stem)
    if m:
        return (
            _SIDE_MAP.get(m.group("side").lower()),
            _MOTION_MAP.get(m.group("motion").lower(), m.group("motion").lower()),
        )
    return None, None


def _count_frames(h5_path: Path) -> int:
    try:
        import h5py
        with h5py.File(h5_path, "r") as f:
            key = "frames" if "frames" in f else list(f.keys())[0]
            return int(f[key].shape[0])
    except Exception:
        return 0


class TUSRECValAdapter(BaseAdapter):
    """
    Adapter for the TUS-REC2025 validation dataset (forearm freehand 3D US).

    Yields one USManifestEntry per scan (.h5 file in frames/).
    The corresponding tforms .h5 path is stored in source_meta["tforms_path"].

    Parameters
    ----------
    root : str | Path
        Root directory containing frames/, transfs/, landmarks/, calib_matrix.csv.
    split_override : str, optional
        Force all entries to a single split (default: all → "val").
    """

    DATASET_ID     = "TUS-REC-Val"
    ANATOMY_FAMILY = "muscle"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.5281/zenodo.15699958"
    DEFAULT_SPLIT_RATIO = (0.0, 1.0, 0.0)   # everything is val by default

    def iter_entries(self) -> Iterator[USManifestEntry]:
        frames_dir = self.root / "frames"
        transfs_dir = self.root / "transfs"

        # Fallback: if no frames/ dir, try frames_transfs/ (same as train)
        if not frames_dir.is_dir():
            frames_dir  = self.root / "frames_transfs"
            transfs_dir = frames_dir

        h5_files: list[tuple[Path, str]] = []
        for item in sorted(frames_dir.iterdir()):
            if item.is_dir():
                for h5 in sorted(item.glob("*.h5")):
                    h5_files.append((h5, item.name))
            elif item.suffix == ".h5":
                h5_files.append((item, item.stem))

        n = len(h5_files)

        for i, (frame_h5, subject_id) in enumerate(h5_files):
            split        = self.split_override or "val"
            side, motion = _parse_h5_stem(frame_h5.stem)
            n_frames     = _count_frames(frame_h5) or 1500

            # Corresponding tforms file
            tforms_h5 = transfs_dir / subject_id / frame_h5.name
            has_tforms = tforms_h5.exists()

            # Landmark file
            landmark_path = self.root / "landmarks" / f"{subject_id}.h5"
            has_landmarks = landmark_path.exists()

            entry = self._make_entry(
                str(frame_h5),
                split,
                modality      = "video",
                instances     = [],
                has_mask      = False,
                task_type     = "reconstruction_3d",
                ssl_stream    = "video",
                is_promptable = False,
                probe_type    = "curvilinear",
                source_meta   = {
                    "subject_id":    subject_id,
                    "side":          side,
                    "motion":        motion,
                    "tforms_path":   str(tforms_h5) if has_tforms else None,
                    "has_tforms":    has_tforms,
                    "has_landmarks": has_landmarks,
                    "calib_csv":     str(self.root / "calib_matrix.csv"),
                    "doi":           self.DOI,
                },
            )
            entry.num_frames         = n_frames
            entry.has_temporal_order = True
            yield entry
