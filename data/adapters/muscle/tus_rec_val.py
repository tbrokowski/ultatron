"""
data/adapters/muscle/tus_rec_val.py  ·  TUS-REC Validation adapter
===================================================================

TUS-REC2025 Validation Dataset — "Reconstructing 2D to 3D US (Forearms)"
Li et al., MICCAI 2025 Challenge.

  Validation scans with frames and tforms stored in separate folders.
  Landmark files may live under landmark/ (singular) as landmark_{subj}.h5.

DOI     : https://doi.org/10.5281/zenodo.15699958
SonoDQS : gold
Probe   : curvilinear
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


class TUSRECValAdapter(BaseAdapter):
    """
    Adapter for the TUS-REC2025 validation dataset (forearm freehand 3D US).

    Yields one USManifestEntry per scan (.h5 file in frames/).
    """

    DATASET_ID     = "TUS-REC-Val"
    ANATOMY_FAMILY = "muscle"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.5281/zenodo.15699958"
    DEFAULT_SPLIT_RATIO = (0.0, 1.0, 0.0)

    def iter_entries(self) -> Iterator[USManifestEntry]:
        frames_dir = self.root / "frames"
        transfs_dir = self.root / "transfs"

        if not frames_dir.is_dir():
            frames_dir = self.root / "frames_transfs"
            transfs_dir = frames_dir

        h5_files: list[tuple[Path, str]] = []
        for item in sorted(frames_dir.iterdir()):
            if item.is_dir():
                for h5 in sorted(item.glob("*.h5")):
                    h5_files.append((h5, item.name))
            elif item.suffix == ".h5":
                h5_files.append((item, item.stem))

        for frame_h5, subject_id in h5_files:
            split = self.split_override or "val"
            side, motion, scan_name = parse_h5_stem(frame_h5.stem)
            n_frames = count_h5_frames(frame_h5) or 1500

            tforms_h5 = transfs_dir / subject_id / frame_h5.name
            has_tforms = tforms_h5.exists()

            landmark_path, has_landmarks = resolve_landmark_path(self.root, subject_id)

            entry = self._make_entry(
                str(frame_h5),
                split,
                modality="video",
                instances=[],
                has_mask=False,
                task_type="reconstruction_3d",
                ssl_stream="video",
                is_promptable=False,
                probe_type="curvilinear",
                source_meta={
                    "subject_id": subject_id,
                    "side": side,
                    "motion": motion,
                    "scan_name": scan_name,
                    "tforms_path": str(tforms_h5) if has_tforms else None,
                    "has_tforms": has_tforms,
                    "has_landmarks": has_landmarks,
                    "landmark_path": str(landmark_path) if landmark_path else None,
                    "calib_csv": str(self.root / "calib_matrix.csv"),
                    "doi": self.DOI,
                },
            )
            entry.num_frames = n_frames
            entry.has_temporal_order = True
            yield entry
