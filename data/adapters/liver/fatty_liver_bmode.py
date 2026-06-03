"""
data/adapters/liver/fatty_liver_bmode.py  ·  Fatty-Liver B-mode adapter
========================================================================

Byra et al., 2018: "Liver fat assessment in multimodal B-mode ultrasound"
  55 patients, binary fatty-liver labels (class 0=normal, 1=fatty),
  biopsy-confirmed steatosis percentage, 10 B-mode frames per patient.

Dataset: single MATLAB .mat file
  data/adapters/liver/../../../...IJCARS.mat
  Structure: data[0, i] → {id, class, fat, images (10×434×636)}

Lazy extraction:
  On the first call to iter_entries() the adapter checks whether a sibling
  images/ directory already contains all expected PNG files.  If not, it
  reads the .mat file once (requires scipy) and saves each frame as an
  8-bit grayscale PNG.  Subsequent runs skip extraction entirely.

  PNG naming: patient_{id:03d}_frame_{idx:02d}.png
  (e.g. patient_001_frame_00.png … patient_001_frame_09.png)

Split strategy:
  Deterministic 80/10/10 by sorted patient ID (patient-level, not frame-
  level) to prevent data leakage between frames of the same patient.

Entries emitted — one per frame (10 per patient × 55 patients = 550):
  modality_type      = "image"
  anatomy_family     = "liver"
  task_type          = "classification"
  has_mask           = False
  has_temporal_order = False
  num_frames         = 1
  study_id           = "patient_{id:03d}"  (groups all 10 frames)
  series_id          = "patient_{id:03d}"
  instance           = classification instance
    label_raw        = "fatty_liver" | "normal_liver"
    label_ontology   = "liver_steatosis_class"
    classification_label = 0 | 1
  source_meta        = {patient_id, frame_idx, fat_pct, classification_label}
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


_MAT_FILENAME = "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"


class FattyLiverBmodeAdapter(BaseAdapter):
    """
    Fatty-Liver B-mode adapter.

    Lazily extracts all frames from the single .mat file to a sibling
    images/ directory on first use, then yields one image entry per frame.
    """

    DATASET_ID     = "fatty-liver-bmode"
    ANATOMY_FAMILY = "liver"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.5281/zenodo.1009146"

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        for candidate in (root, root / "archive"):
            if (candidate / _MAT_FILENAME).exists():
                return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: {_MAT_FILENAME!r} not found under {root}"
        )

    # ── Public entry point ────────────────────────────────────────────────────

    def iter_entries(self) -> Iterator[USManifestEntry]:
        import scipy.io

        mat_path   = self.root / _MAT_FILENAME
        images_dir = self.root / "images"

        mat  = scipy.io.loadmat(str(mat_path))
        data = mat["data"]            # shape (1, N_patients)

        self._ensure_extracted(data, images_dir)

        records = self._parse_records(data)
        records.sort(key=lambda r: r[0])   # sort by patient_id for reproducibility
        n = len(records)

        for i, (patient_id, cls, fat_pct, n_frames) in enumerate(records):
            split = self.split_override or self._infer_split(
                str(patient_id), i, n
            )

            for frame_idx in range(n_frames):
                png_path = images_dir / f"patient_{patient_id:03d}_frame_{frame_idx:02d}.png"
                if not png_path.exists():
                    continue

                instance = self._make_instance(
                    instance_id          = f"p{patient_id:03d}_f{frame_idx:02d}",
                    label_raw            = "fatty_liver" if cls == 1 else "normal_liver",
                    label_ontology       = "liver_steatosis_class",
                    is_promptable        = False,
                    classification_label = cls,
                )

                yield self._make_entry(
                    str(png_path),
                    split         = split,
                    modality      = "image",
                    instances     = [instance],
                    study_id      = f"patient_{patient_id:03d}",
                    series_id     = f"patient_{patient_id:03d}",
                    view_type     = "liver_bmode",
                    num_frames    = 1,
                    has_mask      = False,
                    has_temporal_order = False,
                    task_type     = "classification",
                    ssl_stream    = "image",
                    is_promptable = False,
                    source_meta   = {
                        "patient_id":           patient_id,
                        "frame_idx":            frame_idx,
                        "fat_pct":              fat_pct,
                        "classification_label": cls,
                    },
                )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _parse_records(
        data: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """Return [(patient_id, cls, fat_pct, n_frames), …] from mat data."""
        records = []
        for i in range(data.shape[1]):
            p = data[0, i]
            patient_id = int(p["id"].flat[0])
            cls        = int(p["class"].flat[0])
            fat_pct    = int(p["fat"].flat[0])
            images     = p["images"]
            n_frames   = images.shape[0] if images.ndim >= 3 else 1
            records.append((patient_id, cls, fat_pct, n_frames))
        return records

    @staticmethod
    def _ensure_extracted(data: np.ndarray, images_dir: Path) -> None:
        """
        Extract every frame from the loaded mat struct to grayscale PNGs.

        Skips extraction if the images/ directory already contains at least
        as many PNGs as there are frames in the .mat file.  Individual files
        that already exist are not overwritten.
        """
        from PIL import Image

        n_patients = data.shape[1]
        total_frames = sum(
            (data[0, i]["images"].shape[0] if data[0, i]["images"].ndim >= 3 else 1)
            for i in range(n_patients)
        )

        existing = sum(1 for _ in images_dir.glob("patient_*.png")) if images_dir.exists() else 0
        if existing >= total_frames:
            return

        images_dir.mkdir(exist_ok=True)

        for i in range(n_patients):
            p          = data[0, i]
            patient_id = int(p["id"].flat[0])
            raw        = p["images"]

            if raw.ndim == 2:
                raw = raw[np.newaxis]   # single frame stored as (H, W)

            for frame_idx in range(raw.shape[0]):
                png_path = images_dir / f"patient_{patient_id:03d}_frame_{frame_idx:02d}.png"
                if png_path.exists():
                    continue

                frame = raw[frame_idx].astype(float)
                lo, hi = frame.min(), frame.max()
                if hi > lo:
                    frame = ((frame - lo) / (hi - lo) * 255)
                frame = frame.clip(0, 255).astype(np.uint8)
                Image.fromarray(frame, mode="L").save(str(png_path))
