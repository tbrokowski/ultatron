"""
data/adapters/maternal_fetal/fetal_abdominal_structures.py  ·  FASS adapter
===========================================================================

Fetal Abdominal Structures Segmentation dataset (FASS): 1,588 fetal-abdomen
ultrasound images paired with multi-structure segmentation annotations stored
as NumPy dicts.

Layout on disk:
  {root}/
  └── Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images/
      ├── IMAGES/
      │   └── P<id>_IMG<N>.png
      └── ARRAY_FORMAT/
          └── P<id>_IMG<N>.npy   ← pickled dict {image, structures: {artery, liver, stomach, vein}}

NPY dict schema:
  {
    "image":      np.ndarray (768, 1024, 3) uint8  — RGB for most files
                            (768, 1024)     uint8  — grayscale for ~33 files
                            (patients P01, P03, P0149, P0162–P0166, P04)
    "structures": {
        "artery":  np.ndarray (768, 1024) uint8 binary,
        "liver":   np.ndarray (768, 1024) uint8 binary,
        "stomach": np.ndarray (768, 1024) uint8 binary,
        "vein":    np.ndarray (768, 1024) uint8 binary,
    }
  }
  Load with: np.load(path, allow_pickle=True).item()

Image modality note:
  1,507 files are RGB PNGs; 81 are 8-bit grayscale (mode L) across 10 patients.
  Grayscale NPY dicts store image as (768, 1024) while RGB store (768, 1024, 3).
  Images are always loaded from IMAGES/*.png (not the embedded NPY copy, which can
  differ from the PNG).  The adapter probes PNG color mode via PIL and stores
  is_grayscale in source_meta.  Grayscale→RGB conversion is the pipeline's job.

Dataset stats (verified on capstor store, Mar 2026):
  1,588 PNG/NPY pairs (100% aligned by stem), 169 patients, 3–17 images/patient.
  All masks are binary uint8 (768, 1024) for artery, liver, stomach, vein.

One Instance per structure is emitted, with mask_channel encoding the
structure index (0=artery, 1=liver, 2=stomach, 3=vein).  All four instances
share the same mask_path pointing to the NPY file.

Split strategy: group by patient_id (extracted from P<id>_IMG<N> stem) to
avoid leakage across images of the same patient.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from PIL import Image

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_STEM_RE = re.compile(r"^P(\d+)_IMG\d+$")

# Canonical NPY structure key order; mask_channel indexes into this tuple.
MASK_STRUCTURE_KEYS: Tuple[str, ...] = ("artery", "liver", "stomach", "vein")

# Fixed structure order; mask_channel is the index used by load_mask downstream.
STRUCTURES: List[Tuple[str, str, int]] = [
    ("artery",  "fetal_abdominal_artery", 0),
    ("liver",   "fetal_liver",            1),
    ("stomach", "fetal_stomach",          2),
    ("vein",    "fetal_abdominal_vein",   3),
]

IMAGE_HEIGHT = 768
IMAGE_WIDTH  = 1024


class FASSAdapter(BaseAdapter):
    DATASET_ID     = "FASS"
    ANATOMY_FAMILY = "fetal_abdomen"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        # Accept root already containing IMAGES/
        if (root / "IMAGES").is_dir():
            return root
        # Descend one level — the dataset ships inside a long-named subdirectory
        if root.is_dir():
            for candidate in sorted(root.iterdir()):
                if candidate.is_dir() and (candidate / "IMAGES").is_dir():
                    return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'IMAGES/' under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir  = self.root / "IMAGES"
        mask_dir = self.root / "ARRAY_FORMAT"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"FASS: image directory not found at {img_dir}"
            )

        # Index NPY files by stem — folder ordering is not aligned with IMAGES/.
        npy_index: Dict[str, Path] = (
            {p.stem: p for p in mask_dir.glob("*.npy")}
            if mask_dir.exists() else {}
        )

        images = sorted(img_dir.glob("*.png"))

        # Patient-level split map to prevent leakage across same-patient images.
        patient_ids = [self._patient_id(p.stem) for p in images]
        patients = sorted(set(patient_ids))
        patient_split: Dict[str, str] = {
            pid: self._infer_split(pid, i, len(patients))
            for i, pid in enumerate(patients)
        }

        for img_path in images:
            stem       = img_path.stem
            patient_id = self._patient_id(stem)
            npy_path   = npy_index.get(stem)
            has_mask   = npy_path is not None
            split      = self.split_override or patient_split.get(patient_id, "train")
            is_gray    = self._is_grayscale(img_path)

            if has_mask:
                instances = [
                    self._make_instance(
                        instance_id    = f"{stem}_{struct_name}",
                        label_raw      = struct_name,
                        label_ontology = label_ontology,
                        mask_path      = str(npy_path),
                        mask_channel   = channel,
                        is_promptable  = True,
                    )
                    for struct_name, label_ontology, channel in STRUCTURES
                ]
                task_type = "segmentation"
            else:
                instances = []
                task_type = "ssl_only"

            yield self._make_entry(
                str(img_path),
                split         = split,
                modality      = "image",
                instances     = instances,
                study_id      = patient_id,
                series_id     = stem,
                view_type     = "fetal_abdomen_bmode",
                height        = IMAGE_HEIGHT,
                width         = IMAGE_WIDTH,
                has_mask      = has_mask,
                task_type     = task_type,
                ssl_stream    = "image",
                is_promptable = has_mask,
                source_meta   = {
                    "patient_id":   patient_id,
                    "stem":         stem,
                    "is_grayscale": is_gray,
                    "mask_format":  "npy_pickled_dict" if has_mask else None,
                    "structures":   list(MASK_STRUCTURE_KEYS) if has_mask else [],
                },
            )

    @staticmethod
    def _patient_id(stem: str) -> str:
        m = _STEM_RE.match(stem)
        return m.group(1) if m else stem

    @staticmethod
    def _is_grayscale(img_path: Path) -> bool:
        """Probe the PNG color mode without loading pixel data."""
        try:
            with Image.open(img_path) as img:
                return img.mode in ("L", "LA")
        except Exception:
            return False
