"""
data/adapters/maternal_fetal/fh_ps_aop.py  ·  FH-PS-AOP adapter
================================================================

Pubic Symphysis–Fetal Head Segmentation and Angle of Progression
(FH-PS-AOP): 4,000 2-D intrapartum transperineal ultrasound frames paired
with multi-class segmentation masks.

Layout on disk:
  {root}/
  └── Pubic Symphysis-Fetal Head Segmentation and Angle of Progression/
      ├── image_mha/
      │   └── <id5digits>.mha   ← grayscale stored as 3 identical channels
      └── label_mha/
          └── <id5digits>.mha   ← single-channel mask, values {0, 1, 2}

Image format:
  MetaImage (.mha), uint8, DimSize = 256 256 3.
  All three channels are identical — the image is grayscale stored
  redundantly.  Load with: arr = sitk.GetArrayFromImage(sitk.ReadImage(...))
  → shape (3, 256, 256); take arr[0] for the (256, 256) grayscale plane.

Label format:
  MetaImage (.mha), uint8, DimSize = 256 256.
  Class encoding:
    0 = background
    1 = pubic symphysis
    2 = fetal head

Two Instance objects are emitted per image, one per foreground class, both
pointing at the same mask_path.  mask_channel encodes the class label value
(1 or 2) so the downstream loader knows which class to extract.

No patient-level metadata exists in the filenames (opaque 5-digit IDs).
Split is deterministic on file stem as a proxy.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

# mask_channel = class label value in the segmentation mask
STRUCTURES: List[Tuple[str, str, int]] = [
    ("pubic_symphysis", "pubic_symphysis", 1),
    ("fetal_head",      "fetal_head",      2),
]


class FHPSAOPAdapter(BaseAdapter):
    DATASET_ID     = "FH-PS-AOP"
    ANATOMY_FAMILY = "intrapartum"
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
        if (root / "image_mha").is_dir():
            return root
        if root.is_dir():
            for candidate in sorted(root.iterdir()):
                if candidate.is_dir() and (candidate / "image_mha").is_dir():
                    return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'image_mha/' under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir  = self.root / "image_mha"
        mask_dir = self.root / "label_mha"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"FH-PS-AOP: image directory not found at {img_dir}"
            )

        # Index masks by stem — pair by name, not by position.
        mask_index: Dict[str, Path] = (
            {p.stem: p for p in mask_dir.glob("*.mha")}
            if mask_dir.exists() else {}
        )

        images = sorted(img_dir.glob("*.mha"))
        n = len(images)

        for i, img_path in enumerate(images):
            stem      = img_path.stem
            mask_path = mask_index.get(stem)
            has_mask  = mask_path is not None
            split     = self.split_override or self._infer_split(stem, i, n)

            if has_mask:
                instances = [
                    self._make_instance(
                        instance_id    = f"{stem}_{struct_name}",
                        label_raw      = struct_name,
                        label_ontology = label_ontology,
                        mask_path      = str(mask_path),
                        mask_channel   = class_label,
                        is_promptable  = True,
                    )
                    for struct_name, label_ontology, class_label in STRUCTURES
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
                study_id      = stem,
                series_id     = stem,
                view_type     = "intrapartum_transperineal",
                has_mask      = has_mask,
                task_type     = task_type,
                ssl_stream    = "image",
                is_promptable = has_mask,
                source_meta   = {
                    "stem":       stem,
                    "structures": [s[0] for s in STRUCTURES] if has_mask else [],
                },
            )
