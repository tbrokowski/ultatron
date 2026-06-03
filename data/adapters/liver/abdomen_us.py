"""
data/adapters/liver/abdomen_us.py  ·  AbdomenUS adapter
=========================================================

AbdomenUS: Abdominal ultrasound simulation and segmentation dataset
  (Vitale et al., Int J CARS 2019).
  Two subsets sharing the same folder layout:
    AUS — real ultrasound scans (617 images)
    RUS — simulated / ray-cast ultrasound scans (617 images)

Layout on disk:
  {root}/
  ├── AUS/
  │   ├── images/
  │   │   ├── train/   # PNG files, e.g. ct14-10.png
  │   │   └── test/
  │   └── annotations/
  │       ├── train/   # same-name PNG masks (AUS train all annotated)
  │       └── test/    # 61 of 293 test images have annotations
  └── RUS/
      ├── images/
      │   ├── train/
      │   └── test/
      └── annotations/
          └── test/    # 61 annotated masks (no train annotations)

Annotation masks are RGB PNGs with one unique color per organ:
  violet  (128,   0, 128) → liver         → liver_parenchyma
  yellow  (255, 255,   0) → kidney        → kidney
  blue    (  0,   0, 255) → pancreas      → pancreas
  red     (255,   0,   0) → vessels       → abdominal_vessel
  cyan    (  0, 255, 255) → adrenals      → adrenal_gland
  green   (  0, 128,   0) → gallbladder   → gallbladder
  white   (255, 255, 255) → bones         → bone
  pink    (255, 192, 203) → spleen        → spleen
  black   (  0,   0,   0) → background    (skipped)

Parsing strategy: load each annotation PNG with PIL, collect unique non-black
pixels, nearest-neighbor match (L2 in RGB) to the table above, create one
Instance per matched structure.  The same mask_path is stored on every
Instance belonging to the same image (downstream separates by label_raw).

Splits: predefined by folder (train/ → "train", test/ → "test").
  No val split in this dataset.

Constructor parameter:
  subset : "AUS" | "RUS"  (default "AUS")
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


# ── Color → (label_raw, label_ontology) ──────────────────────────────────────
# Each entry is (canonical_rgb, label_raw, label_ontology).
_COLOR_MAP: List[Tuple[Tuple[int, int, int], str, str]] = [
    ((128,   0, 128), "liver",       "liver_parenchyma"),
    ((255, 255,   0), "kidney",      "kidney"),
    ((  0,   0, 255), "pancreas",    "pancreas"),
    ((255,   0,   0), "vessels",     "abdominal_vessel"),
    ((  0, 255, 255), "adrenals",    "adrenal_gland"),
    ((  0, 128,   0), "gallbladder", "gallbladder"),
    ((255, 255, 255), "bones",       "bone"),
    ((255, 192, 203), "spleen",      "spleen"),
]

_COLOR_CENTERS = np.array([c for c, _, _ in _COLOR_MAP], dtype=np.float32)
_COLOR_THRESHOLD = 60.0     # maximum L2 distance to accept a match


def _match_colors(
    mask_path: Path,
) -> List[Tuple[str, str]]:
    """
    Return [(label_raw, label_ontology), …] for every organ color present in
    the mask, deduplicated and ordered by _COLOR_MAP index.
    """
    from PIL import Image

    img = Image.open(mask_path).convert("RGB")
    arr = np.array(img, dtype=np.float32).reshape(-1, 3)

    # Drop background pixels (all channels near zero)
    non_bg = arr[arr.max(axis=1) > 10]
    if len(non_bg) == 0:
        return []

    unique_colors = np.unique(non_bg.astype(np.uint8), axis=0).astype(np.float32)

    seen: set[int] = set()
    matches: List[Tuple[str, str]] = []
    for rgb in unique_colors:
        dists = np.linalg.norm(_COLOR_CENTERS - rgb, axis=1)
        idx   = int(np.argmin(dists))
        if dists[idx] < _COLOR_THRESHOLD and idx not in seen:
            seen.add(idx)
            matches.append((_COLOR_MAP[idx][1], _COLOR_MAP[idx][2]))

    return matches


_SPLITS = {"train": "train", "test": "test"}


class AbdomenUSAdapter(BaseAdapter):
    """
    AbdomenUS adapter.

    Yields one image entry per PNG across both splits of a chosen subset.
    Annotation masks are parsed on-the-fly (during iter_entries) to build
    per-structure Instance objects; each Instance carries the same mask_path
    as the other instances from the same image.
    """

    DATASET_ID     = "AbdomenUS"
    ANATOMY_FAMILY = "liver"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.1007/s11548-019-02046-5"

    def __init__(
        self,
        root: str | Path,
        split_override: Optional[str] = None,
        subset: str = "AUS",
    ):
        self.subset = subset
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        # root should contain AUS/ and/or RUS/
        if (root / "AUS").is_dir() or (root / "RUS").is_dir():
            return root
        # one level up — e.g. passed abdominal_US/AUS/ directly
        candidate = root.parent
        if (candidate / "AUS").is_dir() or (candidate / "RUS").is_dir():
            return candidate
        # two-level nesting from dataset root
        for depth in (root / "abdominal_US", root / "archive" / "abdominal_US"):
            if (depth / "AUS").is_dir() or (depth / "RUS").is_dir():
                return depth
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected AUS/ or RUS/ under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        subset_dir = self.root / self.subset
        if not subset_dir.exists():
            raise FileNotFoundError(
                f"{self.DATASET_ID}: subset directory not found: {subset_dir}"
            )

        images_base = subset_dir / "images"
        ann_base    = subset_dir / "annotations"

        for dir_name, split_label in _SPLITS.items():
            img_dir = images_base / dir_name
            if not img_dir.exists():
                continue

            ann_dir = ann_base / dir_name  # may not exist (e.g. RUS/train)

            for img_path in sorted(img_dir.glob("*.png")):
                stem      = img_path.stem
                mask_path = (ann_dir / f"{stem}.png") if ann_dir.exists() else None
                has_mask  = mask_path is not None and mask_path.exists()

                instances = []
                if has_mask:
                    for label_raw, label_ontology in _match_colors(mask_path):
                        instances.append(self._make_instance(
                            instance_id    = f"{stem}_{label_raw}",
                            label_raw      = label_raw,
                            label_ontology = label_ontology,
                            mask_path      = str(mask_path),
                            is_promptable  = True,
                        ))

                # Only consider the mask "present" when at least one structure was found.
                has_mask = bool(instances)

                yield self._make_entry(
                    str(img_path),
                    split         = self.split_override or split_label,
                    modality      = "image",
                    instances     = instances,
                    study_id      = stem,
                    series_id     = stem,
                    view_type     = "abdominal_us",
                    has_mask      = has_mask,
                    task_type     = "segmentation" if has_mask else "ssl_only",
                    ssl_stream    = "image",
                    is_promptable = has_mask,
                    source_meta   = {
                        "subset":     self.subset,
                        "image_id":   stem,
                        "has_mask":   has_mask,
                        "mask_path":  str(mask_path) if has_mask else None,
                        "structures": [i.label_raw for i in instances],
                    },
                )
