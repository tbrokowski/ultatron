"""
data/adapters/liver/aul.py  ·  AUL adapter
==========================================

AUL (Annotated Ultrasound Liver): 735 liver ultrasound images classified into
Benign (200), Malignant (435), and Normal (100), each with polygon contour
annotations for liver parenchyma, liver outline, and (for non-Normal) tumor mass.

Layout on disk:
  AUL/
  ├── Benign/
  │   ├── image/          # 200 JPEG files, integer IDs (1.jpg, 25.jpg, ...)
  │   └── segmentation/
  │       ├── liver/      # 200 JSON polygon files
  │       ├── outline/    # 200 JSON polygon files
  │       └── mass/       # 200 JSON polygon files
  ├── Malignant/
  │   ├── image/          # 435 JPEG files
  │   └── segmentation/
  │       ├── liver/      # 432 JSONs (3 missing — handled gracefully)
  │       ├── outline/    # 435 JSON polygon files
  │       └── mass/       # 435 JSON polygon files
  └── Normal/
      ├── image/          # 100 JPEG files
      └── segmentation/
          ├── liver/      # 100 JSON polygon files
          ├── outline/    # 100 JSON polygon files
          └── mass/       # NOT PRESENT — normal liver has no mass

JSON format: list of [x, y] float pairs — polygon vertices.
  e.g. [[102.5, 340.0], [103.1, 341.8], ...]
  Convert to a binary mask with cv2.fillPoly or PIL.ImageDraw.polygon downstream.
  Polygons are stored in Instance.polygon; no pixel-mask files are generated.

Classification labels:
  0 = Normal
  1 = Benign
  2 = Malignant

Segmentation instances (polygon, one per annotation type that exists):
  label_ontology="liver_parenchyma"  ← liver/ JSON
  label_ontology="liver_outline"     ← outline/ JSON
  label_ontology="liver_lesion"      ← mass/ JSON (Benign and Malignant only)

Split strategy:
  No predefined split. Deterministic 80/10/10 by sorted integer ID within each
  class folder to keep class distribution balanced across splits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


CLASS_LABELS: Dict[str, int] = {
    "Normal":    0,
    "Benign":    1,
    "Malignant": 2,
}

# (segmentation subfolder, label_raw, label_ontology)
SEGMENT_SPECS: List[Tuple[str, str, str]] = [
    ("liver",   "liver",         "liver_parenchyma"),
    ("outline", "liver_outline", "liver_outline"),
    ("mass",    "liver_mass",    "liver_lesion"),
]


class AULAdapter(BaseAdapter):
    DATASET_ID     = "AUL"
    ANATOMY_FAMILY = "liver"
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
        if any((root / cls_name).is_dir() for cls_name in CLASS_LABELS):
            return root
        candidate = root / "AUL"
        if any((candidate / cls_name).is_dir() for cls_name in CLASS_LABELS):
            return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected Benign/ Malignant/ Normal/ under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for cls_name in sorted(CLASS_LABELS):
            img_dir = self.root / cls_name / "image"
            if not img_dir.exists():
                continue

            images = sorted(
                img_dir.glob("*.jpg"),
                key=lambda p: int(p.stem) if p.stem.isdigit() else float("inf"),
            )
            n = len(images)
            seg_dir = self.root / cls_name / "segmentation"
            cls_idx = CLASS_LABELS[cls_name]

            for i, img_path in enumerate(images):
                stem  = img_path.stem
                split = self.split_override or self._infer_split(
                    f"{cls_name}/{stem}", i, n
                )

                classification = self._make_instance(
                    instance_id          = f"{cls_name}_{stem}_cls",
                    label_raw            = cls_name,
                    label_ontology       = "aul_liver_class",
                    is_promptable        = False,
                    classification_label = cls_idx,
                )

                seg_instances = []
                for subfolder, label_raw, label_ontology in SEGMENT_SPECS:
                    json_path = seg_dir / subfolder / f"{stem}.json"
                    if not json_path.exists():
                        continue
                    polygon = self._load_polygon(json_path)
                    if polygon is None:
                        continue
                    seg_instances.append(
                        self._make_instance(
                            instance_id    = f"{cls_name}_{stem}_{subfolder}",
                            label_raw      = label_raw,
                            label_ontology = label_ontology,
                            is_promptable  = True,
                            polygon        = polygon,
                        )
                    )

                has_seg = bool(seg_instances)
                present_segs = [
                    sf for sf, _, _ in SEGMENT_SPECS
                    if (seg_dir / sf / f"{stem}.json").exists()
                ]

                yield self._make_entry(
                    str(img_path),
                    split         = split,
                    modality      = "image",
                    instances     = [classification] + seg_instances,
                    study_id      = f"{cls_name}_{stem}",
                    series_id     = f"{cls_name}_{stem}",
                    view_type     = "liver_bmode",
                    has_mask      = has_seg,
                    task_type     = "segmentation" if has_seg else "classification",
                    ssl_stream    = "image",
                    is_promptable = has_seg,
                    source_meta   = {
                        "class":                  cls_name,
                        "image_id":               stem,
                        "classification_label":   cls_idx,
                        "seg_subfolders_present": present_segs,
                        "polygon_format":         "xy_list",
                    },
                )

    @staticmethod
    def _load_polygon(json_path: Path) -> Optional[List[List[float]]]:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, list) and all(
                isinstance(pt, (list, tuple)) and len(pt) >= 2 for pt in data
            ):
                return [[float(pt[0]), float(pt[1])] for pt in data]
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return None
