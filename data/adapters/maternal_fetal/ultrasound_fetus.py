"""
data/adapters/maternal_fetal/ultrasound_fetus.py  ·  Ultrasound Fetus Dataset adapter
=======================================================================================

HC18-derived fetal head ultrasound dataset with health classification labels.

Layout on Store:

    {root}/.../Data/Data/
      train/
        benign/      *.png
        malignant/   *.png  *_Annotation.png
        normal/      *.png
      test/          (same structure)
      validation/    (same structure)
      Datasets/      ← ignored (no split assignment)

For malignant images, a paired *_Annotation.png mask may exist in the same
folder; skip annotation files when iterating, attach as mask_path on instance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

_SPLIT_MAP = {
    "train":      "train",
    "validation": "val",
    "test":       "test",
}
_CLASSES = ("benign", "malignant", "normal")


class UltrasoundFetusAdapter(BaseAdapter):
    DATASET_ID     = "ultrasound-fetus-dataset"
    ANATOMY_FAMILY = "fetal_head"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._data_root = self._find_data_root(self.root)

    @staticmethod
    def _find_data_root(root: Path) -> Path:
        for sub in root.rglob("Data"):
            if (sub / "train").is_dir() or (sub / "validation").is_dir():
                return sub
        nested = root / "Ultrasound Fetus Dataset" / "Ultrasound Fetus Dataset" / "Data" / "Data"
        if nested.is_dir():
            return nested
        return root

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for split_dir_name, split in _SPLIT_MAP.items():
            split_root = self._data_root / split_dir_name
            if not split_root.is_dir():
                continue

            effective_split = self.split_override or split

            for cls_name in _CLASSES:
                cls_dir = split_root / cls_name
                if not cls_dir.is_dir():
                    continue

                for img_path in sorted(cls_dir.glob("*.png")):
                    # skip annotation mask files
                    if img_path.stem.endswith("_Annotation"):
                        continue

                    mask_path: Path | None = None
                    if cls_name == "malignant":
                        candidate = cls_dir / f"{img_path.stem}_Annotation.png"
                        if candidate.exists():
                            mask_path = candidate

                    has_mask = mask_path is not None

                    instances: List[Instance] = [
                        self._make_instance(
                            instance_id=img_path.stem,
                            label_raw=cls_name,
                            label_ontology="fetal_health",
                            mask_path=str(mask_path) if has_mask else None,
                            is_promptable=has_mask,
                        )
                    ]

                    yield self._make_entry(
                        str(img_path),
                        split=effective_split,
                        modality="image",
                        instances=instances,
                        study_id=img_path.stem,
                        label_raw=[cls_name],
                        has_mask=has_mask,
                        has_box=False,
                        has_temporal_order=False,
                        num_frames=1,
                        task_type="classification",
                        ssl_stream="image",
                        is_promptable=has_mask,
                        source_meta={"class_name": cls_name},
                    )
