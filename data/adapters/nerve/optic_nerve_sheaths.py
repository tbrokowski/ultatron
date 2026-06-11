"""
data/adapters/nerve/optic_nerve_sheaths.py  - Optic nerve sheath adapter

Layout:
  .../Ultrasound-OpticNerveSheaths/DATA/
    IMAGES_256/*.png
    LABELS_256/*.png
    Paper_CV_folds/f{0-4}/{train,val,test}/ + annot dirs
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class OpticNerveSheathsAdapter(BaseAdapter):
    DATASET_ID     = "optic-nerve-sheaths"
    ANATOMY_FAMILY = "nerve"
    SONODQS        = "gold"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._data_root = self._find_data_root(self.root)
        self._splits = self._load_fold_splits(fold=0)

    @staticmethod
    def _find_data_root(root: Path) -> Path:
        for sub in root.rglob("Ultrasound-OpticNerveSheaths"):
            data = sub / "DATA"
            if data.is_dir():
                return data
        if (root / "DATA").is_dir():
            return root / "DATA"
        if (root / "IMAGES_256").is_dir():
            return root
        return root

    def _load_fold_splits(self, fold: int = 0) -> Dict[str, str]:
        fold_dir = self._data_root / "Paper_CV_folds" / f"f{fold}"
        mapping: Dict[str, str] = {}
        if not fold_dir.is_dir():
            return mapping
        for split in ("train", "val", "test"):
            split_dir = fold_dir / split
            if not split_dir.is_dir():
                continue
            for p in split_dir.glob("*.png"):
                mapping[p.name] = split
        return mapping

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir = self._data_root / "IMAGES_256"
        lbl_dir = self._data_root / "LABELS_256"
        if not img_dir.is_dir():
            return

        images = sorted(img_dir.glob("*.png"))
        n = len(images)
        for i, img_path in enumerate(images):
            mask_path = lbl_dir / img_path.name
            if not mask_path.exists():
                continue

            split = self._splits.get(img_path.name)
            if split is None:
                split = self._infer_split(img_path.stem, i, n)
            if self.split_override:
                split = self.split_override

            instances: List[Instance] = [
                self._make_instance(
                    instance_id=img_path.stem,
                    label_raw="optic_nerve_sheath",
                    label_ontology="optic_nerve",
                    mask_path=str(mask_path),
                    is_promptable=True,
                )
            ]

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                has_mask=True,
                task_type="segmentation",
                ssl_stream="image",
                is_promptable=True,
            )
