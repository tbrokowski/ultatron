"""
data/adapters/ovarian/mmotu2d.py  - MMOTU-2D ovarian tumor adapter
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Set

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class MMOTU2DAdapter(BaseAdapter):
    DATASET_ID     = "MMOTU-2D"
    ANATOMY_FAMILY = "ovarian"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._base = self._find_base(self.root)
        self._train_ids, self._val_ids = self._load_splits()
        self._cls_labels = self._load_cls_labels()

    @staticmethod
    def _find_base(root: Path) -> Path:
        otu = root / "OTU_2d"
        return otu if otu.is_dir() else root

    def _load_splits(self) -> tuple[Set[str], Set[str]]:
        train_ids: Set[str] = set()
        val_ids: Set[str] = set()
        for name, target in (("train.txt", train_ids), ("val.txt", val_ids)):
            p = self._base / name
            if p.exists():
                for line in p.read_text().splitlines():
                    line = line.strip()
                    if line:
                        target.add(Path(line).stem)
        return train_ids, val_ids

    def _load_cls_labels(self) -> Dict[str, int]:
        labels: Dict[str, int] = {}
        for name in ("train_cls.txt", "val_cls.txt"):
            p = self._base / name
            if not p.exists():
                continue
            for line in p.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    labels[Path(parts[0]).stem] = int(parts[1])
        return labels

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir = self._base / "images"
        ann_dir = self._base / "annotations"
        if not img_dir.is_dir():
            return

        for img_path in sorted(img_dir.glob("*.JPG")) + sorted(img_dir.glob("*.jpg")):
            stem = img_path.stem
            mask_path = ann_dir / f"{stem}.PNG"
            if not mask_path.exists():
                mask_path = ann_dir / f"{stem}_binary.PNG"
            if not mask_path.exists():
                continue

            if self.split_override:
                split = self.split_override
            elif stem in self._val_ids:
                split = "val"
            elif stem in self._train_ids:
                split = "train"
            else:
                split = "train"

            cls_id = self._cls_labels.get(stem, -1)
            instances: List[Instance] = [
                self._make_instance(
                    instance_id=stem,
                    label_raw=f"tumor_class_{cls_id}" if cls_id >= 0 else "ovarian_tumor",
                    label_ontology="ovarian_tumor",
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
                source_meta={"class_id": cls_id},
            )
