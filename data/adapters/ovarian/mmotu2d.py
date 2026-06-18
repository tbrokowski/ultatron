"""
data/adapters/ovarian/mmotu2d.py  ·  MMOTU-2D ovarian tumor adapter
=====================================================================

Multi-class ovarian tumor ultrasound segmentation dataset.

Layout on Store:

    {root}/OTU_2d/
      images/       *.JPG
      annotations/  {id}.PNG           ← multiclass semantic mask (primary)
                    {id}_binary.PNG    ← binary lesion mask
      train.txt     image ID stems, one per line
      val.txt
      train_cls.txt filename + class_id, e.g. "658.JPG  5"
      val_cls.txt

Classes (1-indexed):
  1 chocolate_cyst      5 simple_cyst
  2 serous_cystadenoma  6 theca_cell_tumor
  3 mucinous_cystadenoma 7 high_grade_serous_carcinoma
  4 teratoma            8 normal_ovary
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

_CLASS_NAMES: Dict[int, str] = {
    1: "chocolate_cyst",
    2: "serous_cystadenoma",
    3: "mucinous_cystadenoma",
    4: "teratoma",
    5: "simple_cyst",
    6: "theca_cell_tumor",
    7: "high_grade_serous_carcinoma",
    8: "normal_ovary",
}


class MMOTU2DAdapter(BaseAdapter):
    DATASET_ID     = "MMOTU-2D"
    ANATOMY_FAMILY = "ovarian"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._base       = self._find_base(self.root)
        self._train_ids, self._val_ids = self._load_splits()
        self._cls_labels = self._load_cls_labels()

    @staticmethod
    def _find_base(root: Path) -> Path:
        otu = root / "OTU_2d"
        return otu if otu.is_dir() else root

    def _load_splits(self) -> tuple[Set[str], Set[str]]:
        train_ids: Set[str] = set()
        val_ids:   Set[str] = set()
        for fname, target in (("train.txt", train_ids), ("val.txt", val_ids)):
            p = self._base / fname
            if p.exists():
                for line in p.read_text(encoding="utf-8").splitlines():
                    stem = line.strip()
                    if stem:
                        target.add(Path(stem).stem)
        return train_ids, val_ids

    def _load_cls_labels(self) -> Dict[str, int]:
        labels: Dict[str, int] = {}
        for fname in ("train_cls.txt", "val_cls.txt"):
            p = self._base / fname
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        labels[Path(parts[0]).stem] = int(parts[1])
                    except ValueError:
                        pass
        return labels

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir = self._base / "images"
        ann_dir = self._base / "annotations"
        if not img_dir.is_dir():
            return

        images = sorted(
            p for p in img_dir.iterdir()
            if p.suffix.upper() == ".JPG" and p.is_file()
        )

        for img_path in images:
            stem      = img_path.stem
            mask_path = ann_dir / f"{stem}.PNG"
            if not mask_path.exists():
                continue

            binary_mask = ann_dir / f"{stem}_binary.PNG"

            if self.split_override:
                split = self.split_override
            elif stem in self._val_ids:
                split = "val"
            else:
                split = "train"

            cls_id     = self._cls_labels.get(stem, -1)
            cls_name   = _CLASS_NAMES.get(cls_id, "ovarian_tumor")

            instances: List[Instance] = [
                self._make_instance(
                    instance_id=stem,
                    label_raw=cls_name,
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
                study_id=stem,
                label_raw=[cls_name],
                has_mask=True,
                has_box=False,
                has_temporal_order=False,
                num_frames=1,
                task_type="segmentation",
                ssl_stream="image",
                is_promptable=True,
                source_meta={
                    "class_id":          cls_id,
                    "binary_mask_path":  str(binary_mask) if binary_mask.exists() else None,
                },
            )
