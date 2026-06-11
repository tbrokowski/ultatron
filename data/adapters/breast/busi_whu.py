"""
data/adapters/breast/busi_whu.py  - BUSI-WHU breast segmentation adapter
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class BUSIWHUAdapter(BaseAdapter):
    DATASET_ID     = "busi-whu"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        root = Path(root)
        self._base = root / "BUSI-WHU" if (root / "BUSI-WHU").is_dir() else root

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir = self._base / "img"
        gt_dir = self._base / "gt"
        if not img_dir.is_dir():
            return

        images = sorted(img_dir.glob("*.bmp"))
        n = len(images)
        for i, img_path in enumerate(images):
            mask_path = gt_dir / f"{img_path.stem}_anno.bmp"
            if not mask_path.exists():
                continue

            split = self._infer_split(img_path.stem, i, n)
            instances: List[Instance] = [
                self._make_instance(
                    instance_id=img_path.stem,
                    label_raw="breast_lesion",
                    label_ontology="breast_lesion",
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
