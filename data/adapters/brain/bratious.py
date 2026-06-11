"""
data/adapters/brain/bratious.py  - BraTioUS brain ultrasound adapter
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class BraTioUSAdapter(BaseAdapter):
    DATASET_ID     = "braTioUS"
    ANATOMY_FAMILY = "brain"
    SONODQS        = "gold"
    DOI            = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir = self.root / "imagesBraTioUS-public-dataset"
        lbl_dir = self.root / "labelsBraTioUS-public-dataset"
        if not img_dir.is_dir():
            return

        images = sorted(img_dir.glob("*.nii.gz"))
        n = len(images)
        for i, img_path in enumerate(images):
            lbl_path = lbl_dir / img_path.name
            if not lbl_path.exists():
                continue

            split = self._infer_split(img_path.stem, i, n)
            instances: List[Instance] = [
                self._make_instance(
                    instance_id=img_path.stem,
                    label_raw="brain_lesion",
                    label_ontology="brain_tumor",
                    mask_path=str(lbl_path),
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
