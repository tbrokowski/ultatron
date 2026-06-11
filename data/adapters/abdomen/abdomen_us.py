"""
data/adapters/abdomen/abdomen_us.py  - AbdomenUS adapter

Dataset:  Abdominal Ultrasound dataset (AUS + RUS)
Task:     Segmentation / SSL
Layout:

    <root>/
        AUS/
            images/
                {train,test}/  <image>.png
            annotations/
                {train,test}/  <image>.png   (segmentation mask, same name)
        RUS/
            images/            <image>.png
            annotations/       <image>.png

AUS = Annotated Ultrasound; RUS = Raw Ultrasound (unlabelled).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)


class AbdomenUSAdapter(BaseAdapter):
    """
    Adapter for the Abdominal Ultrasound segmentation dataset.
    """

    DATASET_ID     = "AbdomenUS"
    ANATOMY_FAMILY = "abdominal"
    SONODQS        = "silver"
    DOI            = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self.root.exists():
            log.warning("AbdomenUS: root not found at %s", self.root)
            return

        # AUS: annotated subset, official train/test splits
        aus_root = self.root / "AUS"
        if aus_root.exists():
            for split_dir in ["train", "test"]:
                img_dir  = aus_root / "images"       / split_dir
                ann_dir  = aus_root / "annotations"  / split_dir
                if not img_dir.exists():
                    continue
                split = "train" if split_dir == "train" else "test"
                if self.split_override:
                    split = self.split_override
                for img_path in sorted(img_dir.glob("*.png")):
                    mask_path = ann_dir / img_path.name
                    has_mask  = mask_path.exists()
                    instances: List[Instance] = []
                    if has_mask:
                        instances.append(
                            self._make_instance(
                                instance_id=img_path.stem,
                                label_raw="abdominal_structure",
                                label_ontology="abdomen",
                                mask_path=str(mask_path),
                                is_promptable=True,
                            )
                        )
                    yield self._make_entry(
                        str(img_path),
                        split=split, modality="image",
                        instances=instances,
                        has_mask=has_mask,
                        task_type="segmentation" if has_mask else "ssl_only",
                        ssl_stream="image", is_promptable=has_mask,
                        source_meta={"subset": "AUS", "split": split_dir},
                    )

        # RUS: unlabelled — SSL only
        rus_root = self.root / "RUS"
        if rus_root.exists():
            img_dir = rus_root / "images"
            if img_dir.exists():
                split = self.split_override or "train"
                for img_path in sorted(img_dir.glob("*.png")):
                    yield self._make_entry(
                        str(img_path),
                        split=split, modality="image",
                        task_type="ssl_only",
                        ssl_stream="image", is_promptable=False,
                        source_meta={"subset": "RUS"},
                    )
