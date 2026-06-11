"""
data/adapters/thyroid/tn5000.py  ·  TN5000 thyroid nodule adapter
==================================================================

TN5000: 5,000 thyroid ultrasound images in PASCAL VOC format.
  Layout:
    {root}/Main data/JPEGImages/*.jpg
    {root}/Main data/Annotations/*.xml
    {root}/Main data/ImageSets/Main/{train,val,test}.txt

Each annotation provides a nodule bounding box and benign/malignant label
(0 = benign, 1 = malignant).  Bounding boxes are rasterized into cached PNG
masks under ``.tn5000_mask_cache/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from PIL import Image

from data.adapters.base import BaseAdapter
from data.adapters.thyroid.mask_cache import ensure_bbox_mask
from data.adapters.thyroid.tn5000_layout import iter_tn5000_samples
from data.schema.manifest import USManifestEntry, Instance


class TN5000Adapter(BaseAdapter):
    DATASET_ID     = "TN5000"
    ANATOMY_FAMILY = "thyroid"
    SONODQS        = "gold"
    DOI            = "https://www.nature.com/articles/s41597-025-05757-4"

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._mask_cache = self.root / ".tn5000_mask_cache"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for sample in iter_tn5000_samples(
            self.root,
            split_override=self.split_override,
        ):
            width = sample.width
            height = sample.height
            if width <= 0 or height <= 0:
                with Image.open(sample.image_path) as img:
                    width, height = img.size

            mask_path = ensure_bbox_mask(
                cache_dir=self._mask_cache,
                sample_id=sample.stem,
                bbox_xyxy=sample.bbox_xyxy,
                width=width,
                height=height,
            )
            if mask_path is None:
                continue

            nodule_inst = self._make_instance(
                instance_id=f"{sample.stem}_nodule",
                label_raw="thyroid_nodule",
                label_ontology="thyroid_nodule_boundary",
                mask_path=str(mask_path),
                is_promptable=True,
            )
            nodule_inst.bbox_xyxy = list(sample.bbox_xyxy)

            instances: List[Instance] = [nodule_inst]
            if sample.nodule_label in (0, 1):
                instances.append(
                    self._make_instance(
                        instance_id=f"{sample.stem}_cls",
                        label_raw="malignant" if sample.nodule_label == 1 else "benign",
                        label_ontology="thyroid_nodule_class",
                        is_promptable=False,
                        classification_label=sample.nodule_label,
                    )
                )

            yield self._make_entry(
                str(sample.image_path),
                sample.split,
                modality="image",
                instances=instances,
                study_id=sample.stem,
                series_id=sample.stem,
                has_mask=True,
                has_box=True,
                task_type="segmentation",
                ssl_stream="image",
                is_promptable=True,
                source_meta={
                    "xml_path": str(sample.xml_path),
                    "bbox_xyxy": list(sample.bbox_xyxy),
                    "nodule_label": sample.nodule_label,
                    "nodule_class": (
                        "malignant" if sample.nodule_label == 1
                        else "benign" if sample.nodule_label == 0
                        else "unknown"
                    ),
                    "annotation_format": "pascal_voc",
                },
            )
