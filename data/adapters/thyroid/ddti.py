"""
data/adapters/thyroid/ddti.py  ·  DDTI thyroid nodule adapter
==============================================================

DDTI: Digital Database of Thyroid Ultrasound Images.
  Layout:
    {root}/archive/{case}.xml
    {root}/archive/{case}_{image_idx}.jpg

Each XML stores TI-RADS metadata and freehand nodule polygons (JSON in <svg>).
Polygons are rasterized into cached PNG masks under ``.ddti_mask_cache/``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from PIL import Image

from data.adapters.base import BaseAdapter
from data.adapters.thyroid.ddti_layout import iter_ddti_samples
from data.adapters.thyroid.mask_cache import ensure_polygon_mask
from data.schema.manifest import USManifestEntry, Instance


class DDTIAdapter(BaseAdapter):
    DATASET_ID     = "DDTI"
    ANATOMY_FAMILY = "thyroid"
    SONODQS        = "gold"
    DOI            = "https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images"

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._mask_cache = self.root / ".ddti_mask_cache"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for sample, split in iter_ddti_samples(
            self.root,
            split_override=self.split_override,
            infer_split=self._infer_split,
        ):
            with Image.open(sample.image_path) as img:
                width, height = img.size

            sample_id = f"{sample.case_id}_{sample.image_idx}"
            mask_path = ensure_polygon_mask(
                cache_dir=self._mask_cache,
                sample_id=sample_id,
                polygons=sample.polygons,
                width=width,
                height=height,
            )
            if mask_path is None:
                continue

            instances: List[Instance] = [
                self._make_instance(
                    instance_id=f"{sample_id}_nodule",
                    label_raw="thyroid_nodule",
                    label_ontology="thyroid_nodule_boundary",
                    mask_path=str(mask_path),
                    is_promptable=True,
                    polygon=sample.polygons[0] if len(sample.polygons) == 1 else None,
                )
            ]

            if sample.tirads_label is not None:
                instances.append(
                    self._make_instance(
                        instance_id=f"{sample_id}_tirads",
                        label_raw=sample.tirads_raw or "unknown",
                        label_ontology="thyroid_tirads",
                        is_promptable=False,
                        classification_label=sample.tirads_label,
                    )
                )

            yield self._make_entry(
                str(sample.image_path),
                split,
                modality="image",
                instances=instances,
                study_id=sample.case_id,
                series_id=sample_id,
                has_mask=True,
                task_type="segmentation",
                ssl_stream="image",
                is_promptable=True,
                source_meta={
                    "case_id": sample.case_id,
                    "image_idx": sample.image_idx,
                    "xml_path": str(sample.xml_path),
                    "tirads_raw": sample.tirads_raw,
                    "composition": sample.composition,
                    "echogenicity": sample.echogenicity,
                    "margins": sample.margins,
                    "calcifications": sample.calcifications,
                    "age": sample.age,
                    "sex": sample.sex,
                    "polygon_count": len(sample.polygons),
                    "polygon_format": "ddti_svg_json",
                },
            )
