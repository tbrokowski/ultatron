"""
data/adapters/tn3k.py  ·  TN3K thyroid nodule adapter
===========================================================

TN3K: 3,493 thyroid ultrasound images with expert nodule segmentation.
  Format: JPG images + JPG masks
  Layout (official):
    {root}/trainval-image/*.jpg
    {root}/trainval-mask/*.jpg
    {root}/test-image/*.jpg
    {root}/test-mask/*.jpg
    {root}/tn3k-trainval-fold{N}.json  (train/val split within trainval)
"""
from __future__ import annotations

from typing import Iterator

from data.adapters.base import BaseAdapter
from data.adapters.thyroid.tn3k_layout import iter_tn3k_pairs
from data.schema.manifest import USManifestEntry


class TN3KAdapter(BaseAdapter):
    DATASET_ID     = "TN3K"
    ANATOMY_FAMILY = "thyroid"
    SONODQS        = "silver"
    DOI            = "https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation"
    fold: int      = 0

    def __init__(self, root, split_override=None, fold: int = 0):
        super().__init__(root, split_override=split_override)
        self.fold = fold

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for img_path, lbl_path, sample_id, split in iter_tn3k_pairs(
            self.root,
            fold=self.fold,
            split_override=self.split_override,
        ):
            instances = [
                self._make_instance(
                    instance_id    = sample_id,
                    label_raw      = "thyroid_nodule",
                    label_ontology = "thyroid_nodule_boundary",
                    mask_path      = str(lbl_path),
                    is_promptable  = True,
                )
            ]

            yield self._make_entry(
                str(img_path), split,
                modality      = "image",
                instances     = instances,
                has_mask      = True,
                task_type     = "segmentation",
                ssl_stream    = "image",
                is_promptable = True,
            )
