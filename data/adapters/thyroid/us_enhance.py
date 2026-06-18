"""
data/adapters/thyroid/us_enhance.py  ·  US Image Enhancement multi-organ adapter
==================================================================================

Dataset layout (Store):

    {root}/
      Training set/
        train_datasets/
          thyroid/  high_quality/*.png  low_quality/*.png
          breast/   high_quality/*.png  low_quality/*.png
          carotid/  high_quality/*.png  low_quality/*.png
          liver/    high_quality/*.png  low_quality/*.png
          kidney/   high_quality/*.png  low_quality/*.png
      Testing set/
        low_quality_images/  *.png

Training: paired low/high quality images per organ — one entry per pair.
Testing:  low quality images only, organ unknown — one entry per image.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, assign_curriculum_tier

_ORGANS = ("thyroid", "breast", "carotid", "liver", "kidney")
_IMG_EXTS = {".png", ".jpg", ".jpeg"}

_ORGAN_TO_ANATOMY = {
    "thyroid": "thyroid",
    "breast":  "breast",
    "carotid": "carotid",
    "liver":   "liver",
    "kidney":  "kidney",
}


class USEnhanceAdapter(BaseAdapter):
    DATASET_ID     = "us-enhance-2023"
    ANATOMY_FAMILY = "multi"
    SONODQS        = "silver"
    DOI            = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        yield from self._iter_train()
        yield from self._iter_test()

    def _iter_train(self) -> Iterator[USManifestEntry]:
        train_root = self.root / "Training set" / "train_datasets"
        if not train_root.is_dir():
            return

        for organ in _ORGANS:
            lq_dir = train_root / organ / "low_quality"
            hq_dir = train_root / organ / "high_quality"
            if not lq_dir.is_dir():
                continue

            anatomy = _ORGAN_TO_ANATOMY[organ]
            split   = self.split_override or "train"

            for lq_path in sorted(
                p for p in lq_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMG_EXTS
            ):
                hq_path  = hq_dir / lq_path.name if hq_dir.is_dir() else None
                has_pair = hq_path is not None and hq_path.exists()

                entry = self._make_entry(
                    str(lq_path),
                    split=split,
                    modality="image",
                    instances=[],
                    study_id=lq_path.stem,
                    label_raw=None,
                    has_mask=False,
                    has_temporal_order=False,
                    num_frames=1,
                    task_type="weak_label",
                    ssl_stream="image",
                    is_promptable=False,
                    source_meta={
                        "organ": organ,
                        "high_quality_path": str(hq_path) if has_pair else None,
                    },
                )
                entry.anatomy_family   = anatomy
                entry.curriculum_tier  = assign_curriculum_tier(entry)
                yield entry

    def _iter_test(self) -> Iterator[USManifestEntry]:
        test_dir = self.root / "Testing set" / "low_quality_images"
        if not test_dir.is_dir():
            return

        split = self.split_override or "test"

        for lq_path in sorted(
            p for p in test_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        ):
            entry = self._make_entry(
                str(lq_path),
                split=split,
                modality="image",
                instances=[],
                study_id=lq_path.stem,
                label_raw=None,
                has_mask=False,
                has_temporal_order=False,
                num_frames=1,
                task_type="weak_label",
                ssl_stream="image",
                is_promptable=False,
                source_meta={},
            )
            entry.anatomy_family  = "other"
            entry.curriculum_tier = assign_curriculum_tier(entry)
            yield entry
