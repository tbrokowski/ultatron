"""
data/adapters/common_carotid.py  ·  Expert-mask carotid ultrasound adapter
===========================================================================

Common-Carotid-Artery-Ultrasound-Images contains PNG ultrasound images paired
with expert binary masks for the carotid intima-media complex.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class CommonCarotidArteryImagesAdapter(BaseAdapter):
    DATASET_ID = "Common-Carotid-Artery-Ultrasound-Images"
    ANATOMY_FAMILY = "vascular"
    SONODQS = "silver"
    DOI = ""

    def __init__(self, root, split_override=None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if root.name == cls.DATASET_ID and root.exists():
            return root
        candidate = root / cls.DATASET_ID
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{cls.DATASET_ID}: expected '{cls.DATASET_ID}' under {root}")

    def iter_entries(self) -> Iterator[USManifestEntry]:
        image_dir = self.root / "US images"
        mask_dir = self.root / "Expert mask images"
        if not image_dir.exists() or not mask_dir.exists():
            return

        images = sorted(image_dir.glob("*.png"))
        study_splits = self._group_split_map(self._study_id(img.stem) for img in images)

        for img_path in images:
            mask_path = mask_dir / img_path.name
            if not mask_path.exists():
                continue

            study_id = self._study_id(img_path.stem)
            instances = [
                self._make_instance(
                    instance_id=img_path.stem,
                    label_raw="expert_intima_media_mask",
                    label_ontology="intima_media",
                    mask_path=str(mask_path),
                    is_promptable=True,
                )
            ]

            yield self._make_entry(
                str(img_path),
                split=study_splits.get(study_id, "train"),
                modality="image",
                instances=instances,
                study_id=study_id,
                view_type="common_carotid_long_axis",
                has_mask=True,
                task_type="segmentation",
                ssl_stream="image",
                is_promptable=True,
                source_meta={
                    "cohort": self.DATASET_ID,
                    "study_id": study_id,
                    "mask_type": "expert_binary_mask",
                },
            )

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }

    @staticmethod
    def _study_id(stem: str) -> str:
        return stem.split("_slice_", 1)[0]
