"""
data/adapters/muscle/msk_nmd.py  - MSK NMD Radboud adapter

Layout:
  <root>/.../{TA,BB,GM}/{Healthy,Pathological}/Images/*.png
  <root>/.../{TA,BB,GM}/{Healthy,Pathological}/Masks/*.png
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

_MUSCLES = ("TA", "BB", "GM")
_CONDITIONS = ("Healthy", "Pathological")


class MSKNMDAdapter(BaseAdapter):
    DATASET_ID     = "msk-nmd-radboud"
    ANATOMY_FAMILY = "musculoskeletal"
    SONODQS        = "gold"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._data_root = self._find_data_root(self.root)

    @staticmethod
    def _find_data_root(root: Path) -> Path:
        if (root / "TA").is_dir() or (root / "BB").is_dir():
            return root
        for sub in root.iterdir() if root.is_dir() else []:
            nested = sub / "Polito-Radboud-DeepLearningUS"
            if nested.is_dir():
                return nested
            if (sub / "TA").is_dir():
                return sub
        return root

    def iter_entries(self) -> Iterator[USManifestEntry]:
        samples: List[tuple[Path, Path, str, str]] = []
        for muscle in _MUSCLES:
            for condition in _CONDITIONS:
                img_dir = self._data_root / muscle / condition / "Images"
                msk_dir = self._data_root / muscle / condition / "Masks"
                if not img_dir.is_dir():
                    continue
                for img_path in sorted(img_dir.glob("*.png")):
                    mask_path = msk_dir / img_path.name
                    if mask_path.exists():
                        samples.append((img_path, mask_path, muscle, condition))

        n = len(samples)
        for i, (img_path, mask_path, muscle, condition) in enumerate(samples):
            split = self._infer_split(img_path.stem, i, n)
            label = "healthy" if condition == "Healthy" else "pathological"

            instances = [
                self._make_instance(
                    instance_id=img_path.stem,
                    label_raw=label,
                    label_ontology="muscle",
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
                source_meta={"muscle": muscle, "condition": condition},
            )
