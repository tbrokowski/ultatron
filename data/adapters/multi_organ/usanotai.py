"""
data/adapters/multi_organ/usanotai.py  ·  USAnotAI abdominal organ adapter
==========================================================================

USAnotAI-master (SIIM 2019 Innovation Challenge): abdominal ultrasound
organ classification across six organ classes.

On-disk layout:
  USAnotAI-master/
  ├── train/   ← {organ}-00##.png
  └── test/    ← {organ}-005#.png

Organs: bladder, bowel, gallbladder, kidney, liver, spleen.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_FILENAME_RE = re.compile(r"^([a-z]+)-(\d+)\.png$", re.IGNORECASE)

_ORGAN_CLASSES: Tuple[Tuple[str, str, int], ...] = (
    ("bladder",     "bladder",     0),
    ("bowel",       "bowel",       1),
    ("gallbladder", "gallbladder", 2),
    ("kidney",      "kidney",      3),
    ("liver",       "liver",       4),
    ("spleen",      "spleen",      5),
)
_ORGAN_TO_ID = {name: idx for name, _, idx in _ORGAN_CLASSES}

_SPLIT_DIRS = (("train", "train"), ("test", "test"))


class USAnotAIAdapter(BaseAdapter):
    DATASET_ID = "USAnotAI-master"
    ANATOMY_FAMILY = "multi"
    SONODQS = "silver"
    DOI = ""

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "train").is_dir() or (root / "test").is_dir():
            return root
        candidate = root / cls.DATASET_ID
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected train/ and test/ under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for split_dir, split in _SPLIT_DIRS:
            dir_path = self.root / split_dir
            if not dir_path.is_dir():
                continue
            for img_path in sorted(dir_path.glob("*.png")):
                parsed = _FILENAME_RE.match(img_path.name)
                if parsed is None:
                    continue
                organ = parsed.group(1).lower()
                if organ not in _ORGAN_TO_ID:
                    continue

                split_label = self.split_override or split
                cls_id = _ORGAN_TO_ID[organ]
                instance = self._make_instance(
                    instance_id=img_path.stem,
                    label_raw=organ,
                    label_ontology="abdominal_organ",
                    is_promptable=False,
                )
                yield self._make_entry(
                    str(img_path),
                    split=split_label,
                    modality="image",
                    instances=[instance],
                    study_id=f"usanotai_{img_path.stem}",
                    view_type="abdominal_bmode",
                    has_mask=False,
                    task_type="multiclass_cls",
                    ssl_stream="image",
                    is_promptable=False,
                    source_meta={
                        "organ": organ,
                        "class_id": cls_id,
                        "original_split": split,
                    },
                )
