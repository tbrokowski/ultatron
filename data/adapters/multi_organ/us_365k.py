"""
data/adapters/us_365k.py  ·  US-365K image-caption adapter
============================================================

JJY-0823/US-365K — 364k ultrasound image / radiology-caption pairs.
Multi-organ coverage; captions stored as source_meta.report_text for CLIP.

On-disk layout (after download_us_365k.sh + materialize_us365k.py):
  US-365K/
    images/           flat JPEG filenames
    metadata/
      train.jsonl
      val.jsonl
      test.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_METADATA_SPLITS = ("train", "val", "test")


class US365KAdapter(BaseAdapter):
    DATASET_ID = "US-365K"
    ANATOMY_FAMILY = "multi"
    SONODQS = "bronze"
    DOI = "https://huggingface.co/datasets/JJY-0823/US-365K"

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)
        self.images_dir = self.root / "images"
        self.metadata_dir = self.root / "metadata"

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "metadata").is_dir() and (root / "images").is_dir():
            return root
        candidate = root / cls.DATASET_ID
        if (candidate / "metadata").is_dir():
            return candidate
        if (root / "metadata").is_dir():
            return root
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected metadata/ and images/ under {root}"
        )

    def _iter_records(self) -> Iterator[dict]:
        for split in _METADATA_SPLITS:
            jsonl = self.metadata_dir / f"{split}.jsonl"
            if not jsonl.exists():
                continue
            with jsonl.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec.setdefault("split", split)
                    yield rec

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for i, rec in enumerate(self._iter_records()):
            img_path = Path(rec["image"])
            if not img_path.is_file():
                alt = self.images_dir / img_path.name
                if alt.is_file():
                    img_path = alt
                else:
                    continue

            caption = (rec.get("caption") or "").strip()
            if not caption:
                continue

            split = self.split_override or rec.get("split", "train")
            if split == "validation":
                split = "val"

            sample_id = img_path.stem
            yield self._make_entry(
                str(img_path),
                split,
                modality="image",
                instances=[],
                has_mask=False,
                task_type="weak_label",
                ssl_stream="image",
                is_promptable=False,
                study_id=sample_id,
                source_meta={
                    "report_text": caption,
                    "caption": caption,
                    "image_fname": img_path.name,
                },
            )
