"""
data/adapters/maternal_fetal/jnu_ifm.py  ·  JNU-IFM adapter
============================================================

JNU-IFM contains intrapartum ultrasound frames grouped by video/session folder.
Each session has a CSV that defines the labelled frames:

  us_data/<video_id>/
  ├── image/<video_id>_<frame_id>.png
  ├── mask/<video_id>_<frame_id>_mask.png
  ├── mask_enhance/                         (visualization only, ignored)
  └── frame_label.csv

Raw mask values are not contiguous:

  0 = background
  7 = pubic symphysis
  8 = fetal head

This adapter writes deterministic remapped categorical masks under
`.jnu_ifm_mask_cache/`:

  0 = background
  1 = pubic symphysis
  2 = fetal head

The CSV frame_label is emitted as a secondary classification instance:

  3 -> 0 = none
  4 -> 1 = only_sp
  5 -> 2 = only_head
  6 -> 3 = sp_head
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


STRUCTURES: List[Tuple[str, str, int]] = [
    ("pubic_symphysis", "pubic_symphysis", 1),
    ("fetal_head",      "fetal_head",      2),
]

FRAME_LABELS: Dict[int, Tuple[str, int]] = {
    3: ("none", 0),
    4: ("only_sp", 1),
    5: ("only_head", 2),
    6: ("sp_head", 3),
}

RAW_TO_CONTIGUOUS = {
    7: 1,
    8: 2,
}


class JNUIFMAdapter(BaseAdapter):
    DATASET_ID     = "JNU-IFM"
    ANATOMY_FAMILY = "intrapartum"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.6084/m9.figshare.14371652"

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )
        self._mask_cache = self.root / ".jnu_ifm_mask_cache"

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if cls._looks_like_us_data(root):
            return root
        for candidate in (root / "us_data", root / "JNU-IFM" / "us_data"):
            if cls._looks_like_us_data(candidate):
                return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'us_data/<video_id>/frame_label.csv' under {root}"
        )

    @staticmethod
    def _looks_like_us_data(path: Path) -> bool:
        return path.is_dir() and any(
            p.is_dir() and (p / "frame_label.csv").exists()
            for p in path.iterdir()
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        video_dirs = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
        video_split = {
            video_dir.name: self._infer_split(video_dir.name, i, len(video_dirs))
            for i, video_dir in enumerate(video_dirs)
        }

        for video_dir in video_dirs:
            csv_path = video_dir / "frame_label.csv"
            image_dir = video_dir / "image"
            mask_dir = video_dir / "mask"
            if not csv_path.exists() or not image_dir.exists():
                continue

            rows = self._load_frame_rows(csv_path)
            for frame_id, raw_frame_label in rows:
                stem = f"{video_dir.name}_{frame_id}"
                img_path = image_dir / f"{stem}.png"
                raw_mask_path = mask_dir / f"{stem}_mask.png"
                if not img_path.exists():
                    continue

                has_mask = raw_mask_path.exists()
                remapped_mask_path = (
                    self._build_remapped_mask(
                        raw_mask_path=raw_mask_path,
                        video_id=video_dir.name,
                        stem=stem,
                    )
                    if has_mask else None
                )

                frame_label_raw, frame_label_idx = FRAME_LABELS.get(
                    raw_frame_label,
                    (f"unknown_{raw_frame_label}", -1),
                )

                instances = []
                if has_mask:
                    instances.extend(
                        self._make_instance(
                            instance_id=f"{stem}_{struct_name}",
                            label_raw=struct_name,
                            label_ontology=label_ontology,
                            mask_path=str(remapped_mask_path),
                            mask_channel=class_value,
                            is_promptable=self._structure_visible(raw_frame_label, struct_name),
                        )
                        for struct_name, label_ontology, class_value in STRUCTURES
                    )

                instances.append(
                    self._make_instance(
                        instance_id=f"{stem}_frame_label",
                        label_raw=frame_label_raw,
                        label_ontology="jnu_ifm_frame_visibility",
                        classification_label=frame_label_idx,
                        is_promptable=False,
                    )
                )

                split = self.split_override or video_split.get(video_dir.name, "train")
                yield self._make_entry(
                    str(img_path),
                    split=split,
                    modality="image",
                    instances=instances,
                    study_id=video_dir.name,
                    series_id=stem,
                    instance_id=stem,
                    frame_indices=[frame_id],
                    view_type="intrapartum_transperineal",
                    has_mask=has_mask,
                    task_type="segmentation" if has_mask else "classification",
                    ssl_stream="image",
                    is_promptable=has_mask,
                    source_meta={
                        "video_id": video_dir.name,
                        "frame_id": frame_id,
                        "frame_label_raw": raw_frame_label,
                        "frame_label": frame_label_raw,
                        "frame_label_index": frame_label_idx,
                        "raw_mask_path": str(raw_mask_path) if raw_mask_path.exists() else None,
                        "remapped_mask_path": str(remapped_mask_path) if remapped_mask_path else None,
                        "mask_value_mapping": {"7": 1, "8": 2},
                        "mask_enhance_ignored": True,
                        "is_grayscale": self._is_grayscale(img_path),
                    },
                )

    @staticmethod
    def _load_frame_rows(csv_path: Path) -> List[Tuple[int, int]]:
        rows: List[Tuple[int, int]] = []
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                frame_id_s = (row.get("frame_id") or "").strip()
                label_s = (row.get("frame_label") or "").strip()
                if not frame_id_s or not label_s:
                    continue
                try:
                    rows.append((int(frame_id_s), int(label_s)))
                except ValueError:
                    continue
        return sorted(rows, key=lambda item: item[0])

    def _build_remapped_mask(self, raw_mask_path: Path, video_id: str, stem: str) -> Path:
        out_dir = self._mask_cache / video_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.npy"

        with Image.open(raw_mask_path) as img:
            raw = np.array(img.convert("L"), dtype=np.uint8)

        mapped = np.zeros_like(raw, dtype=np.uint8)
        for raw_value, mapped_value in RAW_TO_CONTIGUOUS.items():
            mapped[raw == raw_value] = mapped_value
        np.save(out_path, mapped)
        return out_path

    @staticmethod
    def _structure_visible(raw_frame_label: int, structure: str) -> bool:
        if structure == "pubic_symphysis":
            return raw_frame_label in (4, 6)
        if structure == "fetal_head":
            return raw_frame_label in (5, 6)
        return False

    @staticmethod
    def _is_grayscale(img_path: Path) -> bool:
        try:
            with Image.open(img_path) as img:
                return img.mode in ("L", "LA")
        except Exception:
            return False
