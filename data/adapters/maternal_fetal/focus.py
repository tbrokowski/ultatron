"""
data/adapters/maternal_fetal/focus.py  ·  FOCUS adapter
=======================================================

FOCUS contains fetal ultrasound snapshots with cardiac and thorax annotations.
Each split has the same structure:

  training|validation|testing/
  ├── images/<id>.png
  ├── annfiles_mask/<id>-cardiac.png
  ├── annfiles_mask/<id>-thorax.png
  ├── annfiles_ellipse/<id>.txt
  └── annfiles_rectangle/<id>.txt

Masks are stored one PNG per class.  This adapter builds a deterministic
two-channel NumPy mask stack per image so missing class masks become explicit
zero channels:

  channel 0 = cardiac
  channel 1 = thorax

The generated stack lives under `.focus_mask_cache/` inside the resolved dataset
root.  Instance.mask_channel selects the structure channel, matching the pattern
used by other multi-structure maternal/fetal adapters.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


SPLITS: Dict[str, str] = {
    "training": "train",
    "validation": "val",
    "testing": "test",
}

STRUCTURES: List[Tuple[str, str, int]] = [
    ("cardiac", "fetal_cardiac", 0),
    ("thorax",  "fetal_thorax",  1),
]

_NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


class FOCUSAdapter(BaseAdapter):
    DATASET_ID     = "FOCUS"
    ANATOMY_FAMILY = "fetal_cardiac"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )
        self._mask_cache = self.root / ".focus_mask_cache"

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if any((root / split_name).is_dir() for split_name in SPLITS):
            return root
        candidate = root / cls.DATASET_ID
        if any((candidate / split_name).is_dir() for split_name in SPLITS):
            return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected one of {sorted(SPLITS)} under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not any((self.root / split_name).exists() for split_name in SPLITS):
            raise FileNotFoundError(
                f"FOCUS: no split directories found under {self.root}"
            )

        for split_dir_name, split_label in SPLITS.items():
            split_dir = self.root / split_dir_name
            if not split_dir.exists():
                continue

            images_dir = split_dir / "images"
            if not images_dir.exists():
                continue

            mask_dir = split_dir / "annfiles_mask"
            ellipse_dir = split_dir / "annfiles_ellipse"
            rectangle_dir = split_dir / "annfiles_rectangle"

            for img_path in sorted(images_dir.glob("*.png")):
                stem = img_path.stem
                raw_masks = {
                    struct_name: mask_dir / f"{stem}-{struct_name}.png"
                    for struct_name, _ontology, _channel in STRUCTURES
                }
                present_masks = {
                    name: path for name, path in raw_masks.items()
                    if path.exists()
                }

                ellipse_path = ellipse_dir / f"{stem}.txt"
                rectangle_path = rectangle_dir / f"{stem}.txt"
                ellipses = self._parse_ellipse_file(ellipse_path)
                rectangles = self._parse_rectangle_file(rectangle_path)

                if present_masks:
                    stack_path = self._build_mask_stack(
                        img_path=img_path,
                        raw_masks=raw_masks,
                        split_dir_name=split_dir_name,
                        stem=stem,
                    )
                    instances = [
                        self._make_instance(
                            instance_id=f"{stem}_{struct_name}",
                            label_raw=struct_name,
                            label_ontology=label_ontology,
                            mask_path=str(stack_path),
                            mask_channel=channel,
                            bbox_xyxy=self._first_bbox(rectangles, struct_name),
                            is_promptable=True,
                        )
                        for struct_name, label_ontology, channel in STRUCTURES
                    ]
                    task_type = "segmentation"
                    has_mask = True
                    is_promptable = True
                else:
                    instances = []
                    task_type = "detection" if rectangles else "ssl_only"
                    has_mask = False
                    is_promptable = bool(rectangles)

                yield self._make_entry(
                    str(img_path),
                    split=self.split_override or split_label,
                    modality="image",
                    instances=instances,
                    study_id=stem,
                    series_id=stem,
                    view_type="fetal_cardiothoracic",
                    has_mask=has_mask,
                    has_box=bool(rectangles),
                    task_type=task_type,
                    ssl_stream="image",
                    is_promptable=is_promptable,
                    source_meta={
                        "split_dir": split_dir_name,
                        "stem": stem,
                        "is_grayscale": self._is_grayscale(img_path),
                        "raw_mask_paths": {
                            name: str(path) if path.exists() else None
                            for name, path in raw_masks.items()
                        },
                        "mask_stack_path": str(stack_path) if present_masks else None,
                        "missing_mask_classes": [
                            name for name, path in raw_masks.items() if not path.exists()
                        ],
                        "ellipse_txt": str(ellipse_path) if ellipse_path.exists() else None,
                        "rectangle_txt": str(rectangle_path) if rectangle_path.exists() else None,
                        "ellipses": ellipses,
                        "rectangles": rectangles,
                    },
                )

    def _build_mask_stack(
        self,
        img_path: Path,
        raw_masks: Dict[str, Path],
        split_dir_name: str,
        stem: str,
    ) -> Path:
        out_dir = self._mask_cache / split_dir_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{stem}.npy"

        with Image.open(img_path) as img:
            width, height = img.size

        channels = []
        for struct_name, _ontology, _channel in STRUCTURES:
            mask_path = raw_masks[struct_name]
            if mask_path.exists():
                with Image.open(mask_path) as mask_img:
                    mask = np.array(mask_img.convert("L"), dtype=np.uint8)
                mask = (mask > 127).astype(np.uint8)
            else:
                mask = np.zeros((height, width), dtype=np.uint8)
            channels.append(mask)

        stack = np.stack(channels, axis=0)
        np.save(out_path, stack)
        return out_path

    @classmethod
    def _parse_ellipse_file(cls, path: Path) -> List[dict]:
        if not path.exists():
            return []

        out = []
        for line in path.read_text().splitlines():
            numbers, label = cls._numbers_and_label(line)
            if len(numbers) < 5:
                continue
            out.append({
                "label": label,
                "center_x": numbers[0],
                "center_y": numbers[1],
                "axis_a": numbers[2],
                "axis_b": numbers[3],
                "rotation_angle": numbers[4],
            })
        return out

    @classmethod
    def _parse_rectangle_file(cls, path: Path) -> List[dict]:
        if not path.exists():
            return []

        out = []
        for line in path.read_text().splitlines():
            numbers, label = cls._numbers_and_label(line)
            if len(numbers) < 8:
                continue
            coords = numbers[:8]
            xs = coords[0::2]
            ys = coords[1::2]
            out.append({
                "label": label,
                "points": [[xs[i], ys[i]] for i in range(4)],
                "bbox_xyxy": [min(xs), min(ys), max(xs), max(ys)],
            })
        return out

    @staticmethod
    def _numbers_and_label(line: str) -> Tuple[List[float], Optional[str]]:
        numbers = [float(m.group(0)) for m in _NUMBER_RE.finditer(line)]
        cleaned = _NUMBER_RE.sub(" ", line)
        tokens = [
            t.strip("[](),;:")
            for t in cleaned.split()
            if t.strip("[](),;:")
        ]
        label = next((t.lower() for t in tokens if t.lower() in {"cardiac", "thorax"}), None)
        if label is None and numbers:
            label_idx = int(numbers[-1])
            if label_idx in (0, 1):
                label = STRUCTURES[label_idx][0]
            elif label_idx in (1, 2):
                label = STRUCTURES[label_idx - 1][0]
        return numbers, label

    @staticmethod
    def _first_bbox(rectangles: List[dict], label: str) -> Optional[List[float]]:
        for rect in rectangles:
            if rect.get("label") == label:
                return rect.get("bbox_xyxy")
        return None

    @staticmethod
    def _is_grayscale(img_path: Path) -> bool:
        try:
            with Image.open(img_path) as img:
                return img.mode in ("L", "LA")
        except Exception:
            return False
