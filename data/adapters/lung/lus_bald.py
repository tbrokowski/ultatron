"""
data/adapters/lung/lus_bald.py  ·  LUS-BALD adapter
=====================================================

Dataset layout (Store):

    /capstor/.../lung/LUS-BALD/LUS-BALD/
      train/
        images/   *.png  *.jpeg
        labels/   *.txt
      val/
        images/
        labels/
      test/
        images/
        labels/

Label format — YOLO polygon, one annotation per line:
    class_id x1 y1 x2 y2 x3 y3 x4 y4   (normalized coords)

Class 0 is "b_line". One entry per image; one Instance per annotation line.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

_IMG_EXTS = {".png", ".jpg", ".jpeg"}
_CLASS_NAMES = {0: "b_line"}
_SPLITS = ("train", "val", "test")


def _parse_label_file(label_path: Path) -> List[Tuple[int, List[List[float]]]]:
    """Return list of (class_id, polygon_points) from a YOLO polygon .txt file."""
    if not label_path.exists():
        return []
    annotations = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 9:
            continue
        try:
            cls_id = int(parts[0])
            coords = [float(v) for v in parts[1:9]]
        except ValueError:
            continue
        polygon = [[coords[i], coords[i + 1]] for i in range(0, 8, 2)]
        annotations.append((cls_id, polygon))
    return annotations


def _bbox_from_polygon(polygon: List[List[float]]) -> List[float]:
    """Derive axis-aligned bbox_xyxy from polygon vertices."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


class LUSBALDAdapter(BaseAdapter):
    DATASET_ID     = "LUS-BALD"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "silver"
    DOI            = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for split in _SPLITS:
            images_dir = self.root / split / "images"
            labels_dir = self.root / split / "labels"
            if not images_dir.is_dir():
                continue

            for img_path in sorted(
                p for p in images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMG_EXTS
            ):
                label_path = labels_dir / (img_path.stem + ".txt")
                annotations = _parse_label_file(label_path)

                instances: List[Instance] = []
                for i, (cls_id, polygon) in enumerate(annotations):
                    cls_name = _CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                    inst = self._make_instance(
                        instance_id=f"{img_path.stem}_{i}",
                        label_raw=cls_name,
                        label_ontology=cls_name,
                        is_promptable=False,
                    )
                    inst.polygon   = polygon
                    inst.bbox_xyxy = _bbox_from_polygon(polygon)
                    instances.append(inst)

                has_box   = bool(instances)
                task_type = "detection" if has_box else "ssl_only"
                actual_split = self.split_override or split

                yield self._make_entry(
                    str(img_path),
                    split=actual_split,
                    modality="image",
                    instances=instances,
                    study_id=img_path.stem,
                    label_raw=["b_line"] if has_box else None,
                    has_mask=False,
                    has_box=has_box,
                    has_temporal_order=False,
                    num_frames=1,
                    task_type=task_type,
                    ssl_stream="image",
                    is_promptable=False,
                    source_meta={"label_path": str(label_path)},
                )
