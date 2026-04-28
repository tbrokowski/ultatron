"""
data/adapters/maternal_fetal/fpus23.py  ·  FPUS23 adapter
==========================================================

FPUS23 contains two independent fetal ultrasound sub-datasets:

  archive/
  ├── Dataset/                 fetal pose/orientation streams
  │   ├── four_poses/<stream>/frame_<id>.png
  │   ├── annos/annotation/<stream>/annotations.xml
  │   └── boxes/annotation/<stream>/annotations.xml
  └── Dataset_Plane/           fetal plane classification images
      ├── AC_PLANE/*.png
      ├── BPD_PLANE/*.png
      ├── FL_PLANE/*.png
      └── NO_PLANE/*.png

For the stream sub-dataset, one manifest entry is emitted per frame so that
per-frame CVAT boxes can be represented as Instance.bbox_xyxy values.  The
split is assigned by stream name to avoid temporal leakage within streams.

For Dataset_Plane, one image classification entry is emitted per PNG.  No
patient or sequence metadata is available, so deterministic file-level splitting
is used as a fallback.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


PLANE_LABELS: Dict[str, Tuple[str, int]] = {
    "AC_PLANE": ("abdominal_circumference_plane", 0),
    "BPD_PLANE": ("biparietal_diameter_plane", 1),
    "FL_PLANE": ("femur_length_plane", 2),
    "NO_PLANE": ("no_standard_plane", 3),
}

POSE_LABELS: Dict[str, int] = {
    "hdvb": 0,
    "huvb": 1,
    "hdvf": 2,
    "huvf": 3,
}

BOX_LABEL_ONTOLOGY = {
    "abdomen": "fetal_abdomen",
    "arm": "fetal_arm",
    "head": "fetal_head",
    "diagnostic_plane": "diagnostic_plane",
}

_FRAME_RE = re.compile(r"frame_(\d+)$")


class FPUS23Adapter(BaseAdapter):
    DATASET_ID     = "FPUS23"
    ANATOMY_FAMILY = "fetal_planes"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "Dataset").is_dir() or (root / "Dataset_Plane").is_dir():
            return root
        if (root / "archive" / "Dataset").is_dir() or (root / "archive" / "Dataset_Plane").is_dir():
            return root / "archive"
        for name in ("FPUS23",):
            candidate = root / name
            if (candidate / "Dataset").is_dir() or (candidate / "Dataset_Plane").is_dir():
                return candidate
            archive = candidate / "archive"
            if (archive / "Dataset").is_dir() or (archive / "Dataset_Plane").is_dir():
                return archive
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'archive/Dataset' or 'archive/Dataset_Plane' under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        dataset_root = self.root / "Dataset"
        plane_root   = self.root / "Dataset_Plane"

        if not dataset_root.exists() and not plane_root.exists():
            raise FileNotFoundError(
                f"FPUS23: no Dataset/ or Dataset_Plane/ found under {self.root}"
            )

        yield from self._iter_pose_frames(dataset_root)
        yield from self._iter_plane_images(plane_root)

    # ── Dataset/four_poses ───────────────────────────────────────────────────

    def _iter_pose_frames(self, dataset_root: Path) -> Iterator[USManifestEntry]:
        poses_root = dataset_root / "four_poses"
        if not poses_root.exists():
            return

        stream_dirs = sorted(p for p in poses_root.iterdir() if p.is_dir())
        stream_split = {
            stream_dir.name: self._infer_split(stream_dir.name, i, len(stream_dirs))
            for i, stream_dir in enumerate(stream_dirs)
        }

        boxes_root = dataset_root / "boxes" / "annotation"
        annos_root = dataset_root / "annos" / "annotation"

        for stream_dir in stream_dirs:
            frames = sorted(stream_dir.glob("frame_*.png"))
            if not frames:
                continue

            boxes_xml = boxes_root / stream_dir.name / "annotations.xml"
            annos_xml = annos_root / stream_dir.name / "annotations.xml"
            box_annos = self._load_cvat_annotations(boxes_xml)
            tag_annos = self._load_cvat_annotations(annos_xml)

            for frame_idx, img_path in enumerate(frames):
                frame_name = img_path.name
                box_ann = box_annos.get(frame_name, {})
                tag_ann = tag_annos.get(frame_name, {})
                boxes = box_ann.get("boxes", [])
                tags = dict(tag_ann.get("tags", {}))
                tags.update(box_ann.get("tags", {}))
                orientation = tags.get("Orientation", {}).get("Pose")
                probe_orientation = tags.get("Probe", {}).get("orientation")
                view_fetus = tags.get("location", {}).get("View_fetus")

                if orientation is None:
                    orientation = self._pose_from_stream_name(stream_dir.name)

                instances = []
                if orientation in POSE_LABELS:
                    instances.append(
                        self._make_instance(
                            instance_id=f"{stream_dir.name}_{img_path.stem}_orientation",
                            label_raw=orientation,
                            label_ontology="fetal_pose_orientation",
                            classification_label=POSE_LABELS[orientation],
                            is_promptable=False,
                        )
                    )

                for box_i, box in enumerate(boxes):
                    label_raw = box["label"]
                    instances.append(
                        self._make_instance(
                            instance_id=f"{stream_dir.name}_{img_path.stem}_{label_raw}_{box_i}",
                            label_raw=label_raw,
                            label_ontology=BOX_LABEL_ONTOLOGY.get(label_raw, label_raw),
                            bbox_xyxy=box["bbox_xyxy"],
                            is_promptable=True,
                        )
                    )

                has_box = bool(boxes)
                task_type = "detection" if has_box else (
                    "classification" if orientation in POSE_LABELS else "ssl_only"
                )
                split = self.split_override or stream_split.get(stream_dir.name, "train")
                frame_number = self._frame_number(img_path.stem)

                yield self._make_entry(
                    str(img_path),
                    split=split,
                    modality="image",
                    instances=instances,
                    study_id=stream_dir.name,
                    series_id=stream_dir.name,
                    instance_id=f"{stream_dir.name}:{img_path.stem}",
                    frame_indices=[frame_number] if frame_number is not None else None,
                    view_type="fetal_pose_stream",
                    has_box=has_box,
                    task_type=task_type,
                    ssl_stream="image",
                    is_promptable=has_box,
                    source_meta={
                        "sub_dataset": "Dataset",
                        "stream_name": stream_dir.name,
                        "frame_name": frame_name,
                        "frame_index": frame_idx,
                        "frame_number": frame_number,
                        "orientation": orientation,
                        "probe_orientation": probe_orientation,
                        "view_fetus": view_fetus,
                        "boxes_xml": str(boxes_xml) if boxes_xml.exists() else None,
                        "annos_xml": str(annos_xml) if annos_xml.exists() else None,
                    },
                )

    @classmethod
    def _load_cvat_annotations(cls, xml_path: Path) -> Dict[str, dict]:
        if not xml_path.exists():
            return {}

        root = ET.parse(xml_path).getroot()
        out: Dict[str, dict] = {}
        for image_el in root.findall("image"):
            name = image_el.get("name")
            if not name:
                continue

            boxes = []
            for box_el in image_el.findall("box"):
                bbox = cls._box_xyxy(box_el)
                if bbox is None:
                    continue
                boxes.append({
                    "label": box_el.get("label", "unknown"),
                    "bbox_xyxy": bbox,
                })

            tags = {}
            for tag_el in image_el.findall("tag"):
                label = tag_el.get("label")
                if not label:
                    continue
                attrs = {
                    attr.get("name"): (attr.text or "").strip()
                    for attr in tag_el.findall("attribute")
                    if attr.get("name")
                }
                tags[label] = attrs

            out[name] = {"boxes": boxes, "tags": tags}
        return out

    @staticmethod
    def _box_xyxy(box_el: ET.Element) -> Optional[List[float]]:
        try:
            return [
                float(box_el.get("xtl", "")),
                float(box_el.get("ytl", "")),
                float(box_el.get("xbr", "")),
                float(box_el.get("ybr", "")),
            ]
        except ValueError:
            return None

    @staticmethod
    def _pose_from_stream_name(stream_name: str) -> Optional[str]:
        parts = stream_name.split("_")
        return parts[1] if len(parts) > 1 and parts[1] in POSE_LABELS else None

    @staticmethod
    def _frame_number(stem: str) -> Optional[int]:
        match = _FRAME_RE.match(stem)
        return int(match.group(1)) if match else None

    # ── Dataset_Plane ────────────────────────────────────────────────────────

    def _iter_plane_images(self, plane_root: Path) -> Iterator[USManifestEntry]:
        if not plane_root.exists():
            return

        samples: List[Tuple[Path, str, str, int]] = []
        for class_name, (label_ontology, class_idx) in PLANE_LABELS.items():
            class_dir = plane_root / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.glob("*.png")):
                samples.append((img_path, class_name, label_ontology, class_idx))

        n = len(samples)
        for i, (img_path, class_name, label_ontology, class_idx) in enumerate(samples):
            split = self.split_override or self._infer_split(f"{class_name}/{img_path.name}", i, n)
            instance = self._make_instance(
                instance_id=f"{class_name}_{img_path.stem}",
                label_raw=class_name,
                label_ontology=label_ontology,
                classification_label=class_idx,
                is_promptable=False,
            )

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=[instance],
                study_id=img_path.stem,
                series_id=img_path.stem,
                view_type=label_ontology,
                has_mask=False,
                task_type="classification",
                ssl_stream="image",
                is_promptable=False,
                source_meta={
                    "sub_dataset": "Dataset_Plane",
                    "plane_class": class_name,
                    "plane_label": label_ontology,
                },
            )
