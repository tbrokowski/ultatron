"""
data/adapters/maternal_fetal/pbf_us1.py  ·  PBF-US1 adapter
=============================================================

PBF-US1 (NatalIA Phantom Blind-sweep Fetal Ultrasound):
19,407 JPEG frames from 90 freehand sweep exams of a 23-week phantom,
labelled per-frame with a 6-class fetal standard plane taxonomy.

Task
----
Per-frame **fetal standard plane classification** (6 classes).  Frames were
extracted from blind freehand sweeps recorded by non-expert volunteers on a
23-week phantom (Clarius C3 HD3, 24 fps, 16 cm depth).

Layout on disk:
  {root}/
  ├── Obstetrics Exam - <DD-Mon-YYYY>_<HH>_<AM/PM>/   # 90 folders, names with spaces
  │   ├── cineframe_<N>_<ISO_timestamp>.jpeg          # extracted frames
  │   └── <optional sweep video>.{mp4,avi,...}        # if present on disk
  ├── resume.csv    — per-frame labels  (19,407 rows)
  └── metadata.csv  — per-exam metadata (90 rows)

resume.csv (comma-delimited):
  file_name   JPEG filename (no path)
  studie      Exam folder name — join key to locate file
  class       Human-readable plane label
  value       Integer class 0–5
  image       Empty column (ignored)

metadata.csv (comma-delimited):
  Study Name  Exam folder name (join key)
  protocol    Sweep type: Vertical / Horizontal / Diagonal / Diagonal \\
  position    Fetal pose: OP / SP / OA / SA
  (+ volunteer demographics and ultrasound experience fields)

Class taxonomy (pbf_us1_planes):
  0  Biparietal standard plane  (   42 frames)
  1  Abdominal standard plane   (   63 frames)
  2  Heart standard plane       (   61 frames)
  3  Spine standard plane       (  134 frames)
  4  Femur standard plane       (   46 frames)
  5  No plane                   (19,061 frames — 98.2 %)

Manifest:
  * One image entry per labelled frame (classification / image SSL stream).
  * One sweep entry per exam (video SSL + paired stream, ssl_stream="both").
    Native video files are used when present; otherwise frames are emitted as a
    temporally ordered pseudo_video sequence.  Per-frame labels are preserved in
    source_meta and as instances on standard-plane frames (classes 0–4).

Split strategy:
  No predefined split. Assign by exam folder (studie) to prevent temporal
  leakage within sweeps. Sorted studie names → 80/10/10 by index.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

PLANE_CLASSES: List[str] = [
    "Biparietal standard plane",  # 0
    "Abdominal standard plane",   # 1
    "Heart standard plane",       # 2
    "Spine standard plane",       # 3
    "Femur standard plane",       # 4
    "No plane",                   # 5
]

PLANE_VIEW_TYPES: List[str] = [
    "biparietal",
    "abdominal",
    "heart",
    "spine",
    "femur",
    "no_plane",
]

_NO_PLANE_IDX = 5
_SWEEP_FPS = 24.0

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}
_FRAME_TS_RE = re.compile(
    r"^cineframe_\d+_(?P<ts>.+)\.(?:jpe?g|jpg)$",
    re.IGNORECASE,
)

_EXAM_META_KEYS = (
    "Age",
    "Gender",
    "Level of Education",
    "Ultrasound Experience",
    "Years of Experience",
    "Race/Ethnicity",
    "Visual Impairment",
    "specify",
    "dominant hand",
)


@dataclass(frozen=True)
class _FrameLabel:
    path: Path
    class_idx: int
    class_name: str


def _frame_sort_key(path: Path) -> Tuple[int, str]:
    match = _FRAME_TS_RE.match(path.name)
    if match:
        return (0, match.group("ts"))
    return (1, path.name)


def _find_exam_video(exam_dir: Path) -> Optional[Path]:
    if not exam_dir.is_dir():
        return None
    videos = sorted(
        p for p in exam_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
    )
    return videos[0] if videos else None


def _clean_field(value: object) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


class PBFUS1Adapter(BaseAdapter):
    DATASET_ID     = "PBF-US1"
    ANATOMY_FAMILY = "fetal_planes"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.5281/zenodo.14193949"

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "resume.csv").exists():
            return root
        candidate = root / "PBF-US1"
        if (candidate / "resume.csv").exists():
            return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected resume.csv under {root}"
        )

    def _load_resume(self) -> List[dict]:
        path = self.root / "resume.csv"
        with path.open(encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))

    def _load_metadata(self) -> Dict[str, dict]:
        path = self.root / "metadata.csv"
        if not path.exists():
            return {}
        with path.open(encoding="utf-8-sig") as f:
            return {row["Study Name"].strip(): row for row in csv.DictReader(f)}

    def _exam_source_meta(self, studie: str, meta: Dict[str, dict]) -> dict:
        exam_meta = meta.get(studie, {})
        out = {
            "studie":   studie,
            "protocol": _clean_field(exam_meta.get("protocol")),
            "position": _clean_field(exam_meta.get("position")),
        }
        for key in _EXAM_META_KEYS:
            value = _clean_field(exam_meta.get(key))
            if value is not None:
                out[key.lower().replace(" ", "_")] = value
        return out

    @staticmethod
    def _view_type(class_idx: int) -> str:
        if 0 <= class_idx < len(PLANE_VIEW_TYPES):
            return PLANE_VIEW_TYPES[class_idx]
        return "fetal_plane_unknown"

    @staticmethod
    def _build_sweep_frame_labels(
        ordered_paths: List[Path],
        labels_by_path: Dict[Path, _FrameLabel],
    ) -> List[dict]:
        frame_labels: List[dict] = []
        for frame_idx, path in enumerate(ordered_paths):
            label = labels_by_path.get(path)
            if label is None:
                continue
            frame_labels.append({
                "frame_index": frame_idx,
                "file_name":   path.name,
                "class_idx":   label.class_idx,
                "class_name":  label.class_name,
            })
        return frame_labels

    @staticmethod
    def _plane_index_map(frame_labels: List[dict]) -> Dict[int, List[int]]:
        by_plane: Dict[int, List[int]] = {i: [] for i in range(len(PLANE_CLASSES))}
        for item in frame_labels:
            by_plane[item["class_idx"]].append(item["frame_index"])
        return by_plane

    def _sweep_instances(
        self,
        studie: str,
        frame_labels: List[dict],
    ) -> list:
        instances = []
        for item in frame_labels:
            cls_idx = item["class_idx"]
            if cls_idx == _NO_PLANE_IDX:
                continue
            stem = Path(item["file_name"]).stem
            instances.append(
                self._make_instance(
                    instance_id          = f"{studie}_{stem}",
                    label_raw            = item["class_name"],
                    label_ontology       = "pbf_us1_planes",
                    is_promptable        = False,
                    classification_label = cls_idx,
                )
            )
        return instances

    def iter_entries(self) -> Iterator[USManifestEntry]:
        rows = self._load_resume()
        meta = self._load_metadata()

        studies: List[str] = sorted({
            (r.get("studie") or "").strip()
            for r in rows
            if (r.get("studie") or "").strip()
        })
        n = len(studies)
        study_split: Dict[str, str] = {
            s: self._infer_split(s, i, n)
            for i, s in enumerate(studies)
        }

        study_labels_by_path: Dict[str, Dict[Path, _FrameLabel]] = {
            s: {} for s in studies
        }

        for row in rows:
            studie    = (row.get("studie")    or "").strip()
            file_name = (row.get("file_name") or "").strip()
            cls_name  = (row.get("class")     or "").strip()
            value_str = (row.get("value")     or "").strip()

            if not studie or not file_name:
                continue

            img_path = self.root / studie / file_name
            if not img_path.exists():
                continue

            try:
                cls_idx = int(value_str)
            except (ValueError, TypeError):
                continue

            if cls_idx < 0 or cls_idx >= len(PLANE_CLASSES):
                continue

            label = _FrameLabel(
                path       = img_path,
                class_idx  = cls_idx,
                class_name = cls_name or PLANE_CLASSES[cls_idx],
            )
            study_labels_by_path.setdefault(studie, {})[img_path] = label

            split      = self.split_override or study_split.get(studie, "train")
            stem       = img_path.stem
            exam_meta  = self._exam_source_meta(studie, meta)

            instance = self._make_instance(
                instance_id          = stem,
                label_raw            = label.class_name,
                label_ontology       = "pbf_us1_planes",
                is_promptable        = False,
                classification_label = cls_idx,
            )

            yield self._make_entry(
                str(img_path),
                split         = split,
                modality      = "image",
                instances     = [instance],
                study_id      = studie,
                series_id     = stem,
                view_type     = self._view_type(cls_idx),
                height        = 500,
                width         = 700,
                has_mask      = False,
                task_type     = "classification",
                ssl_stream    = "image",
                is_promptable = False,
                source_meta   = {
                    **exam_meta,
                    "file_name":  file_name,
                    "class_idx":  cls_idx,
                    "class_name": label.class_name,
                },
            )

        for studie in studies:
            labels_by_path = study_labels_by_path.get(studie) or {}
            if not labels_by_path:
                continue

            split      = self.split_override or study_split.get(studie, "train")
            exam_dir   = self.root / studie
            exam_meta  = self._exam_source_meta(studie, meta)
            video_path = _find_exam_video(exam_dir)

            ordered_paths = sorted(labels_by_path.keys(), key=_frame_sort_key)
            frame_labels  = self._build_sweep_frame_labels(ordered_paths, labels_by_path)
            plane_indices = self._plane_index_map(frame_labels)
            instances = self._sweep_instances(studie, frame_labels)
            standard_plane_indices = [
                item["frame_index"]
                for item in frame_labels
                if item["class_idx"] != _NO_PLANE_IDX
            ]

            sweep_meta = {
                **exam_meta,
                "frame_labels": frame_labels,
                "plane_frame_indices": {
                    PLANE_CLASSES[i]: plane_indices[i]
                    for i in range(len(PLANE_CLASSES))
                    if plane_indices[i]
                },
                "standard_plane_frame_indices": standard_plane_indices,
                "n_standard_plane_frames": len(standard_plane_indices),
                "n_no_plane_frames": len(plane_indices.get(_NO_PLANE_IDX, [])),
            }

            if video_path is not None:
                yield self._make_entry(
                    str(video_path),
                    split              = split,
                    modality           = "video",
                    instances          = instances,
                    study_id           = studie,
                    series_id          = studie,
                    view_type          = "fetal_standard_plane_sweep",
                    height             = 500,
                    width              = 700,
                    fps                = _SWEEP_FPS,
                    is_cine            = True,
                    has_temporal_order = True,
                    task_type          = "classification",
                    ssl_stream         = "both",
                    is_promptable      = False,
                    source_meta        = {
                        **sweep_meta,
                        "video_source": "native",
                    },
                )
                continue

            if len(ordered_paths) < 2:
                continue

            yield self._make_entry(
                [str(p) for p in ordered_paths],
                split              = split,
                modality           = "pseudo_video",
                instances          = instances,
                study_id           = studie,
                series_id          = studie,
                view_type          = "fetal_standard_plane_sweep",
                height             = 500,
                width              = 700,
                num_frames         = len(ordered_paths),
                fps                = _SWEEP_FPS,
                is_cine            = True,
                has_temporal_order = True,
                task_type          = "classification",
                ssl_stream         = "both",
                is_promptable      = False,
                source_meta        = {
                    **sweep_meta,
                    "video_source": "extracted_frames",
                    "n_frames": len(ordered_paths),
                },
            )
