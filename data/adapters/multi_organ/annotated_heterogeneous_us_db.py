"""
data/adapters/annotated_heterogeneous_us_db.py  ·  Heterogeneous multi-organ US
===============================================================================

annotated_heterogeneous_us_db contains weakly-labelled ultrasound videos and
photo sequences. Labels are encoded in file and directory names, so this is
represented as weak-label multi-organ data rather than expert annotations.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


PATHOLOGY_VOCAB: Tuple[str, ...] = (
    "anomalies",
    "calcification",
    "cyst",
    "ectopic",
    "hernia",
    "inflammation",
    "injury",
    "nodule",
    "occupancy",
    "polyp",
    "stone",
    "tumor",
    "vascular",
)
_PATHOLOGY_TO_INDEX = {label: idx for idx, label in enumerate(PATHOLOGY_VOCAB)}


class AnnotatedHeterogeneousUSDBAdapter(BaseAdapter):
    DATASET_ID = "annotated_heterogeneous_us_db"
    ANATOMY_FAMILY = "multi"
    SONODQS = "bronze"
    DOI = ""

    def __init__(self, root, split_override=None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)
        self.filtered_root = self.root / "Filtered Data"
        self.noisy_photo_frames = self._load_noisy_photo_frames()

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "Filtered Data").exists():
            return root
        candidate = root / cls.DATASET_ID
        if (candidate / "Filtered Data").exists():
            return candidate
        raise FileNotFoundError(f"{cls.DATASET_ID}: expected 'Filtered Data/' under {root}")

    def iter_entries(self) -> Iterator[USManifestEntry]:
        video_paths = sorted(self.filtered_root.glob("Video_*.mp4"))
        photo_dirs = sorted(
            p for p in self.filtered_root.glob("Photos_*")
            if p.is_dir()
        )

        case_ids = [self._parse_name(p.stem)[0] for p in video_paths]
        case_ids.extend(self._parse_name(p.name)[0] for p in photo_dirs)
        split_map = self._group_split_map(case_ids)

        for video_path in video_paths:
            case_id, raw_labels = self._parse_name(video_path.stem)
            labels = self._normalize_labels(raw_labels)
            instances = [
                self._make_instance(
                    instance_id=f"{case_id}_{label}",
                    label_raw=label,
                    label_ontology=label,
                    is_promptable=False,
                )
                for label in labels
            ]

            yield self._make_entry(
                str(video_path),
                split=split_map.get(case_id, "train"),
                modality="video",
                instances=instances,
                study_id=case_id,
                view_type="heterogeneous_ultrasound",
                is_cine=True,
                has_temporal_order=True,
                task_type="weak_label" if labels else "ssl_only",
                ssl_stream="both",
                is_promptable=False,
                source_meta={
                    "case_id": case_id,
                    "sample_kind": "video",
                    "raw_labels": raw_labels,
                    "pathology_vector": self._label_vector(labels),
                },
            )

        for photo_dir in photo_dirs:
            case_id, raw_labels = self._parse_name(photo_dir.name)
            labels = self._normalize_labels(raw_labels)
            image_paths = self._filtered_photo_frames(photo_dir)
            if not image_paths:
                continue

            modality = "pseudo_video" if len(image_paths) > 1 else "image"
            ssl_stream = "both" if len(image_paths) > 1 else "image"
            instances = [
                self._make_instance(
                    instance_id=f"{case_id}_{label}",
                    label_raw=label,
                    label_ontology=label,
                    is_promptable=False,
                )
                for label in labels
            ]

            yield self._make_entry(
                [str(p) for p in image_paths],
                split=split_map.get(case_id, "train"),
                modality=modality,
                instances=instances,
                study_id=case_id,
                view_type="heterogeneous_ultrasound",
                has_temporal_order=len(image_paths) > 1,
                num_frames=len(image_paths),
                task_type="weak_label" if labels else "ssl_only",
                ssl_stream=ssl_stream,
                is_promptable=False,
                source_meta={
                    "case_id": case_id,
                    "sample_kind": "photo_sequence",
                    "raw_labels": raw_labels,
                    "pathology_vector": self._label_vector(labels),
                    "kept_frames": len(image_paths),
                },
            )

    def _group_split_map(self, group_ids: Sequence[str]) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }

    def _filtered_photo_frames(self, photo_dir: Path) -> List[Path]:
        kept: List[Path] = []
        for frame_path in sorted(photo_dir.glob("*.jpg")):
            frame_idx = self._frame_index(frame_path)
            if frame_idx is not None and (photo_dir.name, frame_idx) in self.noisy_photo_frames:
                continue
            kept.append(frame_path)
        return kept

    def _load_noisy_photo_frames(self) -> set[tuple[str, int]]:
        noise_path = self.root / "Processed Data" / "noise_files.txt"
        if not noise_path.exists():
            return set()

        noise_entries: set[tuple[str, int]] = set()
        for raw_line in noise_path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            folder_name = parts[-2].strip()
            frame_stem = Path(parts[-1]).stem
            if not folder_name or not frame_stem.isdigit():
                continue
            noise_entries.add((f"Photos_{folder_name}", int(frame_stem)))
        return noise_entries

    @staticmethod
    def _frame_index(frame_path: Path) -> int | None:
        match = re.search(r"(\d+)", frame_path.stem)
        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_name(name: str) -> Tuple[str, List[str]]:
        match = re.match(r"^(?:Video|Photos)_(\d+)_(.+)$", name)
        if not match:
            raise ValueError(f"Cannot parse heterogeneous sample name: {name}")
        case_id = match.group(1)
        labels = [token.strip() for token in match.group(2).split(",") if token.strip()]
        return case_id, labels

    @staticmethod
    def _normalize_labels(raw_labels: Sequence[str]) -> List[str]:
        out: List[str] = []
        for raw_label in raw_labels:
            normalized = raw_label.lower().strip().replace(" ", "_").replace("-", "_")
            normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
            if normalized in _PATHOLOGY_TO_INDEX and normalized not in out:
                out.append(normalized)
        return out

    @staticmethod
    def _label_vector(labels: Sequence[str]) -> List[float]:
        vec = [0.0] * len(PATHOLOGY_VOCAB)
        for label in labels:
            idx = _PATHOLOGY_TO_INDEX.get(label)
            if idx is not None:
                vec[idx] = 1.0
        return vec
