"""
data/adapters/lung/luss_phantom.py  - LUSS PHANTOM lung ultrasound adapter

Dataset:  LUSS PHANTOM / LUCPD (Howell & McLaughlan 2024)
Source:   https://doi.org/10.5518/1485
Paper:    Deep learning for real-time multi-class segmentation of artefacts
          in lung ultrasound (Ultrasonics 140, 2024)

Layout::

    {root}/
        data-3/
            train/images/   *.png
            train/masks/    *.png   integer class map {0..5}
            test/images/
            test/masks/
        VideoClips/         297 × 2 s cineloops (.webm) used to generate frames
        vidtoImage.m        Matlab script for frame extraction

Mask encoding (single-channel PNG, values 0–5):
    0 = background
    1 = rib
    2 = pleural line
    3 = A-line
    4 = B-line
    5 = B-line confluence

Tasks:
  - Multi-class semantic segmentation (primary task from the paper)
  - Per-structure binary segmentation (one Instance per class, mask_channel set)
  - Frame-level multilabel finding presence (derived from mask pixels)
  - Video SSL from raw cineloops (VideoClips/)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from data.adapters.base import BaseAdapter
from data.adapters.lung.benin_lus import LUS_CANONICAL_FINDINGS
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)

_VIDEO_EXTS = (".webm", ".mp4", ".avi", ".mov", ".mkv")

# mask_channel = integer label value in the shared mask PNG
SEG_STRUCTURES: List[Tuple[str, str, int]] = [
    ("rib",              "rib",              1),
    ("pleural_line",     "pleural_line",     2),
    ("a_line",           "a_line",           3),
    ("b_line",           "b_line",           4),
    ("confluent_b_line", "confluent_b_line", 5),
]

SEG_CLASS_NAMES: List[str] = ["background"] + [s[0] for s in SEG_STRUCTURES]

# Map segmentation class → index in LUS_CANONICAL_FINDINGS (for multilabel vector)
_FINDING_TO_CANONICAL: Dict[str, int] = {
    "a_line":           LUS_CANONICAL_FINDINGS.index("a_line"),
    "b_line":           LUS_CANONICAL_FINDINGS.index("b_line"),
    "confluent_b_line": LUS_CANONICAL_FINDINGS.index("confluent_b_line"),
}


class LUSSPhantomAdapter(BaseAdapter):
    DATASET_ID     = "LUSS-PHANTOM"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.5518/1485"

    _SPLIT_DIRS = {"train": "train", "test": "test"}

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._clip_paths: Dict[str, Path] = self._index_video_clips()
        self._clip_splits: Dict[str, str] = self._build_clip_splits()

    def iter_entries(self) -> Iterator[USManifestEntry]:
        data_root = self.root / "data-3"
        if not data_root.is_dir():
            data_root = self.root

        n_img, n_vid = 0, 0
        for entry in self._iter_labeled_images(data_root):
            n_img += 1
            yield entry
        for entry in self._iter_video_clips():
            n_vid += 1
            yield entry

        log.info("LUSS-PHANTOM: emitted %d labeled images, %d video clips", n_img, n_vid)

    def _index_video_clips(self) -> Dict[str, Path]:
        clips_dir = self.root / "VideoClips"
        if not clips_dir.is_dir():
            return {}
        index: Dict[str, Path] = {}
        for ext in _VIDEO_EXTS:
            for vpath in clips_dir.glob(f"*{ext}"):
                index[vpath.stem] = vpath
        return index

    def _build_clip_splits(self) -> Dict[str, str]:
        """Infer clip split from labeled frames; unlabeled clips default to train."""
        splits: Dict[str, str] = {}
        data_root = self.root / "data-3"
        if not data_root.is_dir():
            data_root = self.root

        for split_name, split_label in self._SPLIT_DIRS.items():
            img_dir = data_root / split_name / "images"
            if not img_dir.is_dir():
                continue
            for img_path in img_dir.glob("*.png"):
                clip_stem = self._clip_stem(img_path.stem)
                splits.setdefault(clip_stem, split_label)

        unlabeled = sorted(set(self._clip_paths) - set(splits))
        for i, clip_stem in enumerate(unlabeled):
            splits[clip_stem] = self.split_override or self._infer_split(
                clip_stem, i, len(unlabeled)
            )
        return splits

    @staticmethod
    def _clip_stem(frame_stem: str) -> str:
        if "-F" in frame_stem:
            return frame_stem.rsplit("-F", 1)[0]
        return frame_stem

    def _iter_labeled_images(self, data_root: Path) -> Iterator[USManifestEntry]:
        for split_name, split_label in self._SPLIT_DIRS.items():
            img_dir = data_root / split_name / "images"
            msk_dir = data_root / split_name / "masks"
            if not img_dir.is_dir():
                continue

            for img_path in sorted(img_dir.glob("*.png")):
                mask_path = msk_dir / img_path.name
                if not mask_path.exists():
                    continue

                split = self.split_override or split_label
                clip_stem = self._clip_stem(img_path.stem)
                clip_path = self._clip_paths.get(clip_stem)
                present_classes = self._present_classes(mask_path)
                instances = self._make_seg_instances(img_path.stem, mask_path, present_classes)
                finding_presence, video_labels = self._finding_labels(present_classes)

                yield self._make_entry(
                    str(img_path),
                    split=split,
                    modality="image",
                    instances=instances,
                    study_id=clip_stem,
                    series_id=img_path.stem,
                    has_mask=True,
                    task_type="segmentation",
                    ssl_stream="both" if clip_path is not None else "image",
                    is_promptable=True,
                    source_meta={
                        "mask_encoding":    "multiclass",
                        "seg_class_names":  SEG_CLASS_NAMES,
                        "num_seg_classes":  len(SEG_CLASS_NAMES),
                        "present_classes":  [
                            s[0] for s in SEG_STRUCTURES if s[2] in present_classes
                        ],
                        "finding_presence": finding_presence,
                        "video_labels":     video_labels,
                        "frame_stem":       img_path.stem,
                        "clip_stem":        clip_stem,
                        "video_path":       str(clip_path) if clip_path else None,
                    },
                )

    def _iter_video_clips(self) -> Iterator[USManifestEntry]:
        clips = sorted(self._clip_paths.items())
        labeled_clips: Set[str] = set()

        data_root = self.root / "data-3"
        if data_root.is_dir():
            for split_name in self._SPLIT_DIRS:
                img_dir = data_root / split_name / "images"
                if img_dir.is_dir():
                    for img_path in img_dir.glob("*.png"):
                        labeled_clips.add(self._clip_stem(img_path.stem))

        for clip_stem, vpath in clips:
            split = self.split_override or self._clip_splits.get(clip_stem, "train")
            has_labeled_frames = clip_stem in labeled_clips

            yield self._make_entry(
                str(vpath),
                split=split,
                modality="video",
                study_id=clip_stem,
                series_id=clip_stem,
                view_type="lung_bmode",
                is_cine=True,
                has_temporal_order=True,
                task_type="ssl_only",
                ssl_stream="both",
                is_promptable=False,
                source_meta={
                    "clip_stem":          clip_stem,
                    "has_labeled_frames": has_labeled_frames,
                    "duration_s":         2.0,
                },
            )

    def _make_seg_instances(
        self,
        stem: str,
        mask_path: Path,
        present_classes: set[int],
    ) -> List[Instance]:
        return [
            self._make_instance(
                instance_id=stem,
                label_raw="luss_multiclass",
                label_ontology="lung",
                mask_path=str(mask_path),
                is_promptable=True,
            ),
            *[
                self._make_instance(
                    instance_id=f"{stem}_{struct_name}",
                    label_raw=struct_name,
                    label_ontology=label_ontology,
                    mask_path=str(mask_path),
                    mask_channel=class_label,
                    is_promptable=True,
                )
                for struct_name, label_ontology, class_label in SEG_STRUCTURES
                if class_label in present_classes
            ],
        ]

    @staticmethod
    def _present_classes(mask_path: Path) -> set[int]:
        arr = np.array(Image.open(mask_path))
        return {int(v) for v in np.unique(arr) if int(v) > 0}

    @staticmethod
    def _finding_labels(
        present_classes: set[int],
    ) -> Tuple[Dict[str, int], List[float]]:
        """Build per-frame LUS finding flags for the 3 artefact classes."""
        finding_presence = {
            name: int(class_label in present_classes)
            for name, _, class_label in SEG_STRUCTURES
            if name in _FINDING_TO_CANONICAL
        }
        video_labels = [0.0] * len(LUS_CANONICAL_FINDINGS)
        for name, idx in _FINDING_TO_CANONICAL.items():
            class_label = next(cl for n, _, cl in SEG_STRUCTURES if n == name)
            video_labels[idx] = float(class_label in present_classes)
        return finding_presence, video_labels
