"""
data/adapters/lung/lus_data.py  ·  LUS-data (COVIDx-US processed output) adapter
==================================================================================

Processed COVIDx-US release stored under lung/data/ on Capstor:

    {root}/
        image/original/   {id}_{source}_{condition}_{probe}_frame{N}.jpg
        video/original/   {id}_{source}_{condition}.{mp4|gif|wmv|...}
        mask/             {id}_{source}_{condition}_prc_{probe}_frame{N}_mask.jpg

Sources include litfl, grepmed, pocusatlas, butterfly, uf, core, clarius, paper.
Conditions: covid, pneumonia, normal, other.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)

_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".wmv", ".gif", ".mpeg", ".mkv"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

_FRAME_SUFFIX_RE = re.compile(r"^(.+)_frame(\d+)$", re.IGNORECASE)
_VIDEO_RE = re.compile(
    r"^(\d+)_(.+?)_(covid|pneumonia|normal|other)(?:_\d+)?$", re.IGNORECASE
)

_CONDITION_ONTOLOGY = {
    "covid": "lung_covid",
    "pneumonia": "lung_pneumonia",
    "normal": "lung_normal",
    "other": "lung_other",
}


def _parse_frame_stem(stem: str) -> Tuple[str, str, str, str, str]:
    """Return uid, source, condition, probe, frame_idx."""
    m = _FRAME_SUFFIX_RE.match(stem)
    if not m:
        return stem, "unknown", "other", "unknown", "0"
    base, frame_idx = m.group(1), m.group(2)
    probe = base.rsplit("_", 1)[-1]
    rest = base.rsplit("_", 1)[0]
    condition = rest.rsplit("_", 1)[-1]
    rest2 = rest.rsplit("_", 1)[0]
    source = rest2.rsplit("_", 1)[-1]
    uid = rest2.rsplit("_", 1)[0]
    return uid, source, condition.lower(), probe.lower(), frame_idx


def _mask_path_for_image(root: Path, img_path: Path) -> Optional[Path]:
    """Map frame image stem to COVIDx-US mask filename."""
    mask_stem = img_path.stem.replace("_prc_", "_") + "_mask"
    mask_path = root / "mask" / f"{mask_stem}.jpg"
    return mask_path if mask_path.exists() else None


def _parse_video_meta(stem: str) -> Tuple[str, str, str]:
    m = _VIDEO_RE.match(stem)
    if not m:
        parts = stem.split("_", 2)
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2].lower()
        return stem, "unknown", "other"
    return m.group(1), m.group(2).lower(), m.group(3).lower()


class LUSDataAdapter(BaseAdapter):
    DATASET_ID = "LUS-data"
    ANATOMY_FAMILY = "lung"
    SONODQS = "silver"
    DOI = "https://github.com/nrc-cnrc/COVID-US"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_root = self.root / "image" / "original"
        vid_root = self.root / "video" / "original"
        if not img_root.exists() and not vid_root.exists():
            log.warning("LUS-data: expected image/original or video/original under %s", self.root)
            return

        emitted = 0

        if img_root.exists():
            images = sorted(
                p for p in img_root.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            )
            for idx, img_path in enumerate(images):
                uid, source, condition, probe, frame_idx = _parse_frame_stem(img_path.stem)
                split = self._infer_split(uid, idx, len(images))
                mask_path = _mask_path_for_image(self.root, img_path)
                instances: List[Instance] = []
                if condition in _CONDITION_ONTOLOGY:
                    instances.append(
                        self._make_instance(
                            instance_id=f"{uid}_{condition}",
                            label_raw=condition,
                            label_ontology=_CONDITION_ONTOLOGY[condition],
                            is_promptable=False,
                        )
                    )
                if mask_path is not None:
                    instances.append(
                        self._make_instance(
                            instance_id=f"{img_path.stem}_lung_region",
                            label_raw="lung_region",
                            label_ontology="lung_region",
                            mask_path=str(mask_path),
                            is_promptable=True,
                        )
                    )
                yield self._make_entry(
                    str(img_path),
                    split=split,
                    modality="image",
                    instances=instances,
                    view_type=probe,
                    has_mask=mask_path is not None,
                    task_type="segmentation" if mask_path else (
                        "multiclass_cls" if instances else "ssl_only"
                    ),
                    ssl_stream="image",
                    is_promptable=mask_path is not None,
                    source_meta={
                        "video_uid": uid,
                        "source": source,
                        "condition": condition,
                        "probe": probe,
                        "frame_idx": frame_idx,
                    },
                )
                emitted += 1

        if vid_root.exists():
            videos = sorted(
                p for p in vid_root.iterdir()
                if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
            )
            for idx, vid_path in enumerate(videos):
                uid, source, condition = _parse_video_meta(vid_path.stem)
                split = self._infer_split(uid, idx, len(videos))
                instances: List[Instance] = []
                if condition in _CONDITION_ONTOLOGY:
                    instances.append(
                        self._make_instance(
                            instance_id=f"{uid}_{condition}",
                            label_raw=condition,
                            label_ontology=_CONDITION_ONTOLOGY[condition],
                            is_promptable=False,
                        )
                    )
                yield self._make_entry(
                    str(vid_path),
                    split=split,
                    modality="video",
                    instances=instances,
                    is_cine=True,
                    has_temporal_order=True,
                    task_type="multiclass_cls" if instances else "ssl_only",
                    ssl_stream="video",
                    is_promptable=False,
                    source_meta={
                        "video_uid": uid,
                        "source": source,
                        "condition": condition,
                    },
                )
                emitted += 1

        log.info("LUS-data: emitted %d entries", emitted)
