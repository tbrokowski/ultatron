"""
data/adapters/lung/covidx_us.py  - COVIDx-US lung ultrasound adapter

Dataset:  COVIDx-US (COVID-US-master)
Source:   https://github.com/nrc-cnrc/COVID-US
Task:     3-class classification: COVID / Pneumonia / Normal
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)

_CLASS_LABEL: Dict[str, int] = {"COVID": 2, "Pneumonia": 1, "Normal": 0}
_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".wmv", ".gif", ".mpeg", ".mkv")


class COVIDxUSAdapter(BaseAdapter):
    DATASET_ID     = "COVIDx-US"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "silver"
    DOI            = "https://github.com/nrc-cnrc/COVID-US"

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._master_root = self.root / "COVID-US-master"
        self._utils_root  = self._master_root / "utils"
        self._data_root   = self._master_root / "data"
        self._metadata: List[dict] = []
        self._cropped:  Dict[str, str] = {}
        self._splits:   Dict[str, str] = {}
        if self._master_root.exists():
            self._load_metadata()

    def _load_metadata(self) -> None:
        meta_path = self._utils_root / "video_metadata.csv"
        crop_path = self._utils_root / "video_cropping_metadata.csv"
        if not meta_path.exists():
            log.warning("COVIDx-US: video_metadata.csv not found at %s", meta_path)
            return
        with meta_path.open() as f:
            self._metadata = list(csv.DictReader(f))
        if crop_path.exists():
            with crop_path.open() as f:
                for row in csv.DictReader(f):
                    orig = row.get("filename", "").strip()
                    crop = row.get("cropped_filename", "").strip()
                    if orig and crop:
                        self._cropped[orig] = crop
        ids   = sorted({r["id"] for r in self._metadata if r.get("id")})
        n     = len(ids)
        n_tr  = int(0.80 * n)
        n_val = int(0.10 * n)
        for i, uid in enumerate(ids):
            if self.split_override:
                self._splits[uid] = self.split_override
            elif i < n_tr:
                self._splits[uid] = "train"
            elif i < n_tr + n_val:
                self._splits[uid] = "val"
            else:
                self._splits[uid] = "test"

    def _resolve_video_path(self, uid: str) -> Optional[Path]:
        """
        Map a metadata row id (e.g. ``30_grepmed_covid``) to an on-disk video.

        Processed releases store files as ``{id}.{ext}`` under
        ``data/video/original/`` (or cropped variants under ``data/video/cropped/``).
        The human-readable ``filename`` column in video_metadata.csv is not used
        as a path component.
        """
        cropped_name = self._cropped.get(f"{uid}.mp4")
        search_dirs = (
            self._data_root / "video" / "cropped",
            self._data_root / "video" / "original",
        )
        if cropped_name:
            for d in search_dirs:
                if d.exists():
                    candidate = d / cropped_name
                    if candidate.exists():
                        return candidate
        for d in search_dirs:
            if not d.exists():
                continue
            for ext in _VIDEO_EXTS:
                candidate = d / f"{uid}{ext}"
                if candidate.exists():
                    return candidate
            matches = sorted(d.glob(f"{uid}.*"))
            if matches:
                return matches[0]
        return None

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self._metadata:
            log.warning("COVIDx-US: no metadata loaded — root may be missing or empty.")
            return
        skipped = 0
        for row in self._metadata:
            uid      = row.get("id", "").strip()
            filename = row.get("filename", "").strip()
            cls_str  = row.get("class", "").strip()
            probe    = row.get("probe", "").strip()
            if not uid or not filename:
                continue
            vpath = self._resolve_video_path(uid)
            if vpath is None:
                skipped += 1
                continue
            cls_label = _CLASS_LABEL.get(cls_str, -1)
            split     = self._splits.get(uid, "train")
            instances: List[Instance] = []
            if cls_label >= 0:
                instances.append(self._make_instance(
                    instance_id=uid, label_raw=cls_str,
                    label_ontology=cls_str.lower(), is_promptable=False,
                ))
            yield self._make_entry(
                str(vpath),
                split=split, modality="video", instances=instances,
                view_type=probe, is_cine=True, has_temporal_order=True,
                task_type="multiclass_cls" if cls_label >= 0 else "ssl_only",
                ssl_stream="both", is_promptable=False,
                source_meta={"probe": probe, "class": cls_str, "title": filename},
            )
        if skipped:
            log.warning(
                "COVIDx-US: skipped %d/%d metadata rows with no video on disk",
                skipped, len(self._metadata),
            )
