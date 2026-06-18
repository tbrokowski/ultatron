"""
data/adapters/lung/covidx_us.py  - COVIDx-US lung ultrasound adapter

Dataset:  COVIDx-US (COVID-US-master + processed data/)
Source:   https://github.com/nrc-cnrc/COVID-US

Layout (after consolidation)::

    {root}/
        COVID-US-master/
            utils/video_metadata.csv
            utils/video_cropping_metadata.csv
            data -> ../data          # symlink
        data/
            image/original/   {id}_{source}_{condition}_{probe}_frame{N}.jpg
            video/original/   {id}_{source}_{condition}.{mp4|gif|...}
            mask/             {id}_{source}_{condition}_prc_{probe}_frame{N}_mask.jpg

Tasks:
  - 3-class video classification: COVID / Pneumonia / Normal
  - Video-level LUS finding multilabels (a_line, b_line, confluent_b_line, ...)
  - Frame-level segmentation masks where available
"""
from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

from data.adapters.base import BaseAdapter
from data.adapters.lung.benin_lus import LUS_CANONICAL_FINDINGS
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)

_CLASS_LABEL: Dict[str, int] = {"COVID": 2, "Pneumonia": 1, "Normal": 0}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".wmv", ".gif", ".mpeg", ".mkv"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

_CONDITION_ONTOLOGY = {
    "covid": "lung_covid",
    "pneumonia": "lung_pneumonia",
    "normal": "lung_normal",
    "other": "lung_other",
}

_FRAME_SUFFIX_RE = re.compile(r"^(.+)_frame(\d+)$", re.IGNORECASE)
_VIDEO_RE = re.compile(
    r"^(\d+)_(.+?)_(covid|pneumonia|normal|other)(?:_\d+)?$", re.IGNORECASE
)


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


def _parse_video_stem(stem: str) -> Tuple[str, str, str]:
    m = _VIDEO_RE.match(stem)
    if not m:
        parts = stem.split("_", 2)
        if len(parts) >= 3:
            return stem, parts[1], parts[2].lower()
        return stem, "unknown", "other"
    uid = f"{m.group(1)}_{m.group(2)}_{m.group(3)}"
    return uid, m.group(2).lower(), m.group(3).lower()


def _mask_path_for_image(data_root: Path, img_path: Path) -> Optional[Path]:
    """Map a frame image to its lung-region mask (handles _prc_ naming)."""
    stem = img_path.stem
    candidates: List[str] = [f"{stem}_mask"]
    m = _FRAME_SUFFIX_RE.match(stem)
    if m:
        base, frame_idx = m.group(1), m.group(2)
        if "_prc_" in base:
            candidates.append(f"{base}_frame{frame_idx}_mask")
        else:
            probe = base.rsplit("_", 1)[-1]
            prefix = base.rsplit("_", 1)[0]
            candidates.append(f"{prefix}_prc_{probe}_frame{frame_idx}_mask")
    for mask_stem in candidates:
        mask_path = data_root / "mask" / f"{mask_stem}.jpg"
        if mask_path.exists():
            return mask_path
    return None


def _infer_video_findings(title: str, folder: str = "") -> List[float]:
    """
    Map a human-readable video title / folder to the 7-dim LUS finding vector
    (same vocabulary as Benin-LUS / RSA-LUS).
    """
    text = f"{title} {folder}".lower().replace("\\", "/")
    vec = [0.0] * len(LUS_CANONICAL_FINDINGS)

    def _set(name: str) -> None:
        vec[LUS_CANONICAL_FINDINGS.index(name)] = 1.0

    if any(k in text for k in ("a line", "a-line", "aline", "a lines", "normal lung")):
        _set("a_line")
    if any(k in text for k in ("confluent", "coalescent", "coalescing")):
        _set("confluent_b_line")
        _set("b_line")
    elif any(k in text for k in ("b line", "b-line", "bline", "blines", "b lines")):
        _set("b_line")
    if "effusion" in text:
        _set("pleural_effusion")
    if any(k in text for k in ("consolidation", "hepatization", "bronchogram", "shred sign")):
        if any(k in text for k in ("subpleural", "small", "skip lesion")):
            _set("small_consolidation")
        else:
            _set("large_consolidation")
    if "pneumothorax" in text:
        _set("pneumothorax")

    # Butterfly folder categories
    if "/b lines" in text or text.endswith("b lines"):
        _set("b_line")
    if "/consolidation" in text:
        _set("large_consolidation")
    if "/normal lung" in text:
        _set("a_line")
    if "/irregular pleural" in text:
        _set("b_line")

    return vec


def _condition_from_class(cls_str: str) -> str:
    return {"COVID": "covid", "Pneumonia": "pneumonia", "Normal": "normal"}.get(
        cls_str.strip(), "other"
    )


class COVIDxUSAdapter(BaseAdapter):
    DATASET_ID     = "COVIDx-US"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "silver"
    DOI            = "https://github.com/nrc-cnrc/COVID-US"

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._master_root = self._resolve_master_root()
        self._utils_root  = self._master_root / "utils"
        self._data_root   = self._resolve_data_root()
        self._metadata_by_id: Dict[str, dict] = {}
        self._cropped: Dict[str, str] = {}
        self._splits: Dict[str, str] = {}
        self._load_metadata()

    def _resolve_master_root(self) -> Path:
        candidate = self.root / "COVID-US-master"
        if candidate.exists():
            return candidate
        if (self.root / "utils" / "video_metadata.csv").exists():
            return self.root
        return candidate

    def _resolve_data_root(self) -> Path:
        for candidate in (
            self.root / "data",
            self.root / "COVID-US-master" / "data",
        ):
            if candidate.exists():
                return candidate.resolve()
        return self.root / "data"

    def _load_metadata(self) -> None:
        meta_path = self._utils_root / "video_metadata.csv"
        crop_path = self._utils_root / "video_cropping_metadata.csv"
        if not meta_path.exists():
            log.warning("COVIDx-US: video_metadata.csv not found at %s", meta_path)
            return
        with meta_path.open() as f:
            rows = list(csv.DictReader(f))
        self._metadata_by_id = {r["id"].strip(): r for r in rows if r.get("id")}
        if crop_path.exists():
            with crop_path.open() as f:
                for row in csv.DictReader(f):
                    orig = row.get("filename", "").strip()
                    crop = row.get("cropped_filename", "").strip()
                    if orig and crop:
                        self._cropped[orig] = crop
        ids = sorted(self._metadata_by_id)
        n = len(ids)
        n_tr = int(0.80 * n)
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

    def _lookup_metadata(self, stem_or_uid: str) -> Optional[dict]:
        if stem_or_uid in self._metadata_by_id:
            return self._metadata_by_id[stem_or_uid]
        for meta_id, row in self._metadata_by_id.items():
            if stem_or_uid.startswith(meta_id + "_") or stem_or_uid == meta_id:
                return row
        uid, source, condition, _, _ = _parse_frame_stem(
            stem_or_uid if "_frame" in stem_or_uid else f"{stem_or_uid}_frame0"
        )
        composite = f"{uid}_{source}_{condition}"
        return self._metadata_by_id.get(composite)

    def _split_for_uid(self, uid: str, idx: int, total: int) -> str:
        if uid in self._splits:
            return self._splits[uid]
        return self._infer_split(uid, idx, total)

    def _build_finding_instances(
        self, video_labels: List[float], prefix: str
    ) -> List[Instance]:
        instances: List[Instance] = []
        for idx, val in enumerate(video_labels):
            if val <= 0.0:
                continue
            finding = LUS_CANONICAL_FINDINGS[idx]
            instances.append(
                self._make_instance(
                    instance_id=f"{prefix}_{finding}",
                    label_raw=finding,
                    label_ontology=finding,
                    is_promptable=False,
                )
            )
        return instances

    def _class_instance(self, uid: str, condition: str, cls_str: str = "") -> Optional[Instance]:
        label_raw = cls_str or condition
        ontology = _CONDITION_ONTOLOGY.get(condition)
        if not ontology:
            return None
        return self._make_instance(
            instance_id=f"{uid}_{condition}",
            label_raw=label_raw,
            label_ontology=ontology,
            is_promptable=False,
        )

    def _emit_video_entry(
        self,
        vpath: Path,
        uid: str,
        split: str,
        *,
        meta_row: Optional[dict] = None,
        condition: Optional[str] = None,
        source: str = "unknown",
        probe: str = "",
    ) -> USManifestEntry:
        title = (meta_row or {}).get("filename", vpath.stem)
        folder = (meta_row or {}).get("folder", "")
        cls_str = (meta_row or {}).get("class", "")
        if condition is None:
            condition = _condition_from_class(cls_str) if cls_str else "other"
        if not probe and meta_row:
            probe = meta_row.get("probe", "")

        video_labels = _infer_video_findings(title, folder)
        if not any(video_labels):
            slug_labels = _infer_video_findings(
                vpath.stem.replace("_", " ").replace("-", " ")
            )
            video_labels = [
                max(a, b) for a, b in zip(video_labels, slug_labels)
            ]
        instances: List[Instance] = []
        cls_inst = self._class_instance(uid, condition, cls_str)
        if cls_inst:
            instances.append(cls_inst)
        instances.extend(self._build_finding_instances(video_labels, uid))

        has_findings = any(v > 0 for v in video_labels)
        has_class = cls_inst is not None
        if has_findings:
            task_type = "multilabel_cls"
        elif has_class:
            task_type = "multiclass_cls"
        else:
            task_type = "ssl_only"

        return self._make_entry(
            str(vpath),
            split=split,
            modality="video",
            instances=instances,
            view_type=probe,
            is_cine=True,
            has_temporal_order=True,
            task_type=task_type,
            ssl_stream="both",
            is_promptable=has_findings,
            source_meta={
                "video_uid": uid,
                "source": source,
                "condition": condition,
                "probe": probe,
                "class": cls_str,
                "title": title,
                "video_labels": video_labels,
            },
        )

    def _iter_video_entries(self) -> Iterator[USManifestEntry]:
        emitted_paths: Set[str] = set()
        skipped_meta = 0

        meta_ids = sorted(self._metadata_by_id)
        for i, uid in enumerate(meta_ids):
            row = self._metadata_by_id[uid]
            vpath = self._resolve_video_path(uid)
            if vpath is None:
                skipped_meta += 1
                continue
            split = self._split_for_uid(uid, i, len(meta_ids))
            condition = _condition_from_class(row.get("class", ""))
            source = row.get("source", "unknown").strip().lower()
            yield self._emit_video_entry(
                vpath, uid, split,
                meta_row=row,
                condition=condition,
                source=source,
                probe=row.get("probe", "").strip(),
            )
            emitted_paths.add(str(vpath.resolve()))

        if skipped_meta:
            log.warning(
                "COVIDx-US: %d/%d metadata videos missing on disk",
                skipped_meta, len(meta_ids),
            )

        vid_root = self._data_root / "video" / "original"
        if not vid_root.exists():
            return
        extra_videos = sorted(
            p for p in vid_root.iterdir()
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTS
            and str(p.resolve()) not in emitted_paths
        )
        for idx, vpath in enumerate(extra_videos):
            uid, source, condition = _parse_video_stem(vpath.stem)
            split = self._split_for_uid(uid, idx, len(extra_videos))
            yield self._emit_video_entry(
                vpath, uid, split,
                condition=condition,
                source=source,
            )

    def _iter_image_entries(self) -> Iterator[USManifestEntry]:
        img_root = self._data_root / "image" / "original"
        if not img_root.exists():
            return
        images = sorted(
            p for p in img_root.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        )
        for idx, img_path in enumerate(images):
            uid, source, condition, probe, frame_idx = _parse_frame_stem(img_path.stem)
            split = self._split_for_uid(uid, idx, len(images))
            mask_path = _mask_path_for_image(self._data_root, img_path)

            meta_row = self._lookup_metadata(img_path.stem)
            title = meta_row.get("filename", "") if meta_row else img_path.stem
            folder = meta_row.get("folder", "") if meta_row else ""
            cls_str = meta_row.get("class", "") if meta_row else ""
            if meta_row:
                condition = _condition_from_class(cls_str)

            video_labels = _infer_video_findings(title, folder) if meta_row else [0.0] * 7

            instances: List[Instance] = []
            cls_inst = self._class_instance(uid, condition, cls_str)
            if cls_inst:
                instances.append(cls_inst)
            if meta_row:
                instances.extend(self._build_finding_instances(video_labels, img_path.stem))
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

            has_mask = mask_path is not None
            has_findings = any(v > 0 for v in video_labels)
            if has_mask:
                task_type = "segmentation"
            elif has_findings or cls_inst:
                task_type = "multiclass_cls" if cls_inst and not has_findings else "multilabel_cls"
            else:
                task_type = "ssl_only"

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                view_type=probe,
                has_mask=has_mask,
                task_type=task_type,
                ssl_stream="image",
                is_promptable=has_mask or has_findings,
                source_meta={
                    "video_uid": uid,
                    "source": source,
                    "condition": condition,
                    "probe": probe,
                    "frame_idx": frame_idx,
                    "class": cls_str,
                    "title": title,
                    "video_labels": video_labels if meta_row else None,
                },
            )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self._data_root.exists():
            log.warning("COVIDx-US: processed data/ not found under %s", self.root)
            return

        n_vid, n_img = 0, 0
        for entry in self._iter_video_entries():
            n_vid += 1
            yield entry
        for entry in self._iter_image_entries():
            n_img += 1
            yield entry

        log.info("COVIDx-US: emitted %d videos, %d images", n_vid, n_img)
