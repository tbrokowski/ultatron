"""
manifest.py  ·  Master manifest schema, builder, and helpers
=============================================================

Every dataset (image / video / volume / pseudo-video) is reduced to a
USManifestEntry.  The manifest is written as newline-delimited JSON (JSONL)
and consumed by both image-SSL and video-SSL data streams.

Design principles
-----------------
* One entry == one loadable sample  (frame | clip | volume)
* All paths are absolute; dataset root is stored in source_meta so the
  manifest is portable across machines via remap_roots().
* Label heterogeneity → list[Instance]; task_type tells downstream code
  what the labels mean.
* ssl_stream  ("image" | "video" | "both")  routes samples to the two
  DataLoader streams without re-reading files.
* anatomy_family is normalised to ~20 canonical strings for stratified sampling.
* curriculum_tier (1|2|3) drives the OPENUS-style difficulty schedule.
"""
from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import logging

log = logging.getLogger(__name__)

# ── Anatomy taxonomy ──────────────────────────────────────────────────────────
ANATOMY_FAMILIES = {
    "cardiac", "cardiac_fetal",
    "lung",
    "breast",
    "thyroid",
    "fetal_head", "fetal_abdomen", "fetal_brain", "fetal_cardiac",
    "fetal_femur", "fetal_thorax", "fetal_planes", "intrapartum", "cervix",
    "prostate", "ovarian",
    "liver", "pancreas", "gallbladder", "abdomen", "kidney", "colon",
    "muscle", "spine", "nerve", "joint",
    "carotid", "vascular",
    "brain",
    "skin", "ocular", "multi", "other",
}

ANATOMY_ALIASES: Dict[str, str] = {
    "heart": "cardiac", "echo": "cardiac", "echocardiogram": "cardiac",
    "lv": "cardiac", "aortic_stenosis": "cardiac",
    "lus": "lung", "covid": "lung", "b-line": "lung", "b_line": "lung",
    "fetus": "fetal_abdomen", "fetal": "fetal_abdomen",
    "fetal_abdominal": "fetal_abdomen", "maternal_cervix": "cervix",
    "fh": "fetal_head", "head_circumference": "fetal_head",
    "muscles": "muscle", "lower_leg": "muscle",
    "lumbar_multifidus": "spine", "msk": "muscle",
    "musculoskeletal": "muscle",
    "knee": "joint", "mtj": "muscle",
    "brachial_plexus": "nerve", "plexus": "nerve",
    "carotid_artery": "carotid", "microbubbles": "vascular",
    "gallbladder_cancer": "gallbladder", "appendicitis": "abdomen",
    "pcos": "ovarian", "skin_lesion": "skin",
    "retinal_detachment": "ocular", "gist": "abdomen",
}

def normalize_anatomy(raw: Optional[str]) -> str:
    if raw is None:
        return "other"
    key = raw.lower().replace(" ", "_").replace("-", "_")
    return ANATOMY_FAMILIES.__contains__(key) and key \
        or ANATOMY_ALIASES.get(key, "other")

SONODQS_SCORE = {
    "diamond": 7, "platinum": 6, "gold": 5, "silver": 4,
    "bronze": 3,  "steel": 2,   "unrated": 1,
}

# ── Instance  (normalised label) ──────────────────────────────────────────────
@dataclass
class Instance:
    instance_id: str
    label_raw: str
    label_ontology: str
    anatomy_family: str
    mask_path: Optional[str] = None
    mask_channel: Optional[int] = None
    bbox_xyxy: Optional[List[float]] = None
    polygon: Optional[List[List[float]]] = None
    keypoints: Optional[List[List[float]]] = None
    classification_label: Optional[int] = None
    measurement_mm: Optional[float] = None
    is_promptable: bool = False

# ── Master manifest entry ─────────────────────────────────────────────────────
@dataclass
class USManifestEntry:
    sample_id: str
    dataset_id: str
    study_id: Optional[str] = None
    series_id: Optional[str] = None
    instance_id: Optional[str] = None
    modality_type: Literal["image","video","volume","pseudo_video"] = "image"
    split: Literal["train","val","test","unlabeled"] = "train"
    image_paths: List[str] = field(default_factory=list)
    height: int = 0
    width: int = 0
    num_frames: int = 1
    is_3d: bool = False
    is_cine: bool = False
    fps: Optional[float] = None
    frame_indices: Optional[List[int]] = None
    clip_duration_s: Optional[float] = None
    probe_type: Optional[str] = None
    view_type: Optional[str] = None
    acquisition_country: Optional[str] = None
    anatomy_family: str = "other"
    instances: List[Instance] = field(default_factory=list)
    label_raw: Optional[List[str]] = None
    label_ids_raw: Optional[List[int]] = None
    task_type: Literal[
        "segmentation","classification","detection","sequence",
        "measurement","regression","keypoint","ssl_only","weak_label",
        "reconstruction_3d",
    ] = "ssl_only"
    has_mask: bool = False
    has_box: bool = False
    has_points: bool = False
    has_temporal_order: bool = False
    is_promptable: bool = False
    ssl_stream: Literal["image","video","both"] = "image"
    curriculum_tier: int = 1
    sonodqs: str = "unrated"
    quality_score: int = 1
    source_meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "USManifestEntry":
        payload = dict(d)
        raw_instances = payload.pop("instances", []) or []
        inst_fields = set(Instance.__dataclass_fields__)
        instances = []
        for inst in raw_instances:
            if not isinstance(inst, dict):
                continue
            instances.append(Instance(**{k: v for k, v in inst.items() if k in inst_fields}))
        known = set(cls.__dataclass_fields__)
        filtered = {k: v for k, v in payload.items() if k in known}
        return cls(**filtered, instances=instances)

    @staticmethod
    def make_sample_id(dataset_id: str, path: str) -> str:
        return hashlib.md5(f"{dataset_id}::{path}".encode()).hexdigest()[:16]


def coerce_entry_dict(rec: dict, ssl_stream: Optional[str] = None) -> dict:
    """Map a POCUS / HF / adapter row onto a ``USManifestEntry.to_dict()`` payload.

    Extra keys (caption, clip_plan, body_system, …) land in ``source_meta`` so
    ``from_dict`` stays strict about the dataclass fields.
    """
    rec = dict(rec)
    meta = dict(rec.get("source_meta") or {})
    known = set(USManifestEntry.__dataclass_fields__)

    def _first(*names: str):
        for n in names:
            v = rec.get(n)
            if v not in (None, "", [], {}):
                return v
            v = meta.get(n)
            if v not in (None, "", [], {}):
                return v
        return None

    paths = rec.get("image_paths")
    if not paths:
        raw = _first("image", "video_path", "path")
        if isinstance(raw, str):
            paths = [raw]
        elif isinstance(raw, list):
            paths = raw
        else:
            paths = []
    rec["image_paths"] = [str(p) for p in paths]

    if not rec.get("num_frames"):
        nf = _first("n_frames", "num_frames")
        if nf:
            rec["num_frames"] = int(nf)
    if not rec.get("dataset_id"):
        rec["dataset_id"] = str(_first("dataset_id", "dataset") or "unknown")
    if not rec.get("sample_id"):
        stem = Path(rec["image_paths"][0]).stem if rec["image_paths"] else "unknown"
        rec["sample_id"] = str(_first("sample_id", "study_id") or stem)
    caption = _first("caption", "report_text")
    if caption:
        meta.setdefault("caption", caption)
        meta.setdefault("report_text", caption)
    for k in ("body_system", "organ", "attributes", "diagnosis", "findings",
              "view", "probe", "clip_plan", "labels", "H", "W", "bytes"):
        v = rec.get(k)
        if v not in (None, "", [], {}) and k not in meta:
            meta[k] = v
    extras = {k: v for k, v in rec.items() if k not in known and k not in ("instances",)}
    for k, v in extras.items():
        meta.setdefault(k, v)
    rec["source_meta"] = meta
    if ssl_stream:
        rec["ssl_stream"] = ssl_stream
    rec.setdefault("ssl_stream", "image")
    rec.setdefault("split", rec.get("split") or "train")
    return USManifestEntry.from_dict(rec).to_dict()


def concat_manifests(
    dest: Path,
    parts: List[tuple],
) -> dict:
    """Write ``dest`` as JSONL. ``parts`` is ``[(path, ssl_stream_or_None), ...]``."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    per_stream: Dict[str, int] = {}
    with dest.open("w", encoding="utf-8") as out:
        for path, stream in parts:
            p = Path(path)
            if not p.exists():
                continue
            with p.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = coerce_entry_dict(json.loads(line), ssl_stream=stream)
                    if stream == "video" and rec.get("ssl_stream") == "image":
                        rec["ssl_stream"] = "video"
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
                    per_stream[rec["ssl_stream"]] = per_stream.get(rec["ssl_stream"], 0) + 1
    return {"path": str(dest), "n": n, "by_ssl_stream": per_stream}


def assign_curriculum_tier(e: USManifestEntry) -> int:
    if e.num_frames > 64 or (not e.has_mask and e.anatomy_family == "other"):
        return 3
    if e.has_mask and e.anatomy_family != "other" and e.num_frames <= 16:
        return 1
    return 2


# ── I/O ───────────────────────────────────────────────────────────────────────
class ManifestWriter:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w")
        self._count = 0
    def write(self, e: USManifestEntry):
        self._fh.write(json.dumps(e.to_dict()) + "\n")
        self._count += 1
    def close(self) -> int:
        self._fh.close()
        log.info(f"Wrote {self._count} entries → {self.path}")
        return self._count
    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def load_manifest(
    path: Path,
    split: Optional[str] = None,
    ssl_stream: Optional[str] = None,
    anatomy_families: Optional[List[str]] = None,
    min_tier: int = 1,
    max_tier: int = 3,
) -> List[USManifestEntry]:
    entries = []
    with open(path) as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            e = USManifestEntry.from_dict(d)
            if split and e.split != split: continue
            if ssl_stream and e.ssl_stream not in (ssl_stream, "both"): continue
            if anatomy_families and e.anatomy_family not in anatomy_families: continue
            if not (min_tier <= e.curriculum_tier <= max_tier): continue
            entries.append(e)
    return entries


def remap_roots(entries, old_root: str, new_root: str):
    for e in entries:
        e.image_paths = [p.replace(old_root, new_root) for p in e.image_paths]
        for inst in e.instances:
            if inst.mask_path:
                inst.mask_path = inst.mask_path.replace(old_root, new_root)
    return entries


def manifest_stats(entries) -> dict:
    from collections import Counter
    return {
        "total": len(entries),
        "by_anatomy": dict(Counter(e.anatomy_family for e in entries)),
        "by_dataset": dict(Counter(e.dataset_id for e in entries)),
        "by_modality": dict(Counter(e.modality_type for e in entries)),
        "by_ssl_stream": dict(Counter(e.ssl_stream for e in entries)),
        "by_tier": dict(Counter(e.curriculum_tier for e in entries)),
        "has_mask": sum(e.has_mask for e in entries),
        "video_eligible": sum(e.ssl_stream in ("video","both") for e in entries),
    }
