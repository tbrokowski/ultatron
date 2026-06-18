"""
data/adapters/lung/lung_database.py  ·  Lung Database SSL adapter (~325k frames)

The Lung Database stores frames extracted from lung-ultrasound cine clips.
There are no raw video files on disk; each lung zone scan is represented as
an ordered sequence of JPEG frames.

Filename patterns (all map to patient + lung zone + frame index):
  image_001_Pt01_z01_frame_000000.jpg
  image_001_z01_frame_000000.jpg          (patient inferred from case folder)
  pt212_z01_frame_000000.jpg

Case layout:
  Pt*/images/*.jpg
  pt*/images/*.jpg  or  pt*/*.jpg
  ED*/images/*.jpg
  ED*/processed_*_images_batch/images/*.jpg

Entries emitted per zone clip:
  * one image entry per frame       → image SSL stream
  * one pseudo_video entry / zone   → video SSL + paired (combined) streams
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_IMG_EXTS = {".jpg", ".jpeg", ".png"}
_DEFAULT_FPS = 15.0

_FRAME_PATTERNS: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"^image_\d+_(?P<patient>.+?)_(?P<zone>z\d+)_frame_(?P<frame>\d+)\.(?:jpg|jpeg|png)$",
        re.I,
    ),
    re.compile(
        r"^(?P<patient>pt\d+)_(?P<zone>z\d+)_frame_(?P<frame>\d+)\.(?:jpg|jpeg|png)$",
        re.I,
    ),
    re.compile(
        r"^image_\d+_(?P<zone>z\d+)_frame_(?P<frame>\d+)\.(?:jpg|jpeg|png)$",
        re.I,
    ),
)


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in _IMG_EXTS


def _is_duplicate_name(name: str) -> bool:
    """Skip macOS-style duplicate copies such as 'frame_000001 2.jpg'."""
    return " 2." in name


def _parse_frame(path: Path, case_id: str) -> Optional[Tuple[str, int]]:
    """Return (zone, frame_idx) or None if the filename is not recognised."""
    if _is_duplicate_name(path.name):
        return None
    for pat in _FRAME_PATTERNS:
        m = pat.match(path.name)
        if not m:
            continue
        zone = m.group("zone").lower()
        frame_idx = int(m.group("frame"))
        return zone, frame_idx
    return None


class LungDatabaseAdapter(BaseAdapter):
    DATASET_ID     = "Lung-Database"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "bronze"
    DOI            = ""

    def _iter_case_dirs(self) -> Iterator[Path]:
        seen: set[str] = set()
        for pattern in ("Pt*", "pt*", "ED*"):
            for case_dir in sorted(self.root.glob(pattern)):
                if not case_dir.is_dir():
                    continue
                key = case_dir.name.lower()
                if key in seen:
                    continue
                seen.add(key)
                yield case_dir

    def _load_case_metadata(self, case_dir: Path) -> Dict[str, Tuple[int, int]]:
        """Optional metadata.csv → {filename: (width, height)}."""
        meta_path = case_dir / "metadata.csv"
        if not meta_path.is_file():
            return {}
        dims: Dict[str, Tuple[int, int]] = {}
        with meta_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = (row.get("filename") or "").strip()
                if not name:
                    continue
                try:
                    w = int(row.get("output_width") or row.get("original_width") or 0)
                    h = int(row.get("output_height") or row.get("original_height") or 0)
                except (TypeError, ValueError):
                    continue
                if w > 0 and h > 0:
                    dims[name] = (w, h)
                    stem = Path(name).stem
                    dims[f"{stem}.jpg"] = (w, h)
        return dims

    def _frame_search_paths(self, case_dir: Path) -> List[Tuple[Path, int]]:
        """
        Return (directory, priority) pairs to scan for frames.
        Higher priority wins when the same frame index appears twice.
        """
        locations: List[Tuple[Path, int]] = []

        img_dir = case_dir / "images"
        if img_dir.is_dir():
            locations.append((img_dir, 3))

        for batch in sorted(case_dir.glob("processed_*")):
            batch_img = batch / "images"
            if batch_img.is_dir():
                locations.append((batch_img, 2))

        locations.append((case_dir, 1))
        return locations

    def _collect_zone_clips(
        self,
        case_dir: Path,
    ) -> Dict[str, List[Path]]:
        """Group deduplicated, temporally ordered frame paths by lung zone."""
        zone_frames: Dict[str, Dict[int, Tuple[int, Path]]] = defaultdict(dict)

        for directory, priority in self._frame_search_paths(case_dir):
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if not _is_image(path):
                    continue
                parsed = _parse_frame(path, case_dir.name)
                if parsed is None:
                    continue
                zone, frame_idx = parsed
                existing = zone_frames[zone].get(frame_idx)
                if existing is None or priority > existing[0]:
                    zone_frames[zone][frame_idx] = (priority, path)

        clips: Dict[str, List[Path]] = {}
        for zone, frame_map in zone_frames.items():
            ordered = [frame_map[i][1] for i in sorted(frame_map)]
            if ordered:
                clips[zone] = ordered
        return clips

    def _lookup_dims(
        self,
        path: Path,
        meta: Dict[str, Tuple[int, int]],
    ) -> Tuple[int, int]:
        dims = meta.get(path.name)
        if dims:
            return dims
        return 0, 0

    def iter_entries(self) -> Iterator[USManifestEntry]:
        case_dirs = list(self._iter_case_dirs())
        n_cases = len(case_dirs)

        for i, case_dir in enumerate(case_dirs):
            case_id = case_dir.name
            split = self._infer_split(case_id, i, n_cases)
            meta = self._load_case_metadata(case_dir)
            zone_clips = self._collect_zone_clips(case_dir)

            for zone, frame_paths in sorted(zone_clips.items()):
                series_id = f"{case_id}_{zone}"
                common_meta = {
                    "patient_dir": case_id,
                    "lung_zone": zone,
                    "video_source": "extracted_frames",
                    "n_frames": len(frame_paths),
                }

                # ── Image entries (one per extracted frame) ─────────────────
                for frame_idx, frame_path in enumerate(frame_paths):
                    width, height = self._lookup_dims(frame_path, meta)
                    yield self._make_entry(
                        str(frame_path),
                        split=split,
                        modality="image",
                        study_id=case_id,
                        series_id=series_id,
                        view_type=zone,
                        probe_type="phased_array",
                        height=height,
                        width=width,
                        num_frames=1,
                        task_type="ssl_only",
                        ssl_stream="image",
                        is_promptable=False,
                        source_meta={
                            **common_meta,
                            "frame_idx": frame_idx,
                            "source_frame_number": _parse_frame(frame_path, case_id)[1],
                        },
                    )

                # ── Pseudo-video entry (full zone cine as frame sequence) ─────
                if len(frame_paths) < 2:
                    continue

                width, height = self._lookup_dims(frame_paths[0], meta)
                yield self._make_entry(
                    [str(p) for p in frame_paths],
                    split=split,
                    modality="pseudo_video",
                    study_id=case_id,
                    series_id=series_id,
                    view_type=zone,
                    probe_type="phased_array",
                    height=height,
                    width=width,
                    num_frames=len(frame_paths),
                    fps=_DEFAULT_FPS,
                    is_cine=True,
                    has_temporal_order=True,
                    task_type="ssl_only",
                    ssl_stream="both",
                    is_promptable=False,
                    source_meta=common_meta,
                )
