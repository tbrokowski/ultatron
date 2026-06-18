"""
data/adapters/maternal_fetal/iugc2024.py  ·  IUGC2024 adapter
==============================================================

IUGC2024 is an intrapartum ultrasound video dataset with split directories.

On capstor the release is under ``DatasetV3/`` with Google-Drive zip folder names
(``train-*/train/``, etc.).  The canonical ``new/`` layout from the README is
also supported.  Video files may carry ``__`` suffixes that differ from CSV
metadata stems; metadata lookup uses the prefix before ``__`` when needed.
Val/test mask PNG naming differs between releases (frame suffix vs stem-only);
both conventions are accepted.

Each split contains videos, segmentation metadata/masks, classification frame
indices, and split-level metadata CSVs.  This adapter emits one manifest entry
per video.  For videos with segmentation masks, it adds one pair of instances
per labelled frame, each pointing to that frame's mask:

  mask_channel = 1  pubic symphysis
  mask_channel = 2  fetal head

Frame-level standard-plane classification annotations are preserved in
source_meta as parsed positive/negative frame index lists because the manifest
schema only has one scalar classification label per Instance.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


SPLITS: Tuple[Tuple[str, str, str, str], ...] = (
    ("train", "train_info.csv", "class_label.csv", "labeled_index"),
    ("val",   "val_info.csv",   "cls_label.csv",   "labeled_frame_index"),
    ("test",  "test_info.csv",  "cls_label.csv",   "labeled_frame_index"),
)

STRUCTURES = (
    ("pubic_symphysis", "pubic_symphysis", 1),
    ("fetal_head",      "fetal_head",      2),
)


class IUGC2024Adapter(BaseAdapter):
    DATASET_ID     = "IUGC2024"
    ANATOMY_FAMILY = "intrapartum"
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
        if cls._looks_like_dataset_root(root):
            return root
        for name in ("new", "DatasetV3", "IUGC2024", "IUGC-2024"):
            candidate = root / name
            if cls._looks_like_dataset_root(candidate):
                return candidate
            for nested in (candidate / "new", candidate):
                if cls._looks_like_dataset_root(nested):
                    return nested
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected train/val/test split dirs under {root}"
        )

    @staticmethod
    def _looks_like_dataset_root(path: Path) -> bool:
        if not path.is_dir():
            return False
        return any(IUGC2024Adapter._find_split_dir(path, split) is not None
                   for split, *_rest in SPLITS)

    @staticmethod
    def _find_split_dir(root: Path, split_name: str) -> Optional[Path]:
        direct = root / split_name
        if (direct / "videos").is_dir():
            return direct
        for candidate in sorted(root.glob(f"{split_name}*")):
            nested = candidate / split_name
            if (nested / "videos").is_dir():
                return nested
        return None

    def iter_entries(self) -> Iterator[USManifestEntry]:
        found_any = False
        for split_name, info_csv_name, cls_csv_name, seg_index_col in SPLITS:
            split_dir = self._find_split_dir(self.root, split_name)
            if split_dir is None:
                continue
            found_any = True
            yield from self._iter_split(
                split_name=split_name,
                split_dir=split_dir,
                info_csv_name=info_csv_name,
                cls_csv_name=cls_csv_name,
                seg_index_col=seg_index_col,
            )

        if not found_any:
            raise FileNotFoundError(
                f"IUGC2024: no train/val/test split directories found under {self.root}"
            )

    def _iter_split(
        self,
        split_name: str,
        split_dir: Path,
        info_csv_name: str,
        cls_csv_name: str,
        seg_index_col: str,
    ) -> Iterator[USManifestEntry]:
        videos_dir = split_dir / "videos"
        seg_dir = split_dir / "seg"
        cls_dir = split_dir / "cls"
        if not videos_dir.exists():
            return

        info_rows = self._load_csv_by_stem(split_dir / info_csv_name)
        seg_rows = self._load_csv_by_stem(seg_dir / "seg_info.csv")
        cls_rows = self._load_csv_by_stem(cls_dir / cls_csv_name)
        landmarks = self._load_landmarks(seg_dir / "landmark.json")
        known_stems = set(info_rows) | set(seg_rows) | set(cls_rows)

        for video_path in sorted(videos_dir.glob("*.avi")):
            stem = video_path.stem
            meta_stem = self._canonical_stem(stem, known_stems)
            info = info_rows.get(meta_stem, {})
            seg_info = seg_rows.get(meta_stem, {})
            cls_info = cls_rows.get(meta_stem, {})

            frame_count = self._to_int(
                info.get("frame_count") or seg_info.get("frame_count") or cls_info.get("frame_count")
            )
            labeled_indices = self._parse_index_list(
                seg_info.get(seg_index_col)
                or seg_info.get("labeled_index")
                or seg_info.get("labeled_frame_index")
                or info.get("labeled_frame_index")
            )
            mask_infos = [
                mask_info for idx in labeled_indices
                if (mask_info := self._mask_info(split_name, seg_dir, meta_stem, idx)) is not None
            ]

            instances = []
            for mask_info in mask_infos:
                landmark = landmarks.get(mask_info["name"], {})
                ps_points = self._xy_points(landmark.get("ps_points"))
                head_points = [
                    p for p in (
                        self._xy_point(landmark.get("hsd_point")),
                        self._xy_point(landmark.get("aop_tangency")),
                    )
                    if p is not None
                ]
                frame_idx = mask_info["frame_index"]
                for struct_name, label_ontology, channel in STRUCTURES:
                    instances.append(
                        self._make_instance(
                            instance_id=f"{stem}_{frame_idx}_{struct_name}",
                            label_raw=struct_name,
                            label_ontology=label_ontology,
                            mask_path=mask_info["path"],
                            mask_channel=channel,
                            keypoints=ps_points if struct_name == "pubic_symphysis" else head_points,
                            measurement_mm=self._measurement_for_structure(landmark, struct_name),
                            is_promptable=True,
                        )
                    )

            pos_index = cls_info.get("pos_index", "")
            neg_index = cls_info.get("neg_index", "")
            split = self.split_override or split_name
            has_mask = bool(mask_infos)

            yield self._make_entry(
                str(video_path),
                split=split,
                modality="video",
                instances=instances,
                study_id=stem,
                series_id=stem,
                view_type="intrapartum_transperineal",
                num_frames=frame_count or 0,
                frame_indices=labeled_indices or None,
                is_cine=True,
                has_temporal_order=True,
                has_mask=has_mask,
                task_type="segmentation" if has_mask else "classification",
                ssl_stream="both",
                is_promptable=has_mask,
                source_meta={
                    "split_dir": split_name,
                    "video_filename": video_path.name,
                    "video_stem": stem,
                    "metadata_stem": meta_stem,
                    "info": info,
                    "seg_info": seg_info,
                    "cls_info": cls_info,
                    "labeled_frame_indices": labeled_indices,
                    "mask_paths": [m["path"] for m in mask_infos],
                    "mask_names": [m["name"] for m in mask_infos],
                    "landmarks": {
                        m["name"]: landmarks[m["name"]]
                        for m in mask_infos
                        if m["name"] in landmarks
                    },
                    "pos_index_raw": pos_index or None,
                    "neg_index_raw": neg_index or None,
                    "pos_indices": self._parse_special_index(pos_index, frame_count),
                    "neg_indices": self._parse_special_index(neg_index, frame_count),
                    "standard_plane": self._to_bool(info.get("pos")),
                    "sp_count": self._to_int(info.get("SP_count")),
                    "nsp_count": self._to_int(info.get("NSP_count")),
                    "sp_indices": self._parse_index_list(info.get("SP_index")),
                    "nsp_indices": self._parse_index_list(info.get("NSP_index")),
                },
            )

    @staticmethod
    def _canonical_stem(stem: str, known_stems: set[str]) -> str:
        if stem in known_stems:
            return stem
        if "__" in stem:
            base = stem.split("__", 1)[0]
            if base in known_stems:
                return base
        return stem

    _TEXT_ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1", "cp1252")

    @classmethod
    def _read_text(cls, path: Path) -> str:
        raw_bytes = path.read_bytes()
        for encoding in cls._TEXT_ENCODINGS:
            try:
                return raw_bytes.decode(encoding)
            except UnicodeDecodeError:
                continue
        return raw_bytes.decode("latin-1", errors="replace")

    @classmethod
    def _load_csv_by_stem(cls, path: Path) -> Dict[str, dict]:
        if not path.exists():
            return {}
        out: Dict[str, dict] = {}
        text = cls._read_text(path)
        for row in csv.DictReader(text.splitlines()):
            filename = (row.get("filename") or "").strip()
            if not filename:
                continue
            out[Path(filename).stem] = {k: (v or "").strip() for k, v in row.items()}
        return out

    @classmethod
    def _load_landmarks(cls, path: Path) -> Dict[str, dict]:
        if not path.exists():
            return {}
        try:
            raw = cls._read_text(path).strip()
        except (UnicodeDecodeError, OSError):
            return {}
        if not raw:
            return {}
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _parse_index_list(raw: Optional[str]) -> List[int]:
        if raw is None:
            return []
        value = str(raw).strip()
        if not value or value.lower() in {"none", "null", "nan"}:
            return []
        if value.upper() == "ALL":
            return []
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [int(float(v)) for v in parsed]
        except (ValueError, SyntaxError):
            pass
        value = value.strip("[]")
        out = []
        for part in value.replace(";", ",").replace("|", ",").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                out.append(int(float(part)))
            except ValueError:
                continue
        return out

    @classmethod
    def _parse_special_index(cls, raw: Optional[str], frame_count: Optional[int]):
        if raw is None:
            return []
        value = str(raw).strip()
        if value.upper() == "ALL" and frame_count is not None:
            return list(range(frame_count))
        if value.upper() in {"ALL", "NONE"}:
            return value.upper()
        return cls._parse_index_list(value)

    @staticmethod
    def _mask_info(split_name: str, seg_dir: Path, video_stem: str, frame_idx: int) -> Optional[dict]:
        if split_name == "train":
            candidates = (
                seg_dir / video_stem / "mask" / f"{video_stem}_{frame_idx}_6.png",
            )
        elif split_name == "val":
            candidates = (
                seg_dir / f"{video_stem}_{frame_idx}.png",
                seg_dir / f"{video_stem}.png",
            )
        else:
            candidates = (
                seg_dir / f"{video_stem}.png",
                seg_dir / f"{video_stem}_{frame_idx}.png",
            )

        for mask_path in candidates:
            if mask_path.exists():
                return {
                    "frame_index": frame_idx,
                    "name": mask_path.name,
                    "path": str(mask_path),
                }
        return None

    @staticmethod
    def _xy_point(point) -> Optional[List[float]]:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            y = float(point[0])
            x = float(point[1])
        except (TypeError, ValueError):
            return None
        return [x, y]

    @classmethod
    def _xy_points(cls, points) -> List[List[float]]:
        if not isinstance(points, list):
            return []
        return [p for point in points if (p := cls._xy_point(point)) is not None]

    @staticmethod
    def _measurement_for_structure(landmark: dict, structure: str) -> Optional[float]:
        key = "aop" if structure == "pubic_symphysis" else "hsd"
        value = landmark.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_int(value) -> Optional[int]:
        if value is None or str(value).strip().lower() in {"", "null", "nan"}:
            return None
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _to_bool(value) -> Optional[bool]:
        if value is None:
            return None
        normalized = str(value).strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        return None
