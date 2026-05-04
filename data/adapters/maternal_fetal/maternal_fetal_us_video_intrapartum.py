"""
data/adapters/maternal_fetal/maternal_fetal_us_video_intrapartum.py
====================================================================

Maternal-Fetal US Video Intrapartum dataset.

The dataset is organized under DatasetV3/{train,val,test}.  Each split contains
raw AVI videos, frame-level standard-plane classification metadata, and
pre-extracted 512x512 grayscale segmentation masks for pubic symphysis and
fetal head.  Mask naming differs by split:

  train: seg/<video_name>/mask/<video_name>_<frame>_6.png
  val:   seg/<video_name>_<frame>.png
  test:  seg/<video_name>.png

The train suffix "_6" is fixed and is not a frame index.  Landmark coordinates
in landmark.json are [y, x] strings and are converted to [x, y] keypoints.
Empty or missing landmark JSON files are handled as absent annotations.
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


class MaternalFetalUSVideoIntrapartumAdapter(BaseAdapter):
    DATASET_ID     = "maternal-fetal-us-video-intrapartum"
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
        for candidate in (
            root / "DatasetV3",
            root / "maternal-fetal-us-video-intrapartum" / "DatasetV3",
            root / cls.DATASET_ID / "DatasetV3",
        ):
            if cls._looks_like_dataset_root(candidate):
                return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected DatasetV3/{{train,val,test}} under {root}"
        )

    @staticmethod
    def _looks_like_dataset_root(path: Path) -> bool:
        return path.is_dir() and any(
            (path / split_name / "videos").is_dir()
            for split_name, *_rest in SPLITS
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        found_any = False
        for split_name, info_csv_name, cls_csv_name, seg_index_col in SPLITS:
            split_dir = self.root / split_name
            if not split_dir.exists():
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
                f"{self.DATASET_ID}: no train/val/test splits found under {self.root}"
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

        for video_path in sorted(videos_dir.glob("*.avi")):
            stem = video_path.stem
            info = info_rows.get(stem, {})
            seg_info = seg_rows.get(stem, {})
            cls_info = cls_rows.get(stem, {})

            frame_count = self._to_int(
                info.get("frame_count")
                or seg_info.get("frame_count")
                or cls_info.get("frame_count")
            )
            labeled_indices = self._parse_index_list(
                seg_info.get(seg_index_col)
                or seg_info.get("labeled_index")
                or seg_info.get("labeled_frame_index")
                or info.get("labeled_frame_index")
            )
            mask_infos = [
                mask_info for frame_idx in labeled_indices
                if (mask_info := self._mask_info(split_name, seg_dir, stem, frame_idx)) is not None
            ]

            instances = []
            landmark_meta = {}
            for mask_info in mask_infos:
                landmark = landmarks.get(mask_info["name"], {})
                if landmark:
                    landmark_meta[mask_info["name"]] = landmark
                ps_points = self._xy_points(landmark.get("ps_points"))
                head_points = [
                    point for point in (
                        self._xy_point(landmark.get("hsd_point")),
                        self._xy_point(landmark.get("aop_tangency")),
                    )
                    if point is not None
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
            has_mask = bool(mask_infos)

            yield self._make_entry(
                str(video_path),
                split=self.split_override or split_name,
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
                    "info": info,
                    "seg_info": seg_info,
                    "cls_info": cls_info,
                    "labeled_frame_indices": labeled_indices,
                    "mask_paths": [m["path"] for m in mask_infos],
                    "mask_names": [m["name"] for m in mask_infos],
                    "landmarks": landmark_meta,
                    "landmark_coordinates": "yx_strings_converted_to_xy",
                    "measurement_units": {
                        "pubic_symphysis": "aop_degrees",
                        "fetal_head": "hsd_pixels",
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
                    "mask_size": "512x512",
                },
            )

    @staticmethod
    def _load_csv_by_stem(path: Path) -> Dict[str, dict]:
        if not path.exists():
            return {}
        out: Dict[str, dict] = {}
        with path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                filename = (row.get("filename") or "").strip()
                if not filename:
                    continue
                out[Path(filename).stem] = {
                    k: (v or "").strip() for k, v in row.items()
                }
        return out

    @staticmethod
    def _load_landmarks(path: Path) -> Dict[str, dict]:
        if not path.exists():
            return {}
        raw = path.read_text(encoding="utf-8").strip()
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

        out = []
        for part in value.strip("[]").replace(";", ",").replace("|", ",").split(","):
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
            mask_path = seg_dir / video_stem / "mask" / f"{video_stem}_{frame_idx}_6.png"
        elif split_name == "val":
            mask_path = seg_dir / f"{video_stem}_{frame_idx}.png"
        else:
            mask_path = seg_dir / f"{video_stem}.png"

        if not mask_path.exists():
            return None
        return {
            "frame_index": frame_idx,
            "name": mask_path.name,
            "path": str(mask_path),
        }

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
        return [point for raw in points if (point := cls._xy_point(raw)) is not None]

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
