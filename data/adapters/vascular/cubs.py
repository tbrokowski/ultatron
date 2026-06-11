"""
data/adapters/cubs.py  ·  CUBS carotid ultrasound adapter
==========================================================

CUBS contains TIFF carotid ultrasound images with calibration factors and
manual contour annotations that can be converted into IMT measurements.
"""
from __future__ import annotations

import csv
from bisect import bisect_right
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


Point = Tuple[float, float]


class CUBSAdapter(BaseAdapter):
    DATASET_ID = "CUBS"
    ANATOMY_FAMILY = "vascular"
    SONODQS = "silver"
    DOI = "https://doi.org/10.1016/j.ultrasmedbio.2021.03.022"

    def __init__(self, root, split_override=None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)
        self._clinical_rows = self._load_clinical_rows()

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if root.name == "CUBS" and root.exists():
            return root
        candidate = root / "CUBS"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{cls.DATASET_ID}: expected 'CUBS' under {root}")

    def iter_entries(self) -> Iterator[USManifestEntry]:
        image_dir = self.root / "IMAGES"
        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".tiff", ".tif"}
        )
        patient_splits = self._group_split_map(self._patient_id(p.stem) for p in images)

        for img_path in images:
            stem = img_path.stem
            patient_id, side = self._parse_stem(stem)

            cf_mm_per_px = self._read_scalar(self.root / "CF" / f"{stem}_CF.txt")

            manual_a1 = self.root / "SEGMENTATIONS" / "Manual-A1"
            li_path = manual_a1 / f"{stem}-LI.txt"
            ma_path = manual_a1 / f"{stem}-MA.txt"
            rect_path = manual_a1 / f"{stem}_rect.txt"

            measurement_mm = self._read_scalar(
                self.root / "SEGMENTATIONS" / "Manual-A3" / f"{stem}-IMT.txt"
            )
            measurement_source = None
            if measurement_mm is not None:
                measurement_source = "manual_a3_imt"
            else:
                li_points = self._read_points(li_path)
                ma_points = self._read_points(ma_path)
                measurement_mm = self._estimate_imt_mm(li_points, ma_points, cf_mm_per_px)
                if measurement_mm is not None:
                    measurement_source = "manual_a1_contours_interp"

            bbox_xyxy = self._read_rect_bbox(rect_path)

            instances = []
            if measurement_mm is not None:
                instances.append(
                    self._make_instance(
                        instance_id=stem,
                        label_raw="carotid_imt_mm",
                        label_ontology="intima_media",
                        measurement_mm=float(measurement_mm),
                        bbox_xyxy=bbox_xyxy,
                        is_promptable=False,
                    )
                )

            source_meta = {
                "cohort": self.DATASET_ID,
                "patient_id": patient_id,
                "side": side,
                "cf_mm_per_px": cf_mm_per_px,
                "measurement_mm": measurement_mm,
                "measurement_source": measurement_source,
                "manual_a1_li_path": str(li_path) if li_path.exists() else None,
                "manual_a1_ma_path": str(ma_path) if ma_path.exists() else None,
                "manual_a1_rect_path": str(rect_path) if rect_path.exists() else None,
            }
            clinical_meta = self._clinical_rows.get(patient_id)
            if clinical_meta:
                source_meta["clinical"] = clinical_meta

            yield self._make_entry(
                str(img_path),
                split=patient_splits.get(patient_id, "train"),
                modality="image",
                instances=instances,
                study_id=patient_id,
                view_type="common_carotid_long_axis",
                has_box=bbox_xyxy is not None,
                task_type="measurement" if measurement_mm is not None else "ssl_only",
                ssl_stream="image",
                is_promptable=False,
                source_meta=source_meta,
            )

    def _load_clinical_rows(self) -> Dict[str, dict]:
        path = self.root / "ClinicalDatabase-CUBS.csv"
        if not path.exists():
            return {}

        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            header = self._normalize_header(next(reader))
            out: Dict[str, dict] = {}
            for row in reader:
                if not row:
                    continue
                row = row + [""] * (len(header) - len(row))
                raw = {header[i]: row[i].strip() for i in range(len(header))}
                patient_id = raw.get("patient_id")
                if not patient_id:
                    continue
                typed = {}
                for key, value in raw.items():
                    parsed = self._coerce_scalar(value)
                    if parsed is not None:
                        typed[key] = parsed
                out[patient_id] = typed
        return out

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }

    @staticmethod
    def _normalize_header(header: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        normalized: List[str] = []
        for idx, raw_name in enumerate(header):
            key = raw_name.strip().lower().replace(" ", "_").replace("-", "_")
            if not key:
                key = "collection_site" if idx == 0 else f"column_{idx}"
            count = seen.get(key, 0) + 1
            seen[key] = count
            normalized.append(key if count == 1 else f"{key}_{count}")
        return normalized

    @staticmethod
    def _coerce_scalar(value: str):
        value = value.strip()
        if not value:
            return None
        numeric = value.replace(",", ".")
        try:
            as_float = float(numeric)
        except ValueError:
            return value
        return int(as_float) if as_float.is_integer() else as_float

    @staticmethod
    def _patient_id(stem: str) -> str:
        return stem.rsplit("_", 1)[0]

    @staticmethod
    def _parse_stem(stem: str) -> Tuple[str, str]:
        patient_id, side = stem.rsplit("_", 1)
        return patient_id, side

    @staticmethod
    def _read_scalar(path: Path) -> Optional[float]:
        if not path.exists():
            return None
        raw = path.read_text().strip()
        if not raw:
            return None
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            return None

    @staticmethod
    def _read_points(path: Path) -> List[Point]:
        if not path.exists():
            return []
        points: List[Point] = []
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0].replace(",", "."))
                y = float(parts[1].replace(",", "."))
            except ValueError:
                continue
            points.append((x, y))
        return points

    @staticmethod
    def _read_rect_bbox(path: Path) -> Optional[List[float]]:
        if not path.exists():
            return None
        parts = path.read_text().split()
        if len(parts) < 4:
            return None
        try:
            x, y, w, h = [float(p.replace(",", ".")) for p in parts[:4]]
        except ValueError:
            return None
        return [x, y, x + w, y + h]

    @classmethod
    def _estimate_imt_mm(
        cls,
        li_points: List[Point],
        ma_points: List[Point],
        cf_mm_per_px: Optional[float],
    ) -> Optional[float]:
        if cf_mm_per_px is None or cf_mm_per_px <= 0.0:
            return None
        mean_px = cls._mean_boundary_distance_px(li_points, ma_points)
        if mean_px is None:
            return None
        return mean_px * cf_mm_per_px

    @classmethod
    def _mean_boundary_distance_px(
        cls,
        li_points: List[Point],
        ma_points: List[Point],
    ) -> Optional[float]:
        li = sorted(li_points)
        ma = sorted(ma_points)
        if len(li) < 2 or len(ma) < 2:
            return None

        xs_ma = [p[0] for p in ma]
        ys_ma = [p[1] for p in ma]
        x_min = xs_ma[0]
        x_max = xs_ma[-1]

        distances: List[float] = []
        for x_li, y_li in li:
            if x_li < x_min or x_li > x_max:
                continue
            y_ma = cls._interp_y(xs_ma, ys_ma, x_li)
            if y_ma is None:
                continue
            distances.append(abs(y_ma - y_li))

        if not distances:
            return None
        return float(fmean(distances))

    @staticmethod
    def _interp_y(xs: List[float], ys: List[float], x: float) -> Optional[float]:
        if len(xs) < 2:
            return None
        idx = bisect_right(xs, x)
        if idx == 0:
            return ys[0]
        if idx >= len(xs):
            return ys[-1]

        x0, y0 = xs[idx - 1], ys[idx - 1]
        x1, y1 = xs[idx], ys[idx]
        if x1 == x0:
            return y0
        alpha = (x - x0) / (x1 - x0)
        return y0 + alpha * (y1 - y0)
