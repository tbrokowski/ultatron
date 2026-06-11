"""
data/adapters/dermatologic_skin_lesions.py  ·  Dermatologic skin lesion US
============================================================================

Dermatologic-US-Skin-Lesions provides paired grayscale B-mode and Doppler
images for each lesion together with benign/malignant labels and diagnosis
strings. When both images are available we expose them as a 2-frame
`pseudo_video`; otherwise we fall back to the available B-mode image.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


_DX_NORMALIZATION = {
    "venous_malfromation": "venous_malformation",
    "leiomyoma": "leiomyoma",
    "leyomioma": "leiomyoma",
    "schwanoma": "schwannoma",
    "fibrofoliculoma": "fibrofolliculoma",
    "sebk": "seborrheic_keratosis",
    "sk": "seborrheic_keratosis",
}


class DermatologicSkinLesionsAdapter(BaseAdapter):
    DATASET_ID = "Dermatologic-US-Skin-Lesions"
    ANATOMY_FAMILY = "skin"
    SONODQS = "silver"
    DOI = ""

    def __init__(self, root, split_override=None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)
        self.image_root = self.root / "images" / "bw"

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "201database.csv").exists():
            return root
        candidate = root / cls.DATASET_ID
        if (candidate / "201database.csv").exists():
            return candidate
        raise FileNotFoundError(f"{cls.DATASET_ID}: expected 201database.csv under {root}")

    def iter_entries(self) -> Iterator[USManifestEntry]:
        rows = self._load_rows()
        split_map = self._group_split_map(row["case_id"] for row in rows)

        for row in rows:
            bw_path = self._resolve_image_path(row["case_id"], "bw", row["bw_rel"])
            if bw_path is None:
                continue
            doppler_path = self._resolve_image_path(row["case_id"], "doppler", row["doppler_rel"])

            image_paths = [str(bw_path)]
            modality = "image"
            ssl_stream = "image"
            view_type = "bmode"
            num_frames = 1
            if doppler_path is not None:
                image_paths.append(str(doppler_path))
                modality = "pseudo_video"
                ssl_stream = "both"
                view_type = "bmode_doppler_pair"
                num_frames = 2

            cls_label = 1 if row["label"] == "malignant" else 0
            label_ontology = (
                "skin_lesion_malignant" if cls_label else "skin_lesion_benign"
            )
            instances = [
                self._make_instance(
                    instance_id=row["case_id"],
                    label_raw=row["label"],
                    label_ontology=label_ontology,
                    classification_label=cls_label,
                    is_promptable=False,
                )
            ]

            yield self._make_entry(
                image_paths,
                split=split_map.get(row["case_id"], "train"),
                modality=modality,
                instances=instances,
                study_id=row["case_id"],
                view_type=view_type,
                num_frames=num_frames,
                has_temporal_order=num_frames > 1,
                task_type="classification",
                ssl_stream=ssl_stream,
                is_promptable=False,
                source_meta={
                    "case_id": row["case_id"],
                    "diagnosis": row["dx"],
                    "frequency_band": row["freq"],
                    "malignancy_label": cls_label,
                    "has_doppler": doppler_path is not None,
                },
            )

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }

    def _load_rows(self) -> List[Dict[str, str]]:
        csv_path = self.root / "201database.csv"
        out: List[Dict[str, str]] = []
        with csv_path.open(newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 4:
                    continue
                bw_rel = row[0].strip()
                doppler_rel = row[1].strip()
                label = row[2].strip().lower()
                if label not in {"benign", "malignant"}:
                    continue
                dx = self._normalize_dx(row[3].strip())
                freq = row[4].strip().lower() if len(row) > 4 else ""
                case_id = self._case_id_from_rel(bw_rel or doppler_rel)
                if case_id is None:
                    continue
                out.append(
                    {
                        "case_id": case_id,
                        "bw_rel": bw_rel,
                        "doppler_rel": doppler_rel,
                        "label": label,
                        "dx": dx,
                        "freq": freq,
                    }
                )
        return out

    @staticmethod
    def _case_id_from_rel(rel_path: str) -> str | None:
        match = re.search(r"(\d+)_", rel_path)
        return match.group(1).zfill(2) if match else None

    @staticmethod
    def _normalize_dx(raw_dx: str) -> str:
        normalized = raw_dx.strip().lower().replace(" ", "_").replace("-", "_")
        normalized = re.sub(r"[^a-z0-9_]+", "", normalized)
        return _DX_NORMALIZATION.get(normalized, normalized)

    def _resolve_image_path(self, case_id: str, kind: str, rel_path: str) -> Path | None:
        rel_path = rel_path.strip()
        if rel_path:
            direct = self.root / rel_path
            if direct.exists():
                return direct

        # Fall back to case-id based globbing to handle small naming mistakes.
        if kind == "bw":
            candidates = sorted(self.image_root.glob(f"{case_id}_bw*.jpg"))
        else:
            candidates = sorted(self.image_root.glob(f"{case_id}_doppler*.jpg"))
            if not candidates:
                candidates = sorted(self.image_root.glob(f"{case_id}*doppler*.jpg"))
        return min(candidates, key=lambda p: len(p.name)) if candidates else None
