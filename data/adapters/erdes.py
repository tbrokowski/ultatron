"""
data/adapters/erdes.py  ·  ERDES ocular ultrasound adapter
============================================================

ERDES is a clip-level ocular ultrasound benchmark for retinal detachment and
related subtypes. We use the official `non_rd_vs_rd` split by default.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class ERDESAdapter(BaseAdapter):
    DATASET_ID = "ERDES"
    ANATOMY_FAMILY = "ocular"
    SONODQS = "gold"
    DOI = ""

    def __init__(self, root, split_override=None, split_variant: str = "non_rd_vs_rd"):
        self.split_variant = split_variant
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)
        self._split_map = self._load_split_map()

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "erdes_metadata.csv").exists():
            return root
        candidate = root / cls.DATASET_ID
        if (candidate / "erdes_metadata.csv").exists():
            return candidate
        raise FileNotFoundError(f"{cls.DATASET_ID}: expected metadata under {root}")

    def _load_split_map(self) -> Dict[str, str]:
        split_dir = self.root / "splits" / self.split_variant
        split_map: Dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            csv_path = split_dir / f"{split_name}.csv"
            if not csv_path.exists():
                continue
            with csv_path.open(newline="") as f:
                for row in csv.DictReader(f):
                    rel_path = row.get("path", "").strip()
                    if rel_path:
                        split_map[rel_path] = split_name
        return split_map

    def iter_entries(self) -> Iterator[USManifestEntry]:
        metadata_path = self.root / "erdes_metadata.csv"
        with metadata_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        for idx, row in enumerate(rows):
            rel_path = row["file_path"].strip()
            vpath = self.root / "clips" / rel_path
            if not vpath.exists():
                continue

            clip_id = row["clip_id"].strip()
            study_id = clip_id.rsplit("_", 1)[0]
            split = self.split_override or self._split_map.get(rel_path)
            if split is None:
                split = self._infer_split(study_id, idx, len(rows))

            diagnostic_class = row["diagnostic_class"].strip().lower()
            cls_label = 1 if diagnostic_class == "rd" else 0
            label_ontology = "retinal_detachment" if cls_label else "retina"

            instances = [
                self._make_instance(
                    instance_id=clip_id,
                    label_raw=diagnostic_class,
                    label_ontology=label_ontology,
                    classification_label=cls_label,
                    is_promptable=False,
                )
            ]

            yield self._make_entry(
                str(vpath),
                split=split,
                modality="video",
                instances=instances,
                study_id=study_id,
                view_type="ocular_bscan",
                is_cine=True,
                has_temporal_order=True,
                fps=float(row.get("fps", 0.0) or 0.0),
                num_frames=int(float(row.get("frame_count", 0) or 0)),
                height=int(float(row.get("height", 0) or 0)),
                width=int(float(row.get("width", 0) or 0)),
                clip_duration_s=float(row.get("duration_seconds", 0.0) or 0.0),
                task_type="classification",
                ssl_stream="both",
                is_promptable=False,
                source_meta={
                    "clip_id": clip_id,
                    "diagnostic_class": diagnostic_class,
                    "subtype": row.get("subtype"),
                    "anatomical_subclass": row.get("anatomical_subclass"),
                    "split_variant": self.split_variant,
                    "retinal_detachment_label": cls_label,
                },
            )
