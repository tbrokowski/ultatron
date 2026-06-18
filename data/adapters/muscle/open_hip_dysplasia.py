"""
data/adapters/muscle/open_hip_dysplasia.py  - Open hip dysplasia adapter

Sub-datasets:
  radiopedia_ultrasound_2d/data/  - PNG + _label.png + JSON metadata
  hong_kong_poly_ultrasound_2d/data/  - PNG + quality JSON (SSL only)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance


class OpenHipDysplasiaAdapter(BaseAdapter):
    DATASET_ID     = "open-hip-dysplasia"
    ANATOMY_FAMILY = "joint"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._base = self._find_base(Path(root))

    @staticmethod
    def _find_base(root: Path) -> Path:
        candidates = [
            root / "radoss-org-open-hip-dysplasia-8433611",
            root / "radoss-org" / "radoss-org-open-hip-dysplasia-8433611",
        ]
        for p in candidates:
            if p.is_dir():
                return p
        if (root / "radiopedia_ultrasound_2d").is_dir():
            return root
        if root.is_dir():
            for sub in root.iterdir():
                if (sub / "radiopedia_ultrasound_2d").is_dir():
                    return sub
                nested = sub / "radoss-org-open-hip-dysplasia-8433611"
                if nested.is_dir():
                    return nested
        return root

    def iter_entries(self) -> Iterator[USManifestEntry]:
        yield from self._iter_radiopedia()
        yield from self._iter_hong_kong()

    def _iter_radiopedia(self) -> Iterator[USManifestEntry]:
        data_dir = self._base / "radiopedia_ultrasound_2d" / "data"
        if not data_dir.is_dir():
            return

        images = sorted(p for p in data_dir.glob("*.png") if not p.name.endswith("_label.png"))
        n = len(images)
        for i, img_path in enumerate(images):
            mask_path = data_dir / f"{img_path.stem}_label.png"
            json_path = data_dir / f"{img_path.stem}.json"
            if not mask_path.exists():
                continue

            meta = {}
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text())
                except json.JSONDecodeError:
                    meta = {}

            split = self._infer_split(img_path.stem, i, n)
            graf = meta.get("R/L Graf Type") or meta.get("side", "unknown")

            instances = [
                self._make_instance(
                    instance_id=img_path.stem,
                    label_raw=str(graf),
                    label_ontology="hip",
                    mask_path=str(mask_path),
                    is_promptable=True,
                )
            ]

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                has_mask=True,
                task_type="segmentation",
                ssl_stream="image",
                is_promptable=True,
                source_meta={"subset": "radiopedia", **meta},
            )

    def _iter_hong_kong(self) -> Iterator[USManifestEntry]:
        data_dir = self._base / "hong_kong_poly_ultrasound_2d" / "data"
        if not data_dir.is_dir():
            return

        images = sorted(data_dir.glob("standard_*.png"))
        n = len(images)
        for i, img_path in enumerate(images):
            json_path = data_dir / f"{img_path.stem}.json"
            meta = {}
            if json_path.exists():
                try:
                    meta = json.loads(json_path.read_text())
                except json.JSONDecodeError:
                    meta = {}

            split = self._infer_split(img_path.stem, i, n)
            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                task_type="ssl_only",
                ssl_stream="image",
                is_promptable=False,
                source_meta={"subset": "hong_kong", **meta},
            )
