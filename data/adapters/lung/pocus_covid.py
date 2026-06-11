"""
data/adapters/lung/pocus_covid.py  - POCUS COVID lung image/video adapter
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

_CLASS_PREFIX = {
    "Cov":  "covid",
    "Pneu": "pneumonia",
    "Reg":  "normal",
    "Vir":  "viral",
}
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".PNG"}
_VID_EXTS = {".mp4", ".avi", ".mpeg", ".mov", ".gif"}


class PocusCovidAdapter(BaseAdapter):
    DATASET_ID     = "Pocus-covid"
    ANATOMY_FAMILY = "lung"
    SONODQS        = "silver"
    DOI            = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        yield from self._iter_modality("pocus_images", "image")
        yield from self._iter_modality("pocus_videos", "video")

    def _iter_modality(self, subdir: str, modality: str) -> Iterator[USManifestEntry]:
        base = self.root / subdir
        if not base.is_dir():
            return

        samples: List[tuple[Path, str, str]] = []
        for probe in ("convex", "linear"):
            probe_dir = base / probe
            if not probe_dir.is_dir():
                continue
            exts = _IMG_EXTS if modality == "image" else _VID_EXTS
            for p in sorted(probe_dir.rglob("*")):
                if p.suffix in exts and p.is_file():
                    if "label_uncertain" in p.parts:
                        continue
                    cls = self._infer_class(p.name)
                    samples.append((p, probe, cls))

        n = len(samples)
        for i, (path, probe, cls) in enumerate(samples):
            split = self._infer_split(path.stem, i, n)
            instances: List[Instance] = [
                self._make_instance(
                    instance_id=path.stem,
                    label_raw=cls,
                    label_ontology=cls,
                    is_promptable=False,
                )
            ]

            yield self._make_entry(
                str(path),
                split=split,
                modality=modality,
                instances=instances,
                view_type=probe,
                is_cine=(modality == "video"),
                has_temporal_order=(modality == "video"),
                task_type="multiclass_cls",
                ssl_stream="both" if modality == "video" else "image",
                is_promptable=False,
                source_meta={"probe": probe, "class": cls},
            )

    @staticmethod
    def _infer_class(filename: str) -> str:
        for prefix, label in _CLASS_PREFIX.items():
            if filename.startswith(prefix):
                return label
        return "unknown"
