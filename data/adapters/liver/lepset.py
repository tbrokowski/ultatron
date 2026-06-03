"""
data/adapters/liver/lepset.py  ·  LEPset adapter
=================================================

LEPset: Liver/pancreas Endoscopic ultrasound dataset (Zenodo, 2023).
  420 patients (140 NPC + 280 PC), ~13 JPG frames each, plus a flat
  unlabeled pool.  No metadata files — all information is in the folder
  structure.

Layout on disk:
  {root}/
  ├── labeled/
  │   ├── NPC/           # Non-Pancreatic Cancer
  │   │   └── {pid}/     # e.g. 17/, 71/  …
  │   │       └── *.jpg  # 0.jpg, 1.jpg, …, 12.jpg
  │   └── PC/            # Pancreatic Cancer
  │       └── {pid}/
  │           └── *.jpg
  └── unlabeled/
      └── *.jpg          # flat pool, no patient structure

Classification labels:
  NPC → 0 (no pancreatic cancer)
  PC  → 1 (pancreatic cancer)

Split strategy:
  Patient-level deterministic 80/10/10.  All frames from the same patient
  are assigned to the same split so no leakage occurs across frame boundaries.
  Unlabeled frames are always split = "train".

Entries emitted:
  Labeled — one per frame:
    task_type          = "classification"
    has_mask           = False
    study_id / series_id = "{class}_{patient_id}"
    instance           = classification instance
      label_raw        = "PC" | "NPC"
      label_ontology   = "pancreatic_cancer_class"
      classification_label = 1 | 0

  Unlabeled — one per JPG:
    task_type          = "ssl_only"
    has_mask           = False
    instances          = []
    split              = "train"
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


_CLASS_LABELS: Dict[str, int] = {"NPC": 0, "PC": 1}


class LEPsetAdapter(BaseAdapter):
    """
    LEPset adapter.  Yields one image entry per JPG across all labeled patient
    folders and the flat unlabeled pool.
    """

    DATASET_ID     = "LEPset"
    ANATOMY_FAMILY = "liver"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.5281/zenodo.8041285"

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "labeled").is_dir() or (root / "unlabeled").is_dir():
            return root
        candidate = root / "LEPset"
        if (candidate / "labeled").is_dir() or (candidate / "unlabeled").is_dir():
            return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected labeled/ or unlabeled/ under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        yield from self._iter_labeled()
        yield from self._iter_unlabeled()

    # ── Labeled ───────────────────────────────────────────────────────────────

    def _iter_labeled(self) -> Iterator[USManifestEntry]:
        labeled_dir = self.root / "labeled"
        if not labeled_dir.exists():
            return

        for class_name, cls_label in sorted(_CLASS_LABELS.items()):
            class_dir = labeled_dir / class_name
            if not class_dir.exists():
                continue

            patients: List[Path] = sorted(
                p for p in class_dir.iterdir() if p.is_dir()
            )
            n = len(patients)

            for i, patient_dir in enumerate(patients):
                patient_id = patient_dir.name
                study_id   = f"{class_name}_{patient_id}"
                split      = self.split_override or self._infer_split(
                    study_id, i, n
                )

                frames = sorted(
                    patient_dir.glob("*.jpg"),
                    key=lambda p: int(p.stem) if p.stem.isdigit() else float("inf"),
                )

                for img_path in frames:
                    instance = self._make_instance(
                        instance_id          = f"{study_id}_{img_path.stem}",
                        label_raw            = class_name,
                        label_ontology       = "pancreatic_cancer_class",
                        is_promptable        = False,
                        classification_label = cls_label,
                    )

                    yield self._make_entry(
                        str(img_path),
                        split         = split,
                        modality      = "image",
                        instances     = [instance],
                        study_id      = study_id,
                        series_id     = study_id,
                        view_type     = "endoscopic_us",
                        has_mask      = False,
                        task_type     = "classification",
                        ssl_stream    = "image",
                        is_promptable = False,
                        source_meta   = {
                            "class":      class_name,
                            "patient_id": patient_id,
                            "frame":      img_path.stem,
                            "classification_label": cls_label,
                        },
                    )

    # ── Unlabeled ─────────────────────────────────────────────────────────────

    def _iter_unlabeled(self) -> Iterator[USManifestEntry]:
        unlabeled_dir = self.root / "unlabeled"
        if not unlabeled_dir.exists():
            return

        for img_path in sorted(
            unlabeled_dir.glob("*.jpg"),
            key=lambda p: int(p.stem) if p.stem.isdigit() else float("inf"),
        ):
            yield self._make_entry(
                str(img_path),
                split         = self.split_override or "train",
                modality      = "image",
                instances     = [],
                study_id      = img_path.stem,
                series_id     = img_path.stem,
                view_type     = "endoscopic_us",
                has_mask      = False,
                task_type     = "ssl_only",
                ssl_stream    = "image",
                is_promptable = False,
                source_meta   = {"frame": img_path.stem},
            )
