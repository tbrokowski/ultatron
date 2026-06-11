"""
data/adapters/maternal_fetal/acouslic.py  ·  ACOUSLIC-AI adapter
=================================================================

ACOUSLIC-AI: fetal abdominal-circumference blind-sweep ultrasound dataset
(Grand Challenge / MICCAI 2024).  Each case is a MetaImage stack of 840
cine frames (744×562) acquired via the 6-sweep Obstetric Sweep Protocol
(OSP); we emit one manifest entry per OSP sweep (140 frames each).

Layout on disk:
  {root}/
  └── acouslic-ai-train-set/
      ├── images/stacked_fetal_ultrasound/<uuid>.mha   (300 cases)
      ├── masks/stacked_fetal_abdomen/<uuid>.mha       (300 masks)
      └── circumferences/
          └── fetal_abdominal_circumferences_per_sweep.csv

MetaImage format (both image and mask):
  uint8, DimSize = 744 562 840 → numpy (840, 562, 744) after loading.
  Temporal axis stacks all six OSP sweeps (140 frames each); spacing 0.28 mm.

CSV columns:
  uuid, subject_id, sweep_1_ac_mm … sweep_6_ac_mm
  Each row is one case; up to six per-sweep AC reference measurements.

Manifest: 300 cases × 6 OSP sweeps = 1 800 video entries.

Split strategy: group by subject_id to avoid leakage across sweeps of the
same patient.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_SWEEP_COLS        = [f"sweep_{i}_ac_mm" for i in range(1, 7)]
_N_SWEEPS          = 6
_N_FRAMES          = 840
_FRAMES_PER_SWEEP  = _N_FRAMES // _N_SWEEPS


class ACOUSLICAIAdapter(BaseAdapter):
    DATASET_ID     = "ACOUSLIC-AI"
    ANATOMY_FAMILY = "fetal_abdomen"
    SONODQS        = "gold"
    DOI            = "https://acouslic-ai.grand-challenge.org/"

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )
        self._meta = self._load_csv()

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        candidate = root / "acouslic-ai-train-set"
        if candidate.is_dir():
            return candidate
        if (root / "images" / "stacked_fetal_ultrasound").is_dir():
            return root
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'acouslic-ai-train-set/' under {root}"
        )

    def _load_csv(self) -> Dict[str, dict]:
        """Return {uuid: {subject_id, sweep_ac_mm: {1..6: float|None}}} from CSV."""
        csv_path = (
            self.root
            / "circumferences"
            / "fetal_abdominal_circumferences_per_sweep.csv"
        )
        if not csv_path.exists():
            return {}

        out: Dict[str, dict] = {}
        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                uuid = row.get("uuid", "").strip()
                if not uuid:
                    continue
                subject_id = str(row.get("subject_id", "")).strip().lstrip("0") or "0"
                sweep_ac: Dict[int, Optional[float]] = {}
                for sweep_idx, col in enumerate(_SWEEP_COLS, start=1):
                    raw = row.get(col, "").strip()
                    if not raw:
                        sweep_ac[sweep_idx] = None
                        continue
                    try:
                        sweep_ac[sweep_idx] = float(raw)
                    except ValueError:
                        sweep_ac[sweep_idx] = None
                out[uuid] = {
                    "subject_id": subject_id,
                    "sweep_ac_mm": sweep_ac,
                }
        return out

    @staticmethod
    def _sweep_frame_indices(sweep_idx: int) -> List[int]:
        start = (sweep_idx - 1) * _FRAMES_PER_SWEEP
        end   = sweep_idx * _FRAMES_PER_SWEEP
        return list(range(start, end))

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir  = self.root / "images" / "stacked_fetal_ultrasound"
        mask_dir = self.root / "masks"  / "stacked_fetal_abdomen"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"ACOUSLIC-AI: image directory not found at {img_dir}"
            )

        images = sorted(
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".mha"
        )

        subject_uuids: Dict[str, List[str]] = {}
        for img_path in images:
            uuid = img_path.stem
            sid  = self._meta.get(uuid, {}).get("subject_id") or uuid
            subject_uuids.setdefault(sid, []).append(uuid)

        subjects = sorted(subject_uuids)
        subject_split: Dict[str, str] = {
            sid: self._infer_split(sid, i, len(subjects))
            for i, sid in enumerate(subjects)
        }

        for img_path in images:
            uuid       = img_path.stem
            row        = self._meta.get(uuid, {})
            subject_id = row.get("subject_id") or uuid
            sweep_ac   = row.get("sweep_ac_mm") or {}

            mask_path = mask_dir / img_path.name
            has_mask  = mask_path.exists()
            split     = self.split_override or subject_split.get(subject_id, "train")

            for sweep_idx in range(1, _N_SWEEPS + 1):
                series_id = f"{uuid}_sweep{sweep_idx}"
                ac_mm     = sweep_ac.get(sweep_idx)
                frame_idx = self._sweep_frame_indices(sweep_idx)

                if has_mask:
                    instance = self._make_instance(
                        instance_id    = series_id,
                        label_raw      = "fetal_abdomen",
                        label_ontology = "fetal_abdomen",
                        mask_path      = str(mask_path),
                        is_promptable  = True,
                        measurement_mm = ac_mm,
                    )
                    task_type = "segmentation"
                else:
                    instance = self._make_instance(
                        instance_id    = series_id,
                        label_raw      = "fetal_abdomen",
                        label_ontology = "fetal_abdomen",
                        is_promptable  = False,
                        measurement_mm = ac_mm,
                    )
                    task_type = "ssl_only"

                yield self._make_entry(
                    str(img_path),
                    split              = split,
                    modality           = "video",
                    instances          = [instance],
                    study_id           = subject_id,
                    series_id          = series_id,
                    is_3d              = False,
                    num_frames         = _FRAMES_PER_SWEEP,
                    frame_indices      = frame_idx,
                    is_cine            = True,
                    has_temporal_order = True,
                    view_type          = "fetal_abdomen_sweep",
                    has_mask           = has_mask,
                    task_type          = task_type,
                    ssl_stream         = "video",
                    is_promptable      = has_mask,
                    source_meta        = {
                        "uuid":       uuid,
                        "subject_id": subject_id,
                        "sweep_idx":  sweep_idx,
                        "ac_mm":      ac_mm,
                        "frame_start": frame_idx[0],
                        "frame_end":   frame_idx[-1] + 1,
                    },
                )
