"""
data/adapters/maternal_fetal/acouslic.py  ·  ACOUSLIC-AI adapter
=================================================================

ACOUSLIC-AI: fetal abdominal-circumference ultrasound sweep dataset
(Grand Challenge 2023).  Each sample is a 3-D MetaImage sweep of a
fetal abdomen paired with a binary segmentation mask.

Layout on disk:
  {root}/
  └── acouslic-ai-train-set/
      ├── images/stacked_fetal_ultrasound/<uuid>.mha   (300 sweeps)
      ├── masks/stacked_fetal_abdomen/<uuid>.mha       (300 masks)
      └── circumferences/
          └── fetal_abdominal_circumferences_per_sweep.csv

Volume format (both image and mask):
  MetaImage (.mha), uint8
  Shape (W=744, H=562, N=840) on disk → numpy (840, 562, 744) after loading.
  Third axis is the stacked 2-D sweep frames; isotropic spacing 0.28 mm.

CSV columns:
  uuid, subject_id, sweep_1_ac_mm … sweep_6_ac_mm
  Each row is sparse (typically 1–3 of the 6 sweep columns populated).
  Multiple rows (sweeps) may share the same subject_id.

Split strategy: group by subject_id to avoid leakage across sweeps of the
same patient.  Splitting on uuid would contaminate val/test sets.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_SWEEP_COLS = [f"sweep_{i}_ac_mm" for i in range(1, 7)]
_N_FRAMES   = 840   # constant across all ACOUSLIC sweeps


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
        # Accept root pointing to the parent; descend into acouslic-ai-train-set/
        candidate = root / "acouslic-ai-train-set"
        if candidate.is_dir():
            return candidate
        # Accept root already being acouslic-ai-train-set/
        if (root / "images" / "stacked_fetal_ultrasound").is_dir():
            return root
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'acouslic-ai-train-set/' under {root}"
        )

    def _load_csv(self) -> Dict[str, dict]:
        """Return {uuid: {subject_id, ac_mm}} from the circumferences CSV."""
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
                ac_vals: List[float] = []
                for col in _SWEEP_COLS:
                    raw = row.get(col, "").strip()
                    if raw:
                        try:
                            ac_vals.append(float(raw))
                        except ValueError:
                            pass
                out[uuid] = {
                    "subject_id": subject_id,
                    "ac_mm": float(sum(ac_vals) / len(ac_vals)) if ac_vals else None,
                    "n_ac_vals": len(ac_vals),
                }
        return out

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

        # Build subject-level split map to avoid leakage across sweeps.
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
            ac_mm      = row.get("ac_mm")
            n_ac_vals  = row.get("n_ac_vals", 0)

            mask_path = mask_dir / img_path.name
            has_mask  = mask_path.exists()
            split     = self.split_override or subject_split.get(subject_id, "train")

            if has_mask:
                instance = self._make_instance(
                    instance_id    = uuid,
                    label_raw      = "fetal_abdomen",
                    label_ontology = "fetal_abdomen",
                    mask_path      = str(mask_path),
                    is_promptable  = True,
                    measurement_mm = ac_mm,
                )
                task_type = "segmentation"
            else:
                instance = self._make_instance(
                    instance_id    = uuid,
                    label_raw      = "fetal_abdomen",
                    label_ontology = "fetal_abdomen",
                    is_promptable  = False,
                    measurement_mm = ac_mm,
                )
                task_type = "ssl_only"

            yield self._make_entry(
                str(img_path),
                split         = split,
                modality      = "volume",
                instances     = [instance],
                study_id      = subject_id,
                series_id     = uuid,
                is_3d         = True,
                num_frames    = _N_FRAMES,
                view_type     = "fetal_abdomen_sweep",
                has_mask      = has_mask,
                task_type     = task_type,
                ssl_stream    = "image",
                is_promptable = has_mask,
                source_meta   = {
                    "uuid":       uuid,
                    "subject_id": subject_id,
                    "ac_mm":      ac_mm,
                    "n_ac_vals":  n_ac_vals,
                },
            )
