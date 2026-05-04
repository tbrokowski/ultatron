"""
data/adapters/breast/chinese_us_report_adapter.py
==================================================
Chinese US-Report Dataset — Breast (Mammary) subset.
  7,390 reports + images (breast + thyroid + liver combined).
  This adapter loads the breast (Mammary) subset only.
  Source:  github.com / Alsharid 2025
  SonoDQS: bronze

Dataset layout on disk
-----------------------
Chinese_US_Report/
├── new_Mammary2.json      ← breast metadata: uid, finding, image_path, labels, split
├── new_Liver2.json        ← liver  (not loaded by this adapter)
├── new_Thyroid2.json      ← thyroid (not loaded by this adapter)
├── Mammary_report/        ← breast JPEG images: {uid}_{view}.jpeg
├── Liver_report/
└── Thyroid_report/

JSON schema (new_Mammary2.json)
--------------------------------
{
  "train": [
    {
      "uid": 215168,
      "finding": "双侧乳腺体...",   ← Chinese radiology report text
      "image_path": ["215168_1.jpeg", "215168_2.jpeg"],
      "labels": 1,                  ← integer class label
      "split": "train"
    },
    ...
  ]
}

Key observations
----------------
- Labels are integers — no benign/malignant string mapping available from JSON alone.
  We store the raw int and map to weak ontology labels.
- One patient may have multiple images (views) — we yield one entry per patient
  with all image paths included.
- finding text (Chinese report) stored in source_meta for future text branch use.
- No segmentation masks — classification / weak_label task.
- split field inside JSON takes precedence.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


class ChineseUSReportBreastAdapter(BaseAdapter):
    """
    Adapter for the Chinese US-Report Dataset — breast (Mammary) subset.

    Parameters
    ----------
    root : str | Path
        Root directory containing new_Mammary2.json and Mammary_report/.
    split_override : str, optional
        If set, all entries get this split label.
    """

    DATASET_ID     = "Chinese-US-Report-Breast"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "bronze"
    DOI            = "https://github.com/Alsharid2025"

    def __init__(
        self,
        root: str | Path,
        split_override: Optional[str] = None,
    ):
        super().__init__(root=root, split_override=split_override)
        self.json_path   = self.root / "new_Mammary2.json"
        self.images_dir  = self.root / "Mammary_report"

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_records(self) -> list[dict]:
        """Load all records from new_Mammary2.json."""
        if not self.json_path.exists():
            raise FileNotFoundError(
                f"Chinese-US-Report-Breast: {self.json_path} not found"
            )
        with open(self.json_path, encoding="utf-8") as f:
            data = json.load(f)

        records = []
        # JSON top-level may be {"train": [...]} or a flat list
        if isinstance(data, dict):
            for split_key, entries in data.items():
                if isinstance(entries, list):
                    for e in entries:
                        if "split" not in e:
                            e["split"] = split_key
                        records.append(e)
        elif isinstance(data, list):
            records = data
        return records

    @staticmethod
    def _label_id_to_ontology(label_id: int) -> tuple[str, str]:
        """
        Map integer label to (label_raw, label_ontology).
        Without a full label map, we use weak_label for all.
        Subclasses can override this with a proper mapping once known.
        """
        return f"class_{label_id}", "breast_finding"

    # ── Public interface ───────────────────────────────────────────────────

    def iter_entries(self) -> Iterator[USManifestEntry]:
        """Yield one USManifestEntry per patient record."""

        records = self._load_records()
        n = len(records)

        for i, rec in enumerate(records):
            uid         = str(rec.get("uid", i))
            finding     = rec.get("finding", "")
            img_fnames  = rec.get("image_path", [])
            label_id    = int(rec.get("labels", -1))
            json_split  = rec.get("split", "")

            # Resolve split
            if self.split_override:
                split = self.split_override
            elif json_split in ("train", "val", "test"):
                split = json_split
            else:
                split = self._infer_split(uid, i, n)

            # Build absolute image paths
            img_paths = [
                str(self.images_dir / fname)
                for fname in img_fnames
                if (self.images_dir / fname).exists()
            ]
            if not img_paths:
                # Fallback: include paths even if not verified on disk
                img_paths = [str(self.images_dir / fname) for fname in img_fnames]
            if not img_paths:
                continue

            label_raw, label_ontology = self._label_id_to_ontology(label_id)

            instance = self._make_instance(
                instance_id    = uid,
                label_raw      = label_raw,
                label_ontology = label_ontology,
                mask_path      = None,
                is_promptable  = False,
            )

            yield self._make_entry(
                img_paths,
                split,
                modality      = "image",
                instances     = [instance],
                has_mask      = False,
                task_type     = "weak_label",
                ssl_stream    = "image",
                is_promptable = False,
                study_id      = uid,
                num_frames    = len(img_paths),
                probe_type    = "linear",
                source_meta   = {
                    "uid":        uid,
                    "finding":    finding,   # Chinese report text
                    "label_id":   label_id,
                    "n_views":    len(img_fnames),
                    "doi":        self.DOI,
                },
            )
