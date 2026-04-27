"""
data/adapters/breast/breast_adapter.py  ·  BrEaST breast ultrasound adapter
============================================================================
BrEaST (Breast Ultrasound Dataset): 252 images / 266 masks.
  Labels:  binary tumour segmentation masks (no benign/malignant distinction)
  Source:  US-43d
  SonoDQS: silver

Dataset layout on disk
-----------------------
BrEaST/
├── case001.png           ← image
├── case001_tumor.png     ← binary segmentation mask
├── case002.png
├── case002_tumor.png
└── ...

Key observations
----------------
- Everything is flat in a single directory.
- Pairing: case{N}.png  ↔  case{N}_tumor.png
- No class label (benign/malignant) — segmentation only.
- Some cases may have multiple tumors → multiple mask files (case001_tumor.png,
  case001_tumor2.png etc.) — we collect all masks per case.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

# Matches case001.png but NOT case001_tumor.png
_IMAGE_RE = re.compile(r"^(case\d+)\.png$", re.IGNORECASE)
# Matches case001_tumor.png, case001_tumor2.png, etc.
_MASK_RE  = re.compile(r"^(case\d+)_tumor\d*\.png$", re.IGNORECASE)


class BrEaSTAdapter(BaseAdapter):
    """
    Adapter for the BrEaST breast ultrasound dataset.

    Parameters
    ----------
    root : str | Path
        Root directory containing all case*.png and case*_tumor*.png files.
    split_override : str, optional
        If set, all entries get this split label.
    """

    DATASET_ID     = "BrEaST"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1038/s41597-024-03221-3"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        """Yield one USManifestEntry per case image."""

        # ── Collect all images and their masks ────────────────────────────
        all_files = sorted(self.root.glob("*.png"))

        # Build mask index: case_id → list of mask paths
        mask_index: dict[str, list[Path]] = {}
        for f in all_files:
            m = _MASK_RE.match(f.name)
            if m:
                case_id = m.group(1).lower()
                mask_index.setdefault(case_id, []).append(f)

        # Collect images
        images = [f for f in all_files if _IMAGE_RE.match(f.name)]
        n = len(images)

        for i, img_path in enumerate(images):
            case_id   = img_path.stem.lower()   # e.g. "case001"
            masks     = mask_index.get(case_id, [])
            has_mask  = len(masks) > 0
            split     = self._infer_split(case_id, i, n)

            # ── Build one instance per mask (or one ssl instance if no mask) ──
            if has_mask:
                instances = [
                    self._make_instance(
                        instance_id    = f"{case_id}_tumor{j}",
                        label_raw      = "tumor",
                        label_ontology = "breast_lesion",
                        mask_path      = str(m),
                        is_promptable  = True,
                    )
                    for j, m in enumerate(sorted(masks))
                ]
                task_type = "segmentation"
            else:
                instances = [
                    self._make_instance(
                        instance_id    = case_id,
                        label_raw      = "unknown",
                        label_ontology = "unknown",
                        mask_path      = None,
                        is_promptable  = False,
                    )
                ]
                task_type = "ssl_only"

            yield self._make_entry(
                str(img_path),
                split,
                modality      = "image",
                instances     = instances,
                has_mask      = has_mask,
                task_type     = task_type,
                ssl_stream    = "image",
                is_promptable = has_mask,
                probe_type    = "linear",
                source_meta   = {
                    "case_id":    case_id,
                    "num_masks":  len(masks),
                    "doi":        self.DOI,
                },
            )
