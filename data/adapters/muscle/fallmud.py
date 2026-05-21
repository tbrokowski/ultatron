"""
data/adapters/muscle/fallmud.py  ·  FALLMUD adapter
=====================================================

FALLMUD — "AW-Net: Automatic muscle structure analysis on B-mode
ultrasound images for injury prevention"

Two sub-datasets bundled together:
  NeilCronin    — images as .tif, masks as .jpg (img_00001 naming)
  RyanCunningham — images as .jpg, masks as .jpg (0, 1, 2 ... naming)

Each image has TWO mask types:
  fascicle_masks/     → label_raw = "muscle_fascicle"
  aponeurosis_masks/  → label_raw = "muscle_aponeurosis"

→ 2 instances per entry, task = segmentation.

Dataset layout
--------------
  {root}/
    NeilCronin/
      images/            img_00001.tif, img_00002.tif, ...
      fascicle_masks/    img_00001.tif, ...
      aponeurosis_masks/ img_00001.jpg, ...
    RyanCunningham/
      images/            0.jpg, 1.jpg, ...
      fascicle_masks/    0.jpg, 1.jpg, ...
      aponeurosis_masks/ 0.jpg, 1.jpg, ...
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_IMG_EXTS = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
_SUB_DATASETS = ("NeilCronin", "RyanCunningham")


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in _IMG_EXTS


def _find_mask(mask_dir: Path, stem: str) -> Path | None:
    """Find mask by stem, tolerating extension differences."""
    for ext in _IMG_EXTS:
        candidate = mask_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


class FALLMUDAdapter(BaseAdapter):
    """
    Adapter for the FALLMUD muscle fascicle / aponeurosis dataset.

    Yields one USManifestEntry per image with two instances:
      - fascicle mask    (label_raw = "muscle_fascicle")
      - aponeurosis mask (label_raw = "muscle_aponeurosis")

    Parameters
    ----------
    root : str | Path
        Root directory containing NeilCronin/ and/or RyanCunningham/.
    split_override : str, optional
        Force all entries to a single split.
    """

    DATASET_ID     = "FALLMUD"
    ANATOMY_FAMILY = "muscle"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.17632/3jykz7wz8d.1"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        all_samples: list[tuple[Path, Path | None, Path | None, str]] = []

        # Collect from each sub-dataset that exists on disk
        for sub in _SUB_DATASETS:
            sub_dir = self.root / sub
            if not sub_dir.is_dir():
                # Also try root directly (in case user points to NeilCronin/)
                if (self.root / "images").is_dir():
                    sub_dir = self.root
                    sub = self.root.name
                else:
                    continue

            img_dir   = sub_dir / "images"
            fasc_dir  = sub_dir / "fascicle_masks"
            apon_dir  = sub_dir / "aponeurosis_masks"

            if not img_dir.is_dir():
                continue

            for img_path in sorted(img_dir.iterdir()):
                if not _is_image(img_path):
                    continue
                stem = img_path.stem
                fasc_mask = _find_mask(fasc_dir, stem) if fasc_dir.is_dir() else None
                apon_mask = _find_mask(apon_dir, stem) if apon_dir.is_dir() else None
                all_samples.append((img_path, fasc_mask, apon_mask, sub))

        n = len(all_samples)
        for i, (img_path, fasc_mask, apon_mask, sub) in enumerate(all_samples):
            split    = self._infer_split(img_path.stem, i, n)
            has_mask = (fasc_mask is not None) or (apon_mask is not None)

            instances = []
            if fasc_mask is not None:
                instances.append(self._make_instance(
                    instance_id    = f"{img_path.stem}_fascicle",
                    label_raw      = "muscle_fascicle",
                    label_ontology = "muscle",
                    mask_path      = str(fasc_mask),
                    is_promptable  = True,
                ))
            if apon_mask is not None:
                instances.append(self._make_instance(
                    instance_id    = f"{img_path.stem}_aponeurosis",
                    label_raw      = "muscle_aponeurosis",
                    label_ontology = "muscle",
                    mask_path      = str(apon_mask),
                    is_promptable  = True,
                ))

            yield self._make_entry(
                str(img_path),
                split,
                modality      = "image",
                instances     = instances,
                has_mask      = has_mask,
                task_type     = "segmentation" if has_mask else "ssl_only",
                ssl_stream    = "image",
                is_promptable = has_mask,
                probe_type    = "linear",
                source_meta   = {
                    "sub_dataset": sub,
                    "doi":         self.DOI,
                },
            )
