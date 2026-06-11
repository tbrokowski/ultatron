"""
data/adapters/generic_mask.py  - GenericMaskPairAdapter and _make_generic factory

Provides a generic image+segmentation-mask adapter for datasets that follow
the conventional layout:

    <root>/
        images/          # *.png / *.jpg
        masks/           # *.png with matching filenames (optional)

Also provides _make_generic() for creating dataset-specific subclasses in a
single call, useful for simple datasets and for tests.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List, Optional, Type

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


class GenericMaskPairAdapter(BaseAdapter):
    """
    Generic adapter for datasets with images/ and optional masks/ directories.

    Subclass and override DATASET_ID, ANATOMY_FAMILY, SONODQS as usual.
    Optionally override LABEL_ONTOLOGY and TEXT_TEMPLATE.
    """

    DATASET_ID      = "GenericMaskPair"
    ANATOMY_FAMILY  = "other"
    SONODQS         = "unrated"
    LABEL_ONTOLOGY  = "structure"
    TEXT_TEMPLATE   = ""

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self.root.exists():
            log.warning("%s: root not found at %s", self.DATASET_ID, self.root)
            return

        img_dir  = self.root / "images"
        mask_dir = self.root / "masks"

        if not img_dir.exists():
            img_dir = self.root  # fallback: images directly in root

        image_files = sorted(
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_EXTS
        )

        if not image_files:
            log.warning("%s: no images found in %s", self.DATASET_ID, img_dir)
            return

        n     = len(image_files)
        n_tr  = int(0.80 * n)
        n_val = int(0.10 * n)

        for i, img_path in enumerate(image_files):
            if self.split_override:
                split = self.split_override
            elif i < n_tr:
                split = "train"
            elif i < n_tr + n_val:
                split = "val"
            else:
                split = "test"

            mask_path: Optional[Path] = None
            if mask_dir.exists():
                candidate = mask_dir / img_path.name
                if candidate.exists():
                    mask_path = candidate
                else:
                    # Try without extension mismatch (e.g. .jpg -> .png)
                    for ext in _IMG_EXTS:
                        candidate = mask_dir / (img_path.stem + ext)
                        if candidate.exists():
                            mask_path = candidate
                            break

            instances: List[Instance] = []
            has_mask = mask_path is not None and mask_path.exists()
            if has_mask:
                instances.append(self._make_instance(
                    instance_id    = img_path.stem,
                    label_raw      = self.LABEL_ONTOLOGY,
                    label_ontology = self.LABEL_ONTOLOGY,
                    is_promptable  = True,
                ))

            source_meta = {}
            if self.TEXT_TEMPLATE:
                source_meta["text_label"] = self.TEXT_TEMPLATE

            yield self._make_entry(
                str(img_path),
                split         = split,
                modality      = "image",
                instances     = instances,
                mask_path     = str(mask_path) if has_mask else None,
                task_type     = "segmentation" if has_mask else "ssl_only",
                ssl_stream    = "image",
                is_promptable = has_mask,
                source_meta   = source_meta if source_meta else None,
            )


def _make_generic(
    dataset_id:     str,
    anatomy_family: str,
    sonodqs:        str = "unrated",
    label_ontology: str = "structure",
    text_template:  str = "",
) -> Type[GenericMaskPairAdapter]:
    """
    Factory: create a GenericMaskPairAdapter subclass for a specific dataset.

    Usage
    -----
    MyAdapter = _make_generic("MyDS", "thyroid", "bronze",
                              label_ontology="nodule",
                              text_template="thyroid nodule ultrasound")
    adapter = MyAdapter(root_path)
    """
    return type(
        f"{dataset_id.replace('-', '_')}Adapter",
        (GenericMaskPairAdapter,),
        {
            "DATASET_ID":     dataset_id,
            "ANATOMY_FAMILY": anatomy_family,
            "SONODQS":        sonodqs,
            "LABEL_ONTOLOGY": label_ontology,
            "TEXT_TEMPLATE":  text_template,
        },
    )
