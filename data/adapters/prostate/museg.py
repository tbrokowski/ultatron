"""
data/adapters/prostate/museg.py  ·  MuSeg / µ-RegPro prostate MR-US adapter
=============================================================================

Paired MR and transrectal ultrasound volumes from the µ-RegPro challenge.
On Capstor this release lives under thyroid/MuSeg/ but contains prostate data.

Layout:

    {root}/
        train|val/
            us_images/   case######.nii.gz
            us_labels/   case######.nii.gz  (prostate gland segmentation)
            mr_images/   case######.nii.gz
            mr_labels/   case######.nii.gz
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

log = logging.getLogger(__name__)


class MuSegAdapter(BaseAdapter):
    DATASET_ID = "MuSeg"
    ANATOMY_FAMILY = "prostate"
    SONODQS = "gold"
    DOI = "https://muregpro.github.io/"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        split_dirs = sorted(
            p for p in self.root.iterdir()
            if p.is_dir() and p.name in ("train", "val", "test")
        )
        if not split_dirs:
            log.warning("MuSeg: no train/val/test splits under %s", self.root)
            return

        for split_dir in split_dirs:
            split_name = split_dir.name
            us_dir = split_dir / "us_images"
            label_dir = split_dir / "us_labels"
            if not us_dir.is_dir():
                continue

            volumes = sorted(us_dir.glob("*.nii.gz"))
            for idx, vol_path in enumerate(volumes):
                case_id = vol_path.name.replace(".nii.gz", "")
                split = self.split_override or split_name
                label_path = label_dir / vol_path.name
                instances: List[Instance] = []
                has_mask = label_path.exists()
                if has_mask:
                    instances.append(
                        self._make_instance(
                            instance_id=case_id,
                            label_raw="prostate_gland",
                            label_ontology="prostate_gland",
                            mask_path=str(label_path),
                            is_promptable=True,
                        )
                    )

                yield self._make_entry(
                    str(vol_path),
                    split=split,
                    modality="volume",
                    instances=instances,
                    study_id=case_id,
                    series_id=case_id,
                    is_3d=True,
                    has_mask=has_mask,
                    task_type="segmentation" if has_mask else "ssl_only",
                    ssl_stream="image",
                    is_promptable=has_mask,
                    source_meta={
                        "case_id": case_id,
                        "split_dir": split_name,
                        "has_mr_pair": (split_dir / "mr_images" / vol_path.name).exists(),
                    },
                )
