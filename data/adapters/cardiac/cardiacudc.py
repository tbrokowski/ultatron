"""
data/adapters/cardiac/cardiacudc.py  ·  CardiacUDC adapter
===========================================================

CardiacUDC: Multi-site cardiac ultrasound dataset with apical 4-chamber (A4C)
  NIfTI cine volumes from 6 hospital sites.
  Labels:  normal vs patient (disease) classification in label_all_frame/
  Format:  .nii.gz volumes (T × H × W or H × W × T depending on build)

Actual layout after extraction:
  {root}/cardiacUDC_dataset/
    Site_G_100/         patient{id}-4_image.nii.gz + *_label.nii.gz
    Site_G_20/
    Site_G_29/
    Site_R_126/
    Site_R_52/
    Site_R_73/
    label_all_frame/    normal-{id}-4_image.nii.gz + patient-{id}-4_image.nii.gz
                        (paired with *_label.nii.gz)

The dataset_dir is {root}/cardiacUDC_dataset.  Each "*_image.nii.gz" is one
entry; the matching "*_label.nii.gz" (if present) is the segmentation mask.
Files under label_all_frame with "normal" in the name are class 0 (normal);
"patient" files are class 1 (cardiac disease).

Reference:
  Xu et al., "CardiacUDC: Multi-site Cardiac Ultrasound", 2022.
  Kaggle: https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import Instance, USManifestEntry


class CardiacUDCAdapter(BaseAdapter):
    """
    CardiacUDC adapter.  Yields one volume entry per *_image.nii.gz file.

    Entries under label_all_frame/ carry a normal/patient classification
    label.  Entries under Site_*/ are unlabeled SSL candidates.
    """

    DATASET_ID     = "CardiacUDC"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "silver"
    DOI            = "https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset"

    def _dataset_dir(self) -> Path:
        """Return the inner dataset directory (handles both root and root/cardiacUDC_dataset)."""
        inner = self.root / "cardiacUDC_dataset"
        return inner if inner.exists() else self.root

    def iter_entries(self) -> Iterator[USManifestEntry]:
        ds_dir = self._dataset_dir()
        if not ds_dir.exists():
            raise FileNotFoundError(
                f"CardiacUDC: dataset directory not found at {ds_dir}.\n"
                "Extract the multi-part zip first:\n"
                f"  cd {self.root}\n"
                "  cp cardiacUDC_dataset.change2zip cardiacUDC_dataset.zip\n"
                "  zip -s 0 cardiacUDC_dataset.zip --out cardiacUDC_combined.zip\n"
                "  unzip cardiacUDC_combined.zip"
            )

        image_files = sorted(ds_dir.rglob("*_image.nii.gz"))
        n = len(image_files)

        for i, img_path in enumerate(image_files):
            split = self._infer_split(img_path.stem, i, n)
            if self.split_override:
                split = self.split_override

            mask_path: Optional[str] = None
            label_path = img_path.parent / img_path.name.replace("_image.nii.gz", "_label.nii.gz")
            if label_path.exists():
                mask_path = str(label_path)

            # Determine class from directory and filename
            in_label_dir = "label_all_frame" in img_path.parts
            stem = img_path.name.lower()
            if in_label_dir:
                if "normal" in stem:
                    label_raw      = "normal"
                    label_ontology = "cardiac_normal"
                    cls_label      = 0
                else:
                    label_raw      = "cardiac_disease"
                    label_ontology = "cardiac_disease"
                    cls_label      = 1
                has_label = True
            else:
                label_raw = label_ontology = ""
                cls_label = -1
                has_label = False

            # Site from parent directory name
            site = img_path.parent.name

            # Patient ID from filename: patient1-4_image → patient1
            patient_id = img_path.name.split("-")[0]

            instances: list = []
            if has_label:
                inst_kwargs: dict = dict(
                    instance_id    = img_path.stem,
                    label_raw      = label_raw,
                    label_ontology = label_ontology,
                    anatomy_family = "cardiac",
                    is_promptable  = mask_path is not None,
                )
                if mask_path:
                    inst_kwargs["mask_path"] = mask_path
                if cls_label >= 0:
                    inst_kwargs["classification_label"] = cls_label
                instances.append(Instance(**inst_kwargs))

            yield self._make_entry(
                str(img_path), split,
                modality           = "volume",
                instances          = instances,
                study_id           = patient_id,
                view_type          = "A4C",
                is_cine            = True,
                has_temporal_order = True,
                task_type          = "classification" if has_label else "ssl_only",
                ssl_stream         = "video",
                is_promptable      = mask_path is not None,
                has_mask           = mask_path is not None,
                source_meta        = {
                    "root":       str(self.root),
                    "doi":        self.DOI,
                    "site":       site,
                    "patient_id": patient_id,
                },
            )
