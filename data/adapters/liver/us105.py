"""
data/adapters/liver/us105.py  ·  105US adapter
===============================================

105US (Pancreatic Cancer Liver Metastases Ultrasound): 105 RGB ultrasound
images of liver metastases from pancreatic cancer, each paired with an expert
grayscale segmentation mask.

Layout on disk:
  105US/
  ├── 105 US Images/      # folder name contains spaces
  │   ├── readme.txt
  │   └── <id>.png        # 105 PNGs, zero-padded 3-digit IDs (001.png…)
  └── 105 US Masks/       # folder name contains spaces
      └── <id> G man.png  # 105 grayscale masks, e.g. 001 G man.png

Pairing rule: strip " G man" from mask stem to get the image ID.
  image: 105 US Images/001.png  ↔  mask: 105 US Masks/001 G man.png

Image format: PNG, 8-bit RGB, variable size (6 resolutions: 960×720, 744×576,
  736×576, 850×649, 1280×720, 1170×649).  Image and its mask are always the
  same size — do not hardcode dimensions.

Mask format: PNG, 8-bit grayscale.  The mask is NOT binary — it is a
  grayscale rendering with the tumor region delineated by a radiologist.
  Full 0–255 intensity range is used within the annotated region.
  Thresholding or contour detection is the pipeline's responsibility.
  source_meta["mask_type"] = "grayscale_contour" flags this for downstream.

Task: segmentation of liver metastases (pancreatic cancer origin).
  Median Dice > 80% reported in the original paper.
  label_ontology = "liver_lesion" (maps to class index 2 in ANATOMY_LABEL_SPACES["liver"]).

Split strategy:
  No predefined split.  Deterministic 80/10/10 on sorted integer IDs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_IMG_DIR  = "105 US Images"
_MASK_DIR = "105 US Masks"
_MASK_SUFFIX = " G man"


class US105Adapter(BaseAdapter):
    DATASET_ID     = "105US"
    ANATOMY_FAMILY = "liver"
    SONODQS        = "silver"
    DOI            = ""

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(
            self._resolve_dataset_root(root),
            split_override=split_override,
        )

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / _IMG_DIR).is_dir():
            return root
        candidate = root / "105US"
        if (candidate / _IMG_DIR).is_dir():
            return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected '{_IMG_DIR}/' under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir  = self.root / _IMG_DIR
        mask_dir = self.root / _MASK_DIR

        if not img_dir.exists():
            raise FileNotFoundError(
                f"105US: image directory not found at {img_dir}"
            )

        # Index masks by integer ID: "001 G man.png" → {"001": Path(...)}
        mask_index: Dict[str, Path] = {}
        if mask_dir.exists():
            for p in mask_dir.glob(f"*{_MASK_SUFFIX}.png"):
                img_id = p.stem[: -len(_MASK_SUFFIX)].strip()
                if img_id:
                    mask_index[img_id] = p

        images = sorted(
            (p for p in img_dir.glob("*.png") if p.stem.isdigit()),
            key=lambda p: int(p.stem),
        )
        n = len(images)

        for i, img_path in enumerate(images):
            img_id    = img_path.stem
            mask_path = mask_index.get(img_id)
            has_mask  = mask_path is not None
            split     = self.split_override or self._infer_split(img_id, i, n)

            if has_mask:
                instance = self._make_instance(
                    instance_id    = img_id,
                    label_raw      = "liver_metastasis",
                    label_ontology = "liver_lesion",
                    mask_path      = str(mask_path),
                    is_promptable  = True,
                )
                task_type = "segmentation"
            else:
                instance = self._make_instance(
                    instance_id    = img_id,
                    label_raw      = "liver_metastasis",
                    label_ontology = "liver_lesion",
                    is_promptable  = False,
                )
                task_type = "ssl_only"

            yield self._make_entry(
                str(img_path),
                split         = split,
                modality      = "image",
                instances     = [instance],
                study_id      = img_id,
                series_id     = img_id,
                view_type     = "liver_bmode",
                has_mask      = has_mask,
                task_type     = task_type,
                ssl_stream    = "image",
                is_promptable = has_mask,
                source_meta   = {
                    "image_id":  img_id,
                    "mask_type": "grayscale_contour" if has_mask else None,
                    "mask_name": mask_path.name if has_mask else None,
                },
            )
