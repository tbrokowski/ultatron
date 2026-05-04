"""
data/adapters/maternal_fetal/psfhs.py  ·  PSFHS adapter
=======================================================

PSFHS contains paired MetaImage fetal/intrapartum ultrasound images and
categorical segmentation masks.

Layout on disk:
  {root}/
  └── PSFHS/                  (sometimes root points here directly)
      ├── image_mha/<id>.mha
      └── label_mha/<id>.mha

Images are 8-bit 3-channel MHA files. Labels are 8-bit 2-D categorical masks
with foreground values:

  1 = pubic symphysis
  2 = fetal head

Two Instance objects are emitted per labelled image, both pointing to the same
categorical label_mha path.  mask_channel stores the categorical label value,
matching FH-PS-AOP.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


STRUCTURES: List[Tuple[str, str, int]] = [
    ("pubic_symphysis", "pubic_symphysis", 1),
    ("fetal_head",      "fetal_head",      2),
]


class PSFHSAdapter(BaseAdapter):
    DATASET_ID     = "PSFHS"
    ANATOMY_FAMILY = "intrapartum"
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
        if (root / "image_mha").is_dir():
            return root
        candidate = root / cls.DATASET_ID
        if (candidate / "image_mha").is_dir():
            return candidate
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "image_mha").is_dir():
                    return child
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected 'image_mha/' under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        img_dir = self.root / "image_mha"
        label_dir = self.root / "label_mha"

        if not img_dir.exists():
            raise FileNotFoundError(f"PSFHS: image directory not found at {img_dir}")

        label_index: Dict[str, Path] = (
            {p.stem: p for p in label_dir.glob("*.mha")}
            if label_dir.exists() else {}
        )

        images = sorted(img_dir.glob("*.mha"))
        n = len(images)

        for i, img_path in enumerate(images):
            stem = img_path.stem
            label_path = label_index.get(stem)
            has_mask = label_path is not None
            split = self.split_override or self._infer_split(stem, i, n)
            image_header = self._read_mha_header(img_path)
            label_header = self._read_mha_header(label_path) if label_path else {}

            if has_mask:
                instances = [
                    self._make_instance(
                        instance_id=f"{stem}_{struct_name}",
                        label_raw=struct_name,
                        label_ontology=label_ontology,
                        mask_path=str(label_path),
                        mask_channel=class_label,
                        is_promptable=True,
                    )
                    for struct_name, label_ontology, class_label in STRUCTURES
                ]
                task_type = "segmentation"
            else:
                instances = []
                task_type = "ssl_only"

            yield self._make_entry(
                str(img_path),
                split=split,
                modality="image",
                instances=instances,
                study_id=stem,
                series_id=stem,
                view_type="intrapartum_transperineal",
                has_mask=has_mask,
                task_type=task_type,
                ssl_stream="image",
                is_promptable=has_mask,
                source_meta={
                    "stem": stem,
                    "structures": [s[0] for s in STRUCTURES] if has_mask else [],
                    "image_dim_size": image_header.get("DimSize"),
                    "image_element_type": image_header.get("ElementType"),
                    "image_spacing": image_header.get("ElementSpacing"),
                    "image_origin": image_header.get("Offset") or image_header.get("Origin"),
                    "image_channels": image_header.get("ElementNumberOfChannels") or "3",
                    "label_dim_size": label_header.get("DimSize"),
                    "label_element_type": label_header.get("ElementType"),
                    "label_spacing": label_header.get("ElementSpacing"),
                    "label_origin": label_header.get("Offset") or label_header.get("Origin"),
                },
            )

    @staticmethod
    def _read_mha_header(path: Optional[Path]) -> Dict[str, str]:
        if path is None or not path.exists():
            return {}

        header: Dict[str, str] = {}
        with path.open("rb") as f:
            for raw_line in f:
                line = raw_line.decode("ascii", errors="ignore").strip()
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                header[key] = value.strip()
                if key == "ElementDataFile":
                    break
        return header
