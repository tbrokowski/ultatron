"""
data/adapters/maternal_fetal/fast_u_net.py  ·  Fast-U-Net fetal US adapter
==========================================================================

Fast-U-Net: fetal abdominal circumference (AC) and head circumference (HC)
segmentation dataset bundled with the Fast U-Net paper model.

On-disk layout (store: fetal/Fast-U-Net-main/):
  Fast-U-Net-main/
  ├── Dataset/
  │   ├── AC/
  │   │   ├── train.txt / val.txt   ← "img_path mask_path" per line
  │   │   └── resized_data/image/, resized_data/mask/
  │   └── HC/
  │       ├── train.txt / val.txt
  │       └── resized_data/image/, resized_data/mask/
  └── test_data/
      ├── image/   ← *.jpg
      └── mask/    ← paired by stem

Split files list paths relative to each plane directory (AC/ or HC/).
When the .rar archives have not been extracted, only test_data/ is indexed.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_PLANES: Tuple[Tuple[str, str, str, str], ...] = (
    ("AC", "fetal_abdomen", "abdominal_circumference", "fetal_abdomen"),
    ("HC", "fetal_head",    "head_circumference",      "fetal_head"),
)
_SPLIT_FILES = (("train.txt", "train"), ("val.txt", "val"))


class FastUNetAdapter(BaseAdapter):
    DATASET_ID = "Fast-U-Net"
    ANATOMY_FAMILY = "fetal"
    SONODQS = "silver"
    DOI = ""

    def __init__(self, root: str | Path, split_override: Optional[str] = None):
        super().__init__(self._resolve_dataset_root(root), split_override=split_override)

    @classmethod
    def _resolve_dataset_root(cls, root: str | Path) -> Path:
        root = Path(root)
        if (root / "Dataset").is_dir() or (root / "test_data").is_dir():
            return root
        for name in ("Fast-U-Net-main", "Fast-U-Net"):
            candidate = root / name
            if candidate.is_dir():
                return candidate
        if root.name in {"Fast-U-Net-main", "Fast-U-Net"}:
            return root
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected Dataset/ or test_data/ under {root}"
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        plane_entries = list(self._iter_plane_entries())
        test_entries = list(self._iter_test_entries())

        if not plane_entries and not test_entries:
            log.warning(
                "%s: no indexed samples under %s — extract AC/HC .rar archives or add test_data/",
                self.DATASET_ID,
                self.root,
            )
            return

        yield from plane_entries
        yield from test_entries

    def _warn_if_archives_unextracted(self, plane_dir: Path) -> None:
        """Split lists reference resized_data/, but upstream ships images in .rar files."""
        image_dir = plane_dir / "resized_data" / "image"
        if image_dir.is_dir() and any(image_dir.iterdir()):
            return
        rars = sorted(plane_dir.glob("*.rar"))
        if not rars:
            return
        split_lines = 0
        for split_file, _ in _SPLIT_FILES:
            split_path = plane_dir / split_file
            if split_path.exists():
                split_lines += sum(
                    1 for line in split_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                )
        if split_lines == 0:
            return
        log.warning(
            "%s %s: %d train/val samples listed but resized_data/ is missing "
            "(%d .rar archives on disk). Extract with:\n"
            "  python scripts/unpack_fast_u_net_rars.py %s",
            self.DATASET_ID,
            plane_dir.name,
            split_lines,
            len(rars),
            self.root,
        )

    def _iter_plane_entries(self) -> Iterator[USManifestEntry]:
        dataset_dir = self.root / "Dataset"
        if not dataset_dir.is_dir():
            return

        for plane, anatomy, label_raw, label_ontology in _PLANES:
            plane_dir = dataset_dir / plane
            if not plane_dir.is_dir():
                continue
            self._warn_if_archives_unextracted(plane_dir)
            for split_file, split in _SPLIT_FILES:
                split_path = plane_dir / split_file
                if not split_path.exists():
                    continue
                for line in split_path.read_text(encoding="utf-8").splitlines():
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    img_path = plane_dir / parts[0]
                    mask_path = plane_dir / parts[1]
                    if not img_path.is_file():
                        continue
                    split_label = self.split_override or split
                    yield self._make_sample_entry(
                        img_path=img_path,
                        mask_path=mask_path if mask_path.is_file() else None,
                        split=split_label,
                        plane=plane,
                        anatomy=anatomy,
                        label_raw=label_raw,
                        label_ontology=label_ontology,
                        source_split=split,
                    )

    def _iter_test_entries(self) -> Iterator[USManifestEntry]:
        image_dir = self.root / "test_data" / "image"
        mask_dir = self.root / "test_data" / "mask"
        if not image_dir.is_dir():
            return

        image_paths = sorted(image_dir.glob("*"))
        n = len(image_paths)
        for idx, img_path in enumerate(image_paths):
            if not img_path.is_file():
                continue
            mask_path = mask_dir / img_path.name
            split = self.split_override or "test"
            yield self._make_sample_entry(
                img_path=img_path,
                mask_path=mask_path if mask_path.is_file() else None,
                split=split,
                plane="test",
                anatomy="fetal",
                label_raw="fetal_structure",
                label_ontology="fetal_abdomen",
                source_split="test",
                sample_index=idx,
                sample_total=n,
            )

    def _make_sample_entry(
        self,
        *,
        img_path: Path,
        mask_path: Optional[Path],
        split: str,
        plane: str,
        anatomy: str,
        label_raw: str,
        label_ontology: str,
        source_split: str,
        sample_index: Optional[int] = None,
        sample_total: Optional[int] = None,
    ) -> USManifestEntry:
        has_mask = mask_path is not None and mask_path.is_file()
        instance = self._make_instance(
            instance_id=f"{plane}_{img_path.stem}",
            label_raw=label_raw,
            label_ontology=label_ontology,
            mask_path=str(mask_path) if has_mask else None,
            is_promptable=has_mask,
        )
        return self._make_entry(
            str(img_path),
            split=split,
            modality="image",
            instances=[instance],
            study_id=f"fast_unet_{plane}_{img_path.stem}",
            view_type=plane.lower(),
            has_mask=has_mask,
            task_type="segmentation" if has_mask else "ssl_only",
            ssl_stream="image",
            is_promptable=has_mask,
            source_meta={
                "plane": plane,
                "original_split": source_split,
                "anatomy_hint": anatomy,
                **({"sample_index": sample_index, "sample_total": sample_total}
                   if sample_index is not None else {}),
            },
        )
