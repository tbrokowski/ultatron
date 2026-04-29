"""
data/adapters/maternal_fetal/fugc.py  ·  FUGC adapter
======================================================

FUGC is a fetal ultrasound semi-supervised segmentation dataset:

  dataset/
  ├── train/
  │   ├── labeled_data/images/
  │   ├── labeled_data/labels/
  │   └── unlabeled_data/images/
  ├── val/images/, val/labels/
  └── test/images/, test/labels/

Images are RGB PNGs and masks are grayscale PNGs with categorical values
{0, 1, 2}.  The dataset files do not document the semantic meaning of label
values 1 and 2, so this adapter exposes them as generic FUGC label channels
rather than assigning pubic-symphysis/fetal-head ontology names prematurely.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


LABEL_CHANNELS: Tuple[Tuple[int, str, str], ...] = (
    (1, "class_1", "fugc_label_1"),
    (2, "class_2", "fugc_label_2"),
)


class FUGCAdapter(BaseAdapter):
    DATASET_ID     = "FUGC"
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
        if cls._looks_like_dataset_root(root):
            return root
        for candidate in (
            root / "dataset",
            root / "FUGC" / "dataset",
        ):
            if cls._looks_like_dataset_root(candidate):
                return candidate
        raise FileNotFoundError(
            f"{cls.DATASET_ID}: expected dataset/{{train,val,test}} under {root}"
        )

    @staticmethod
    def _looks_like_dataset_root(path: Path) -> bool:
        return (
            path.is_dir()
            and (path / "train").is_dir()
            and (path / "val").is_dir()
            and (path / "test").is_dir()
        )

    def iter_entries(self) -> Iterator[USManifestEntry]:
        yield from self._iter_labeled_subset(
            split="train",
            images_dir=self.root / "train" / "labeled_data" / "images",
            labels_dir=self.root / "train" / "labeled_data" / "labels",
            subset="train_labeled",
        )
        yield from self._iter_unlabeled_train()
        yield from self._iter_labeled_subset(
            split="val",
            images_dir=self.root / "val" / "images",
            labels_dir=self.root / "val" / "labels",
            subset="val",
        )
        yield from self._iter_labeled_subset(
            split="test",
            images_dir=self.root / "test" / "images",
            labels_dir=self.root / "test" / "labels",
            subset="test",
        )

    def _iter_labeled_subset(
        self,
        split: str,
        images_dir: Path,
        labels_dir: Path,
        subset: str,
    ) -> Iterator[USManifestEntry]:
        if not images_dir.exists():
            return

        for img_path in sorted(images_dir.glob("*.png")):
            stem = img_path.stem
            mask_path = labels_dir / img_path.name
            has_mask = mask_path.exists()
            instances = []
            if has_mask:
                instances = [
                    self._make_instance(
                        instance_id=f"{stem}_{label_raw}",
                        label_raw=label_raw,
                        label_ontology=label_ontology,
                        mask_path=str(mask_path),
                        mask_channel=channel,
                        is_promptable=True,
                    )
                    for channel, label_raw, label_ontology in LABEL_CHANNELS
                ]

            yield self._make_entry(
                str(img_path),
                split=self.split_override or split,
                modality="image",
                instances=instances,
                study_id=stem,
                series_id=stem,
                view_type="intrapartum_transperineal",
                has_mask=has_mask,
                task_type="segmentation" if has_mask else "ssl_only",
                ssl_stream="image",
                is_promptable=has_mask,
                source_meta={
                    "subset": subset,
                    "stem": stem,
                    "mask_path": str(mask_path) if has_mask else None,
                    "mask_values": [0, 1, 2] if has_mask else None,
                    "label_semantics_documented": False,
                    "possible_label_meanings": {
                        "1": "pubic_symphysis_unverified",
                        "2": "fetal_head_unverified",
                    },
                },
            )

    def _iter_unlabeled_train(self) -> Iterator[USManifestEntry]:
        images_dir = self.root / "train" / "unlabeled_data" / "images"
        if not images_dir.exists():
            return

        for img_path in sorted(images_dir.glob("*.png")):
            stem = img_path.stem
            yield self._make_entry(
                str(img_path),
                split=self.split_override or "train",
                modality="image",
                instances=[],
                study_id=stem,
                series_id=stem,
                view_type="intrapartum_transperineal",
                has_mask=False,
                task_type="ssl_only",
                ssl_stream="image",
                is_promptable=False,
                source_meta={
                    "subset": "train_unlabeled",
                    "stem": stem,
                    "semi_supervised_unlabeled": True,
                },
            )
