"""
data/adapters/muscle/spinal_cord_injury_us.py  ·  Spinal Cord Injury US adapter
=================================================================================

Spinal Cord Injury Ultrasound — Scientific Reports 2025.
  DOI: https://doi.org/10.1038/s41598-025-16275-z

Two sub-datasets bundled together:

  1. Final dataset for object detection  (Pascal VOC XML annotations)
       {root}/Final dataset for object detection/
         train/  → {stem}.png + {stem}.xml  (co-located)
         val/    → ...
         test/   → ...

  2. SegmentationDataset  (binary PNG masks)
       {root}/SegmentationDataset/
         train_images/ + train_masks/
         val_images/   + val_masks/
         test_images/  + test_masks/

Filename anatomy:
  predict{N}_scaled-A{subject_id:04d}_frame{frame_idx}.png
  e.g. predict4_scaled-A0004_frame1.png

Anatomy   : spine (spinal cord)
Probe     : linear
SonoDQS   : silver (single-centre, preclinical/injury model)
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif"}

# predict4_scaled-A0004_frame1
_FNAME_RE = re.compile(
    r"predict(?P<pred>\d+)_scaled-A(?P<subject>\d+)_frame(?P<frame>\d+)",
    re.IGNORECASE,
)

_DETECTION_DIR  = "Final dataset for object detection"
_SEGMENTATION_DIR = "SegmentationDataset"
_DET_SPLITS  = ("train", "val", "test")
_SEG_SPLITS  = {
    "train": ("train_images", "train_masks"),
    "val":   ("val_images",   "val_masks"),
    "test":  ("test_images",  "test_masks"),
}


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in _IMG_EXTS


def _parse_stem(stem: str) -> tuple[str | None, str | None]:
    """Return (subject_id, frame_idx) from filename stem."""
    m = _FNAME_RE.search(stem)
    if m:
        return m.group("subject"), m.group("frame")
    return None, None


def _parse_voc_xml(xml_path: Path) -> list[dict]:
    """Parse Pascal VOC XML → list of {label, xmin, ymin, xmax, ymax}."""
    boxes = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name", default="spinal_cord")
            bndbox = obj.find("bndbox")
            if bndbox is not None:
                boxes.append({
                    "label": name,
                    "xmin":  float(bndbox.findtext("xmin", "0")),
                    "ymin":  float(bndbox.findtext("ymin", "0")),
                    "xmax":  float(bndbox.findtext("xmax", "0")),
                    "ymax":  float(bndbox.findtext("ymax", "0")),
                })
    except Exception:
        pass
    return boxes


class SpinalCordInjuryUSAdapter(BaseAdapter):
    """
    Adapter for the Spinal Cord Injury Ultrasound dataset (Sci. Reports 2025).

    Yields entries from both sub-datasets:
      - Detection entries: task_type = "detection", instances with bbox
      - Segmentation entries: task_type = "segmentation", instances with mask

    Parameters
    ----------
    root : str | Path
        Root directory containing the two sub-dataset folders.
    split_override : str, optional
        Force all entries to a single split.
    include_detection : bool
        Include object detection sub-dataset (default True).
    include_segmentation : bool
        Include segmentation sub-dataset (default True).
    """

    DATASET_ID     = "SpinalCordInjuryUS"
    ANATOMY_FAMILY = "spine"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1038/s41598-025-16275-z"

    def __init__(
        self,
        root,
        split_override=None,
        include_detection: bool = True,
        include_segmentation: bool = True,
    ):
        super().__init__(root, split_override)
        self._include_det = include_detection
        self._include_seg = include_segmentation

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if self._include_det:
            yield from self._iter_detection()
        if self._include_seg:
            yield from self._iter_segmentation()

    # ── Object detection sub-dataset ─────────────────────────────────────────

    def _iter_detection(self) -> Iterator[USManifestEntry]:
        det_root = self.root / _DETECTION_DIR
        if not det_root.is_dir():
            return

        for split_name in _DET_SPLITS:
            split_dir = det_root / split_name
            if not split_dir.is_dir():
                continue
            split = self.split_override or split_name

            for img_path in sorted(f for f in split_dir.iterdir() if _is_image(f)):
                xml_path = img_path.with_suffix(".xml")
                boxes    = _parse_voc_xml(xml_path) if xml_path.exists() else []

                subject_id, frame_idx = _parse_stem(img_path.stem)

                instances = []
                for b in boxes:
                    inst = self._make_instance(
                        instance_id    = f"{img_path.stem}_{b['label']}",
                        label_raw      = b["label"],
                        label_ontology = "spinal_cord",
                        mask_path      = None,
                        is_promptable  = True,
                    )
                    inst.bbox_xyxy = (b["xmin"], b["ymin"], b["xmax"], b["ymax"])
                    instances.append(inst)

                yield self._make_entry(
                    str(img_path),
                    split,
                    modality      = "image",
                    instances     = instances,
                    has_mask      = False,
                    has_box       = len(boxes) > 0,
                    task_type     = "detection",
                    ssl_stream    = "image",
                    is_promptable = len(boxes) > 0,
                    probe_type    = "linear",
                    source_meta   = {
                        "sub_dataset":  "detection",
                        "subject_id":   subject_id,
                        "frame_idx":    frame_idx,
                        "xml_path":     str(xml_path) if xml_path.exists() else None,
                        "doi":          self.DOI,
                    },
                )

    # ── Segmentation sub-dataset ──────────────────────────────────────────────

    def _iter_segmentation(self) -> Iterator[USManifestEntry]:
        seg_root = self.root / _SEGMENTATION_DIR
        if not seg_root.is_dir():
            return

        for split_name, (img_dir_name, mask_dir_name) in _SEG_SPLITS.items():
            img_dir  = seg_root / img_dir_name
            mask_dir = seg_root / mask_dir_name
            if not img_dir.is_dir():
                continue
            split = self.split_override or split_name

            # Build mask index
            mask_index = {f.stem: f for f in mask_dir.glob("*") if _is_image(f)} \
                if mask_dir.is_dir() else {}

            for img_path in sorted(f for f in img_dir.iterdir() if _is_image(f)):
                mask_path = mask_index.get(img_path.stem)
                has_mask  = mask_path is not None

                subject_id, frame_idx = _parse_stem(img_path.stem)

                instances = []
                if has_mask:
                    instances.append(self._make_instance(
                        instance_id    = img_path.stem,
                        label_raw      = "spinal_cord",
                        label_ontology = "spinal_cord",
                        mask_path      = str(mask_path),
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
                        "sub_dataset": "segmentation",
                        "subject_id":  subject_id,
                        "frame_idx":   frame_idx,
                        "doi":         self.DOI,
                    },
                )
