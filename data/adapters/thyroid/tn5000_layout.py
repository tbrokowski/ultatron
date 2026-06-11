"""
TN5000 on-disk layout helpers.

TN5000: PASCAL VOC thyroid nodule detection/classification dataset.
  {root}/Main data/JPEGImages/*.jpg
  {root}/Main data/Annotations/*.xml
  {root}/Main data/ImageSets/Main/{train,val,test}.txt
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

log = logging.getLogger(__name__)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class TN5000Sample:
    stem: str
    image_path: Path
    xml_path: Path
    split: str
    bbox_xyxy: Tuple[float, float, float, float]
    nodule_label: int
    width: int
    height: int


def resolve_data_dir(root: Path) -> Path:
    candidates = [
        root / "Main data",
        root,
    ]
    for candidate in candidates:
        if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
            return candidate
    raise FileNotFoundError(
        f"TN5000: expected JPEGImages/ and Annotations/ under {root}"
    )


def _load_split_map(data_dir: Path) -> Dict[str, str]:
    split_dir = data_dir / "ImageSets" / "Main"
    mapping: Dict[str, str] = {}
    for split_name in ("train", "val", "test"):
        split_file = split_dir / f"{split_name}.txt"
        if not split_file.exists():
            continue
        for line in split_file.read_text().splitlines():
            stem = line.strip()
            if stem:
                mapping[stem] = split_name
    return mapping


def _parse_voc_annotation(xml_path: Path) -> Optional[dict]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        log.warning("TN5000: failed to parse %s — %s", xml_path.name, exc)
        return None

    size = root.find("size")
    width = int(size.findtext("width", "0")) if size is not None else 0
    height = int(size.findtext("height", "0")) if size is not None else 0

    objects = root.findall("object")
    if not objects:
        return None

    obj = objects[0]
    name = (obj.findtext("name") or "").strip()
    try:
        nodule_label = int(name)
    except ValueError:
        nodule_label = -1

    bndbox = obj.find("bndbox")
    if bndbox is None:
        return None

    try:
        bbox = (
            float(bndbox.findtext("xmin", "0")),
            float(bndbox.findtext("ymin", "0")),
            float(bndbox.findtext("xmax", "0")),
            float(bndbox.findtext("ymax", "0")),
        )
    except ValueError:
        return None

    return {
        "width": width,
        "height": height,
        "bbox_xyxy": bbox,
        "nodule_label": nodule_label,
    }


def _resolve_image_path(img_dir: Path, stem: str) -> Optional[Path]:
    for ext in _IMG_EXTS:
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def iter_tn5000_samples(
    root: str | Path,
    split: Optional[str] = None,
    split_override: Optional[str] = None,
) -> Iterator[TN5000Sample]:
    data_dir = resolve_data_dir(Path(root))
    img_dir = data_dir / "JPEGImages"
    ann_dir = data_dir / "Annotations"
    split_map = _load_split_map(data_dir)

    xml_files = sorted(ann_dir.glob("*.xml"), key=lambda p: p.stem)
    for xml_path in xml_files:
        stem = xml_path.stem
        parsed = _parse_voc_annotation(xml_path)
        if parsed is None:
            continue

        image_path = _resolve_image_path(img_dir, stem)
        if image_path is None:
            log.debug("TN5000: missing image for %s", stem)
            continue

        sample_split = split_override or split_map.get(stem, "train")
        if split is not None and sample_split != split:
            continue

        yield TN5000Sample(
            stem=stem,
            image_path=image_path,
            xml_path=xml_path,
            split=sample_split,
            bbox_xyxy=parsed["bbox_xyxy"],
            nodule_label=parsed["nodule_label"],
            width=parsed["width"],
            height=parsed["height"],
        )
