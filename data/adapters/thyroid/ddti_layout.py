"""
DDTI on-disk layout helpers.

Digital Database of Thyroid Ultrasound Images (DDTI):
  {root}/archive/{case}.xml
  {root}/archive/{case}_{image_idx}.jpg

Each XML case file stores TI-RADS metadata and one or more <mark> entries.
Polygon annotations live in the <svg> field as a JSON list of freehand regions.
"""
from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

log = logging.getLogger(__name__)

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass(frozen=True)
class DDTISample:
    case_id: str
    image_idx: str
    image_path: Path
    xml_path: Path
    polygons: List[List[List[float]]]
    tirads_raw: str
    tirads_label: Optional[int]
    composition: str
    echogenicity: str
    margins: str
    calcifications: str
    age: str
    sex: str


def resolve_data_dir(root: Path) -> Path:
    archive = root / "archive"
    if archive.is_dir():
        return archive
    if any(root.glob("*.xml")):
        return root
    raise FileNotFoundError(f"DDTI: expected archive/ or *.xml under {root}")


def map_tirads_label(raw: Optional[str]) -> Optional[int]:
    """Map DDTI TI-RADS strings to thyroid_tirads ordinal indices."""
    value = (raw or "").strip().lower()
    if not value:
        return None
    if value == "2":
        return 1
    if value == "3":
        return 2
    if value.startswith("4"):
        return 3
    if value == "5":
        return 4
    return None


def _parse_svg_polygons(svg_text: str) -> List[List[List[float]]]:
    if not svg_text:
        return []
    try:
        regions = json.loads(svg_text)
    except json.JSONDecodeError:
        log.debug("DDTI: invalid SVG JSON")
        return []

    polygons: List[List[List[float]]] = []
    if not isinstance(regions, list):
        return polygons

    for region in regions:
        if not isinstance(region, dict):
            continue
        points = region.get("points")
        if not isinstance(points, list) or len(points) < 3:
            continue
        polygon = []
        for point in points:
            if not isinstance(point, dict):
                continue
            try:
                polygon.append([float(point["x"]), float(point["y"])])
            except (KeyError, TypeError, ValueError):
                continue
        if len(polygon) >= 3:
            polygons.append(polygon)
    return polygons


def _resolve_image_path(data_dir: Path, case_id: str, image_idx: str) -> Optional[Path]:
    stem = f"{case_id}_{image_idx}"
    for ext in _IMG_EXTS:
        candidate = data_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _parse_case_xml(xml_path: Path, data_dir: Path) -> List[DDTISample]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        log.warning("DDTI: failed to parse %s — %s", xml_path.name, exc)
        return []

    case_id = (root.findtext("number") or xml_path.stem).strip()
    tirads_raw = (root.findtext("tirads") or "").strip()
    tirads_label = map_tirads_label(tirads_raw)

    meta = {
        "composition":     (root.findtext("composition") or "").strip(),
        "echogenicity":    (root.findtext("echogenicity") or "").strip(),
        "margins":         (root.findtext("margins") or "").strip(),
        "calcifications":  (root.findtext("calcifications") or "").strip(),
        "age":             (root.findtext("age") or "").strip(),
        "sex":             (root.findtext("sex") or "").strip(),
    }

    samples: List[DDTISample] = []
    for mark in root.findall("mark"):
        image_idx = (mark.findtext("image") or "").strip()
        if not image_idx:
            continue
        image_path = _resolve_image_path(data_dir, case_id, image_idx)
        if image_path is None:
            log.debug("DDTI: missing image for case %s mark %s", case_id, image_idx)
            continue

        polygons = _parse_svg_polygons(mark.findtext("svg") or "")
        if not polygons:
            continue

        samples.append(
            DDTISample(
                case_id=case_id,
                image_idx=image_idx,
                image_path=image_path,
                xml_path=xml_path,
                polygons=polygons,
                tirads_raw=tirads_raw,
                tirads_label=tirads_label,
                **meta,
            )
        )
    return samples


def iter_ddti_samples(
    root: str | Path,
    split_override: Optional[str] = None,
    infer_split=None,
) -> Iterator[tuple[DDTISample, str]]:
    """
    Yield ``(sample, split)`` for all annotated DDTI images.

    Splitting is case-level (by sorted case id) unless ``split_override`` is set.
    """
    data_dir = resolve_data_dir(Path(root))
    xml_files = sorted(data_dir.glob("*.xml"), key=lambda p: p.stem)
    case_ids = sorted({p.stem for p in xml_files})
    case_to_idx = {case_id: idx for idx, case_id in enumerate(case_ids)}

    for xml_path in xml_files:
        for sample in _parse_case_xml(xml_path, data_dir):
            if split_override:
                split = split_override
            else:
                split = infer_split(
                    sample.case_id,
                    case_to_idx[sample.case_id],
                    len(case_ids),
                )
            yield sample, split
