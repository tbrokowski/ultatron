"""Shared helpers for rasterizing thyroid annotations into cached mask PNGs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def rasterize_polygons(
    polygons: Sequence[Sequence[Sequence[float]]],
    width: int,
    height: int,
) -> np.ndarray:
    """Fill one or more closed polygons into a binary uint8 mask."""
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for polygon in polygons:
        if len(polygon) < 3:
            continue
        xy = [(float(x), float(y)) for x, y in polygon]
        draw.polygon(xy, outline=1, fill=1)
    return (np.array(mask_img, dtype=np.uint8) > 0).astype(np.uint8)


def rasterize_bbox(
    bbox_xyxy: Sequence[float],
    width: int,
    height: int,
) -> np.ndarray:
    """Fill an axis-aligned bounding box into a binary uint8 mask."""
    xmin, ymin, xmax, ymax = [float(v) for v in bbox_xyxy]
    xmin = max(0, min(width - 1, xmin))
    ymin = max(0, min(height - 1, ymin))
    xmax = max(0, min(width, xmax))
    ymax = max(0, min(height, ymax))
    mask = np.zeros((height, width), dtype=np.uint8)
    if xmax > xmin and ymax > ymin:
        mask[int(ymin):int(ymax), int(xmin):int(xmax)] = 1
    return mask


def write_mask_png(mask: np.ndarray, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask > 0).astype(np.uint8) * 255).save(out_path)
    return out_path


def ensure_polygon_mask(
    *,
    cache_dir: Path,
    sample_id: str,
    polygons: Sequence[Sequence[Sequence[float]]],
    width: int,
    height: int,
) -> Optional[Path]:
    if not polygons:
        return None
    out_path = cache_dir / f"{sample_id}.png"
    if not out_path.exists():
        mask = rasterize_polygons(polygons, width, height)
        if not mask.any():
            return None
        write_mask_png(mask, out_path)
    return out_path


def ensure_bbox_mask(
    *,
    cache_dir: Path,
    sample_id: str,
    bbox_xyxy: Sequence[float],
    width: int,
    height: int,
) -> Optional[Path]:
    out_path = cache_dir / f"{sample_id}.png"
    if not out_path.exists():
        mask = rasterize_bbox(bbox_xyxy, width, height)
        if not mask.any():
            return None
        write_mask_png(mask, out_path)
    return out_path
