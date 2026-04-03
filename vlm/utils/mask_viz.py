"""
vlm/utils/mask_viz.py  ·  Mask visualization utilities
========================================================

Helpers for overlaying SAM2 binary masks on ultrasound images for:
  - Debug visualization during training
  - Creating overlay images for the fallback SAMTok path
  - Reward analysis and logging
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def make_overlay_image(
    image: "PIL.Image.Image",
    mask:  np.ndarray,
    color: Tuple[int, int, int] = (220, 40, 40),
    alpha: float = 0.4,
) -> "PIL.Image.Image":
    """
    Overlay a binary segmentation mask on an image as a semi-transparent tint.

    Parameters
    ----------
    image : PIL Image (RGB)
    mask  : (H, W) bool numpy array
    color : RGB tint colour for the mask overlay (default: red)
    alpha : opacity of the overlay (0 = transparent, 1 = opaque)

    Returns
    -------
    PIL Image with mask overlay
    """
    from PIL import Image
    img_np  = np.array(image.convert("RGB"))
    overlay = img_np.copy().astype(float)

    color_np = np.array(color, dtype=float)
    overlay[mask] = overlay[mask] * (1 - alpha) + color_np * alpha
    overlay = overlay.clip(0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


def make_contour_image(
    image: "PIL.Image.Image",
    mask:  np.ndarray,
    color: Tuple[int, int, int] = (0, 220, 80),
    thickness: int = 2,
) -> "PIL.Image.Image":
    """
    Draw contour outline of a binary mask on an image.

    Parameters
    ----------
    image     : PIL Image (RGB)
    mask      : (H, W) bool numpy array
    color     : RGB contour colour
    thickness : contour line thickness in pixels
    """
    from PIL import Image, ImageDraw
    try:
        import cv2
        img_np   = np.array(image.convert("RGB"))
        mask_u8  = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_np, contours, -1, color, thickness)
        return Image.fromarray(img_np)
    except ImportError:
        # Fallback: simple overlay
        return make_overlay_image(image, mask, color=color, alpha=0.3)


def mask_to_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Convert a binary mask to its tight bounding box [x1, y1, x2, y2].

    Returns None if the mask is empty.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    y1, y2 = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    x1, x2 = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    return x1, y1, x2, y2


def save_debug_visualization(
    image:     "PIL.Image.Image",
    mask:      np.ndarray,
    save_path: str,
    title:     str = "",
):
    """Save a side-by-side original / overlay image for debugging."""
    from PIL import Image, ImageDraw, ImageFont
    overlay = make_overlay_image(image, mask)

    combined = Image.new("RGB", (image.width * 2 + 4, image.height + 24), (30, 30, 30))
    combined.paste(image,   (0,                  24))
    combined.paste(overlay, (image.width + 4,    24))

    draw = ImageDraw.Draw(combined)
    draw.text((4, 4), title or "Original  |  SAM2 Overlay",
              fill=(255, 255, 255))

    import pathlib
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    combined.save(save_path)
