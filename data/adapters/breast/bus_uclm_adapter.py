"""
data/adapters/breast/bus_uclm_adapter.py  ·  BUS-UCLM breast ultrasound adapter
=================================================================================
BUS-UCLM (Vallez et al. 2024): 683 images from 38 patients.
  174 benign, 90 malignant, 419 normal
  Source:  https://doi.org/10.17632/7fvgj4jsp7.1
  License: CC BY NC 3.0
  SonoDQS: silver

Dataset layout on disk
-----------------------
BUS-UCLM/
├── images/
│   ├── ALWI_000.png    ← prefix = patient code, no class info in filename
│   ├── ANAT_000.png
│   └── ...
└── masks/
    ├── ALWI_000.png    ← RGB mask: green=benign, red=malignant, black=normal
    ├── ANAT_000.png
    └── ...

Key observations
----------------
- Label is encoded in mask pixel color, NOT in the filename prefix.
- Masks are RGB PNGs:
    green channel dominant  → benign lesion
    red channel dominant    → malignant lesion
    all black               → normal (no lesion)
- Normal images have a mask file (all black) — no separate normal folder.
- We detect class by sampling the dominant non-black color in the mask.
- For segmentation: binarize the relevant channel (green or red) as the mask.
- Normal images → task_type = "classification", has_mask = False.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

# ── Color thresholds for mask classification ───────────────────────────────
# A pixel is "colored" if its dominant channel exceeds this value
_COLOR_THRESHOLD = 50   # out of 255
_MIN_COLORED_PIXELS = 5  # minimum pixels to confirm a lesion exists


def _classify_mask(mask_path: Path) -> str:
    """
    Read an RGB mask and return 'benign', 'malignant', or 'normal'.

    green dominant → benign
    red dominant   → malignant
    all black      → normal
    """
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(mask_path).convert("RGB")
        arr = np.array(img)          # shape: (H, W, 3)  — R, G, B

        r = arr[:, :, 0].astype(int)
        g = arr[:, :, 1].astype(int)
        b = arr[:, :, 2].astype(int)

        # Count pixels where green is clearly dominant
        green_pixels = int(((g > _COLOR_THRESHOLD) & (g > r) & (g > b)).sum())
        # Count pixels where red is clearly dominant
        red_pixels   = int(((r > _COLOR_THRESHOLD) & (r > g) & (r > b)).sum())

        if green_pixels >= _MIN_COLORED_PIXELS and green_pixels >= red_pixels:
            return "benign"
        if red_pixels >= _MIN_COLORED_PIXELS:
            return "malignant"
        return "normal"

    except Exception:
        # If PIL/numpy not available or mask unreadable → treat as normal
        return "normal"


def _extract_binary_mask(mask_path: Path, cls: str, out_path: Path) -> bool:
    """
    Extract a binary (0/255) PNG from the RGB mask for the given class channel.
    Saves to out_path. Returns True if successful.

    We do NOT save anything for normal (no lesion mask needed).
    """
    if cls == "normal":
        return False
    try:
        from PIL import Image
        import numpy as np

        img = Image.open(mask_path).convert("RGB")
        arr = np.array(img)

        r = arr[:, :, 0].astype(int)
        g = arr[:, :, 1].astype(int)
        b = arr[:, :, 2].astype(int)

        if cls == "benign":
            binary = ((g > _COLOR_THRESHOLD) & (g > r) & (g > b)).astype(np.uint8) * 255
        else:  # malignant
            binary = ((r > _COLOR_THRESHOLD) & (r > g) & (r > b)).astype(np.uint8) * 255

        Image.fromarray(binary, mode="L").save(out_path)
        return True

    except Exception:
        return False


# ── Label ontology mapping ─────────────────────────────────────────────────
CLASSES = {
    "benign":    ("benign_lesion",    "breast_lesion_benign"),
    "malignant": ("malignant_lesion", "breast_lesion_malignant"),
    "normal":    ("normal",           "breast_normal"),
}


class BUSUCLMAdapter(BaseAdapter):
    """
    Adapter for the BUS-UCLM breast ultrasound dataset (Vallez et al. 2024).

    Parameters
    ----------
    root : str | Path
        Root directory containing images/ and masks/ subdirectories.
    split_override : str, optional
        If set, all entries get this split label.
    binary_mask_dir : str | Path, optional
        Directory to cache extracted binary masks.
        Defaults to root / "masks_binary".
        Set to None to use the raw RGB mask path directly
        (useful if PIL/numpy are unavailable).
    """

    DATASET_ID     = "BUS-UCLM"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.17632/7fvgj4jsp7.1"

    def __init__(
        self,
        root: str | Path,
        split_override: Optional[str] = None,
        binary_mask_dir: Optional[str | Path] = "masks_binary",
    ):
        super().__init__(root=root, split_override=split_override)

        self.images_dir      = self.root / "images"
        self.masks_dir       = self.root / "masks"
        self.binary_mask_dir = (
            Path(binary_mask_dir) if binary_mask_dir and not Path(binary_mask_dir).is_absolute()
            else self.root / "masks_binary"
        )

    # ── Public interface ───────────────────────────────────────────────────

    def iter_entries(self) -> Iterator[USManifestEntry]:
        """Yield one USManifestEntry per image in images/."""

        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"BUS-UCLM: images/ not found under {self.root}"
            )

        all_images = sorted(self.images_dir.glob("*.png"))
        n = len(all_images)

        for i, img_path in enumerate(all_images):
            fname      = img_path.name
            mask_path  = self.masks_dir / fname
            has_rgb_mask = mask_path.exists()

            # ── Classify via mask color ────────────────────────────────────
            if has_rgb_mask:
                cls = _classify_mask(mask_path)
            else:
                cls = "normal"

            label_raw, label_ontology = CLASSES[cls]

            # ── Binary mask extraction ─────────────────────────────────────
            binary_mask_path = None
            has_mask = False

            if has_rgb_mask and cls != "normal":
                self.binary_mask_dir.mkdir(parents=True, exist_ok=True)
                bin_path = self.binary_mask_dir / fname
                if bin_path.exists() or _extract_binary_mask(mask_path, cls, bin_path):
                    binary_mask_path = str(bin_path)
                    has_mask = True

            # ── Split ─────────────────────────────────────────────────────
            split = self._infer_split(img_path.stem, i, n)

            # ── Task type ─────────────────────────────────────────────────
            if cls == "normal":
                task_type = "classification"
            elif has_mask:
                task_type = "segmentation"
            else:
                task_type = "classification"

            # ── Instance ──────────────────────────────────────────────────
            instance = self._make_instance(
                instance_id    = img_path.stem,
                label_raw      = label_raw,
                label_ontology = label_ontology,
                mask_path      = binary_mask_path,
                is_promptable  = has_mask,
            )

            # ── Entry ──────────────────────────────────────────────────────
            yield self._make_entry(
                str(img_path),
                split,
                modality      = "image",
                instances     = [instance],
                has_mask      = has_mask,
                task_type     = task_type,
                ssl_stream    = "image",
                is_promptable = has_mask,
                probe_type    = "linear",
                source_meta   = {
                    "cls":           cls,
                    "rgb_mask_path": str(mask_path) if has_rgb_mask else None,
                    "patient_code":  img_path.stem.rsplit("_", 1)[0],
                    "doi":           self.DOI,
                },
            )
