"""
data/adapters/muscle/stmus_nda.py  ·  STMUS NDA (Marzola 2021) adapter
=======================================================================

STMUS NDA — "Deep learning segmentation of transverse musculoskeletal
ultrasound images for neuromuscular disease assessment"
Marzola et al., Computers in Biology and Medicine, 2021.

  ~3,917–4,355 transverse B-mode images, ~4,368 binary segmentation masks.
  3 muscles:
    BB  — biceps brachii      (upper arm)
    TA  — tibialis anterior   (lower leg)
    GM  — gastrocnemius medialis (lower leg / calf)
  1,283 subjects; healthy + neuromuscular disease (NDA) cohorts.
  No temporal order; pure image segmentation dataset.

DOI     : https://doi.org/10.17632/3jykz7wz8d.1
SonoDQS : gold  (public, expert-labelled, clear protocol)
Probe   : linear (high-frequency, 7–14 MHz)

Dataset layout (Mendeley download)
-----------------------------------
The Mendeley zip extracts to one of several common structures.
This adapter handles ALL known variants:

  Variant A — per-muscle subdirs with co-located masks (_mask suffix):
    {root}/BB/img_001.png  +  {root}/BB/img_001_mask.png
    {root}/TA/img_001.png  ...
    {root}/GM/img_001.png  ...

  Variant B — per-muscle subdirs, parallel images/ and masks/ subdirs:
    {root}/BB/images/img_001.png
    {root}/BB/masks/img_001.png

  Variant C — flat with muscle encoded in filename:
    {root}/images/BB_001.png
    {root}/masks/BB_001.png

  Variant D — top-level images/ and masks/ flat (muscle from filename token):
    {root}/images/BB_001.bmp   ->  {root}/masks/BB_001.bmp

The adapter tries A → B → C/D in order.

Label ontology
--------------
  muscle_raw          label_ontology
  BB  biceps_brachii  muscle (→ muscle ANATOMY_LABEL_VOCAB canonical)
  TA  tibialis_anterior   muscle
  GM  gastrocnemius_medialis  muscle

All three → anatomy_family = "muscle", task = segmentation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

# ── Muscle metadata ────────────────────────────────────────────────────────────
# Maps every recognised folder name / filename prefix → canonical info
_MUSCLE_MAP: dict[str, tuple[str, str]] = {
    # key (lowercase) → (label_raw, label_ontology)
    "bb":                        ("biceps_brachii",          "muscle"),
    "biceps_brachii":            ("biceps_brachii",          "muscle"),
    "biceps":                    ("biceps_brachii",          "muscle"),
    "ta":                        ("tibialis_anterior",       "muscle"),
    "tibialis_anterior":         ("tibialis_anterior",       "muscle"),
    "tibialis":                  ("tibialis_anterior",       "muscle"),
    "gm":                        ("gastrocnemius_medialis",  "muscle"),
    "gastrocnemius_medialis":    ("gastrocnemius_medialis",  "muscle"),
    "gastrocnemius":             ("gastrocnemius_medialis",  "muscle"),
    "gastroc":                   ("gastrocnemius_medialis",  "muscle"),
}

_KNOWN_MUSCLE_DIRS = {"bb", "ta", "gm",
                      "biceps_brachii", "tibialis_anterior", "gastrocnemius_medialis",
                      "biceps", "tibialis", "gastrocnemius", "gastroc"}

_IMG_EXTENSIONS = {".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"}
_MASK_SUFFIX_RE = re.compile(r"(_mask|_gt|_label|_seg|_annotation)$", re.IGNORECASE)

# ── Regex to extract muscle token from filename (Variant C/D) ──────────────────
# Matches filenames starting with BB_, TA_, GM_, etc.
_MUSCLE_PREFIX_RE = re.compile(
    r"^(BB|TA|GM|biceps|tibialis|gastrocnemius|gastroc)[_\-]",
    re.IGNORECASE,
)


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in _IMG_EXTENSIONS


def _strip_mask_suffix(stem: str) -> str:
    """Remove trailing _mask / _gt / _label etc. from a stem."""
    return _MASK_SUFFIX_RE.sub("", stem)


def _muscle_info(key: str) -> Optional[tuple[str, str]]:
    """Return (label_raw, label_ontology) for a muscle key string, or None."""
    return _MUSCLE_MAP.get(key.lower().replace("-", "_").replace(" ", "_"))


class STMUSNDAAdapter(BaseAdapter):
    """
    Adapter for the STMUS NDA (Marzola 2021) musculoskeletal ultrasound dataset.

    Yields one ``USManifestEntry`` per image.  Each entry has:
    - modality_type = "image"
    - anatomy_family = "muscle"
    - task_type = "segmentation" (when mask found) or "ssl_only"
    - ssl_stream = "image"
    - one Instance per entry with mask_path and label_ontology = "muscle"
    - source_meta["muscle"] = canonical muscle short name (bb | ta | gm)
    """

    DATASET_ID     = "STMUS-NDA"
    ANATOMY_FAMILY = "muscle"
    SONODQS        = "gold"
    DOI            = "https://doi.org/10.17632/3jykz7wz8d.1"

    # ── Layout detection ───────────────────────────────────────────────────────

    def _detect_layout(self) -> str:
        """
        Return one of: "per_muscle_coloc", "per_muscle_parallel",
                       "flat_named", "flat_images_masks".
        """
        # Check for per-muscle subdirs
        muscle_dirs = [
            d for d in self.root.iterdir()
            if d.is_dir() and d.name.lower() in _KNOWN_MUSCLE_DIRS
        ]
        if muscle_dirs:
            # Sub-layout: co-located masks or parallel images/masks?
            sample_dir = muscle_dirs[0]
            has_images_subdir = (sample_dir / "images").is_dir()
            return "per_muscle_parallel" if has_images_subdir else "per_muscle_coloc"

        # Flat layout: check for images/ and masks/ at root
        if (self.root / "images").is_dir():
            return "flat_images_masks"

        return "flat_named"  # last resort: all files at root

    # ── Entry builders ─────────────────────────────────────────────────────────

    def _iter_per_muscle_coloc(self) -> list[tuple[Path, Optional[Path], str, str]]:
        """
        Variant A: {root}/{MUSCLE}/*.png  with masks as *_mask.png co-located.
        Returns list of (img_path, mask_path|None, label_raw, label_ontology).
        """
        samples = []
        for muscle_dir in sorted(self.root.iterdir()):
            if not muscle_dir.is_dir():
                continue
            info = _muscle_info(muscle_dir.name)
            if info is None:
                continue
            label_raw, label_ontology = info

            # Build mask index: stem_without_suffix → mask path
            mask_index: dict[str, Path] = {}
            for f in muscle_dir.glob("*"):
                if not _is_image(f):
                    continue
                clean = _strip_mask_suffix(f.stem)
                if clean != f.stem:          # it IS a mask file
                    mask_index[clean] = f

            for f in sorted(muscle_dir.glob("*")):
                if not _is_image(f) or _strip_mask_suffix(f.stem) != f.stem:
                    continue  # skip mask files
                mask_path = mask_index.get(f.stem)
                samples.append((f, mask_path, label_raw, label_ontology))
        return samples

    def _iter_per_muscle_parallel(self) -> list[tuple[Path, Optional[Path], str, str]]:
        """
        Variant B: {root}/{MUSCLE}/images/*.png  +  {root}/{MUSCLE}/masks/*.png
        """
        samples = []
        for muscle_dir in sorted(self.root.iterdir()):
            if not muscle_dir.is_dir():
                continue
            info = _muscle_info(muscle_dir.name)
            if info is None:
                continue
            label_raw, label_ontology = info
            img_dir  = muscle_dir / "images"
            mask_dir = muscle_dir / "masks"
            if not img_dir.is_dir():
                continue

            mask_index = {f.stem: f for f in mask_dir.glob("*") if _is_image(f)} \
                if mask_dir.is_dir() else {}

            for f in sorted(img_dir.glob("*")):
                if not _is_image(f):
                    continue
                mask_path = mask_index.get(f.stem) or mask_index.get(
                    _strip_mask_suffix(f.stem)
                )
                samples.append((f, mask_path, label_raw, label_ontology))
        return samples

    def _iter_flat(self, images_dir: Optional[Path] = None,
                   masks_dir: Optional[Path] = None) -> list[tuple[Path, Optional[Path], str, str]]:
        """
        Variants C/D: flat layout.  Muscle type inferred from filename prefix.
        """
        img_dir  = images_dir or self.root
        msk_dir  = masks_dir  or self.root
        samples  = []

        mask_index: dict[str, Path] = {}
        for f in msk_dir.glob("*"):
            if _is_image(f):
                mask_index[f.stem] = f
                mask_index[_strip_mask_suffix(f.stem)] = f

        for f in sorted(img_dir.glob("*")):
            if not _is_image(f):
                continue
            if _strip_mask_suffix(f.stem) != f.stem:
                continue   # skip mask files stored in the same dir

            m = _MUSCLE_PREFIX_RE.match(f.name)
            if m:
                info = _muscle_info(m.group(1))
            else:
                info = ("muscle", "muscle")   # unknown muscle
            label_raw, label_ontology = info or ("muscle", "muscle")

            mask_path = mask_index.get(f.stem)
            samples.append((f, mask_path, label_raw, label_ontology))

        return samples

    # ── Main iterator ──────────────────────────────────────────────────────────

    def iter_entries(self) -> Iterator[USManifestEntry]:
        layout = self._detect_layout()

        if layout == "per_muscle_coloc":
            samples = self._iter_per_muscle_coloc()
        elif layout == "per_muscle_parallel":
            samples = self._iter_per_muscle_parallel()
        elif layout == "flat_images_masks":
            samples = self._iter_flat(
                images_dir=self.root / "images",
                masks_dir=self.root / "masks",
            )
        else:  # flat_named
            samples = self._iter_flat()

        n = len(samples)

        for i, (img_path, mask_path, label_raw, label_ontology) in enumerate(samples):
            has_mask = mask_path is not None and mask_path.exists()
            split    = self._infer_split(img_path.stem, i, n)
            if self.split_override:
                split = self.split_override

            instances = []
            if has_mask:
                instances.append(self._make_instance(
                    instance_id    = img_path.stem,
                    label_raw      = label_raw,
                    label_ontology = label_ontology,
                    mask_path      = str(mask_path),
                    is_promptable  = True,
                ))

            # Short muscle token for source_meta
            short = label_raw.split("_")[0].upper() if label_raw != "muscle" else "UNK"

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
                    "muscle":    short,
                    "label_raw": label_raw,
                    "layout":    layout,
                    "doi":       self.DOI,
                },
            )
