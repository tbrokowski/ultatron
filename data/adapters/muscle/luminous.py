"""
data/adapters/muscle/luminous.py  ·  LUMINOUS adapter
======================================================

LUMINOUS — "LUMINOUS database: lumbar multifidus muscle segmentation
from ultrasound images", Belasso et al., BMC Musculoskeletal Disorders, 2020.

  109 subjects, left + right LM at L5, prone + standing positions.
  341 images total (some subjects have multiple frames).
  Format: .tif images in B-mode/, binary .tif masks in Masks/.
  Subject IDs 1–109, matched by filename stem.
  Probe: curvilinear (GE LOGIQ e, 5 MHz).

DOI     : https://doi.org/10.1186/s12891-020-03679-3
SonoDQS : silver (single-centre, single rater, 109 subjects)

Dataset layout (from Dropbox zip)
----------------------------------
  {root}/
    B-mode/
      1.tif
      2.tif
      ...
      109.tif          (or more frames per subject)
    Masks/
      1.tif
      2.tif
      ...

Some zips include position / side metadata encoded in the filename, e.g.:
  1_prone_left.tif, 1_prone_right.tif, 1_standing_left.tif, ...
The adapter handles both flat numeric and structured naming.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_IMG_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}

# Regex to extract position and side from structured filenames like:
# "1_prone_left.tif"  "42_standing_right.tif"
_STRUCTURED_RE = re.compile(
    r"^(?P<subject_id>\d+)"
    r"(?:_(?P<position>prone|standing))?"
    r"(?:_(?P<side>left|right))?",
    re.IGNORECASE,
)


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in _IMG_EXTS


def _parse_stem(stem: str) -> tuple[str, str | None, str | None]:
    """Return (subject_id, position, side) parsed from filename stem."""
    m = _STRUCTURED_RE.match(stem)
    if m:
        return (
            m.group("subject_id"),
            (m.group("position") or "").lower() or None,
            (m.group("side") or "").lower() or None,
        )
    return stem, None, None


class LUMINOUSAdapter(BaseAdapter):
    """
    Adapter for the LUMINOUS lumbar multifidus ultrasound database.

    Yields one USManifestEntry per image with a binary segmentation mask
    of the lumbar multifidus muscle (LM CSA / EI).

    Parameters
    ----------
    root : str | Path
        Root directory containing B-mode/ and Masks/ subdirectories.
    split_override : str, optional
        Force all entries to a single split.
    """

    DATASET_ID     = "LUMINOUS"
    ANATOMY_FAMILY = "muscle"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1186/s12891-020-03679-3"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        bmode_dir = self.root / "B-mode"
        mask_dir  = self.root / "Masks"

        if not bmode_dir.is_dir():
            # Fallback: some downloads unzip without the top-level wrapper
            bmode_dir = self.root
            mask_dir  = self.root / "Masks"

        # Build mask index: stem → mask path
        mask_index: dict[str, Path] = {}
        if mask_dir.is_dir():
            for f in mask_dir.iterdir():
                if _is_image(f):
                    mask_index[f.stem] = f

        imgs = sorted(f for f in bmode_dir.iterdir() if _is_image(f))
        n    = len(imgs)

        for i, img_path in enumerate(imgs):
            split = self._infer_split(img_path.stem, i, n)

            mask_path = mask_index.get(img_path.stem)
            has_mask  = mask_path is not None

            subject_id, position, side = _parse_stem(img_path.stem)

            instances = []
            if has_mask:
                instances.append(self._make_instance(
                    instance_id    = img_path.stem,
                    label_raw      = "lumbar_multifidus",
                    label_ontology = "muscle",
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
                probe_type    = "curvilinear",
                source_meta   = {
                    "subject_id": subject_id,
                    "position":   position,   # "prone" | "standing" | None
                    "side":       side,        # "left"  | "right"   | None
                    "doi":        self.DOI,
                },
            )
