"""
data/adapters/muscle/deep_mtj.py  ·  deepMTJ adapter
======================================================

deepMTJ — "Automatic Tracking of the Muscle Tendon Junction in Healthy
and Impaired Subjects using Deep Learning", Leitner et al., 2021.

  1,344 annotated ultrasound images of the muscle-tendon junction (MTJ).
  3 US systems × 2 muscles (Lateral / Medial Gastrocnemius) × 2 movements.
  Annotations: single keypoint (x, y) per image — MTJ pixel coordinates.
  Task: keypoint detection / regression (NOT segmentation mask).

Dataset layout (test set released on GitHub)
--------------------------------------------
  {root}/
    fullres/
      deepMTJ_TS_ffullres_f0001_annotated.jpg
      deepMTJ_TS_ffullres_f0002_annotated.jpg
      ...
    256x128px/
      deepMTJ_TS_f256x128px_f0001_annotated.jpg
      ...
    MTJ_Benchmark_Labels.csv      ← keypoint coordinates (optional)

CSV format (if present):
  filename, x, y, annotator_1_x, annotator_1_y, ...
  or simply:  filename, x, y

The adapter works without the CSV (ssl_only). When the CSV is present,
keypoint coordinates are stored in source_meta and task_type = "keypoint".

Filename anatomy:
  deepMTJ_TS_f{resolution}_f{frame:04d}_annotated.jpg
  resolution = "fullres" | "256x128px"
  frame      = 1-based frame index
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_IMG_EXTS  = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_RES_DIRS  = ("fullres", "256x128px")

# deepMTJ_TS_ffullres_f0001_annotated.jpg
_FNAME_RE = re.compile(
    r"deepMTJ_TS_f(?P<resolution>[^_]+)_f(?P<frame>\d+)_annotated",
    re.IGNORECASE,
)


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in _IMG_EXTS


def _parse_filename(stem: str) -> tuple[str | None, int | None]:
    """Return (resolution, frame_idx) or (None, None) if not matched."""
    m = _FNAME_RE.search(stem)
    if m:
        return m.group("resolution"), int(m.group("frame"))
    return None, None


def _load_keypoints(root: Path) -> dict[str, tuple[float, float]]:
    """
    Load MTJ keypoint coordinates from CSV.
    Returns {filename_stem: (x, y)}.
    """
    kp: dict[str, tuple[float, float]] = {}
    for csv_name in ("MTJ_Benchmark_Labels.csv", "labels.csv", "annotations.csv"):
        csv_path = root / csv_name
        if csv_path.exists():
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Flexible column name handling
                    fname_col = next(
                        (k for k in row if "file" in k.lower() or "name" in k.lower()), None
                    )
                    x_col = next((k for k in row if k.lower() in ("x", "x_mean", "mtj_x")), None)
                    y_col = next((k for k in row if k.lower() in ("y", "y_mean", "mtj_y")), None)
                    if fname_col and x_col and y_col:
                        try:
                            stem = Path(row[fname_col]).stem
                            kp[stem] = (float(row[x_col]), float(row[y_col]))
                        except (ValueError, KeyError):
                            pass
            break
    return kp


class DeepMTJAdapter(BaseAdapter):
    """
    Adapter for the deepMTJ muscle-tendon junction tracking test dataset.

    Yields one USManifestEntry per image. When a keypoint CSV is present,
    MTJ (x, y) coordinates are stored in source_meta and task_type is
    set to "keypoint". Without the CSV the task falls back to "ssl_only".

    Both resolution variants (fullres and 256x128px) are indexed.
    Each yields independent entries — the adapter does NOT deduplicate
    across resolutions. Use source_meta["resolution"] to filter.

    Parameters
    ----------
    root : str | Path
        Root directory containing fullres/ and/or 256x128px/ subdirs,
        and optionally MTJ_Benchmark_Labels.csv.
    split_override : str, optional
        Force all entries to a single split.
    resolutions : list[str], optional
        Which resolution dirs to index. Default: all found.
    """

    DATASET_ID     = "deepMTJ"
    ANATOMY_FAMILY = "muscle"
    SONODQS        = "bronze"   # test-set only, annotation by single rater
    DOI            = "https://doi.org/10.1109/TNSRE.2021.3068765"

    def __init__(
        self,
        root,
        split_override=None,
        resolutions: list[str] | None = None,
    ):
        super().__init__(root, split_override)
        self._resolutions = resolutions  # None = all found

    def iter_entries(self) -> Iterator[USManifestEntry]:
        keypoints = _load_keypoints(self.root)

        # Collect images across all resolution subdirs
        samples: list[tuple[Path, str]] = []

        res_dirs = [
            d for d in _RES_DIRS
            if (self.root / d).is_dir()
            and (self._resolutions is None or d in self._resolutions)
        ]

        # Fallback: images directly at root
        if not res_dirs:
            for p in sorted(self.root.iterdir()):
                if _is_image(p):
                    samples.append((p, "unknown"))
        else:
            for res in res_dirs:
                for p in sorted((self.root / res).iterdir()):
                    if _is_image(p):
                        samples.append((p, res))

        n = len(samples)
        for i, (img_path, resolution) in enumerate(samples):
            split = self._infer_split(img_path.stem, i, n)

            _, frame_idx = _parse_filename(img_path.stem)
            kp           = keypoints.get(img_path.stem)
            has_kp       = kp is not None

            yield self._make_entry(
                str(img_path),
                split,
                modality      = "image",
                instances     = [],          # keypoint = no mask Instance
                has_mask      = False,
                task_type     = "keypoint" if has_kp else "ssl_only",
                ssl_stream    = "image",
                is_promptable = False,
                probe_type    = "linear",
                source_meta   = {
                    "resolution": resolution,
                    "frame_idx":  frame_idx,
                    "mtj_x":      kp[0] if kp else None,
                    "mtj_y":      kp[1] if kp else None,
                    "doi":        self.DOI,
                },
            )
