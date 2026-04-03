"""
data/adapters/cardiac/unity.py  ·  Unity Imaging echocardiography adapter
==========================================================================

Unity: 5,724 labelled echocardiography frames with rich cardiac landmark
  keypoint annotations (LV endo/epi curves, MV hinges, valve landmarks, etc.)
  Multi-view: A4C, A2C, A3C, PLAX.
  Split:  labels-train.json (4,629) / labels-tune.json (1,095)

Directory layout:
  {root}/
    labels/
      labels-all.json    # {filename: {labels: {kp_name: {type, x, y}}}}
      labels-train.json  # subset for training
      labels-tune.json   # subset for validation/tuning
      keys.json          # keypoint schema metadata
    png-cache/
      {prefix}/{hash[0:2]}/{hash[2:4]}/{filename}
      e.g. 01/00/0c/01-000cf58f...-0002.png

Filename format:  {prefix}-{sha256_hash}-{frame_index:04d}.png
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry


def _image_path(root: Path, filename: str) -> Path:
    """
    Reconstruct the png-cache path from a label filename.

    Filename format: {prefix}-{hash}-{frame_idx}.png
    Path:            png-cache/{prefix}/{hash[0:2]}/{hash[2:4]}/{filename}
    """
    parts  = filename.split("-", 2)
    prefix = parts[0]
    h      = parts[1]
    return root / "png-cache" / prefix / h[:2] / h[2:4] / filename


def _active_keypoints(kp_dict: dict) -> dict:
    """Return only keypoints with valid coordinates (type not off/blurred)."""
    active = {}
    for name, info in kp_dict.items():
        if info.get("type") in ("off", "blurred", ""):
            continue
        x_str = info.get("x", "")
        y_str = info.get("y", "")
        if not x_str and not y_str:
            continue
        active[name] = {"type": info["type"], "x": x_str, "y": y_str}
    return active


class UnityAdapter(BaseAdapter):
    """
    Unity Imaging echocardiography adapter.

    Each entry is a single annotated frame.  Keypoint annotations (cardiac
    landmarks) are stored in source_meta["keypoints"] so downstream tasks can
    use them without requiring a pixel-level mask file.
    """

    DATASET_ID     = "Unity-Echo"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "gold"
    DOI            = "https://www.united-imaging.com/"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        labels_dir  = self.root / "labels"
        train_file  = labels_dir / "labels-train.json"
        tune_file   = labels_dir / "labels-tune.json"
        all_file    = labels_dir / "labels-all.json"

        # Determine split membership from the two subset files
        train_keys: set = set()
        tune_keys:  set = set()
        if train_file.exists():
            with open(train_file) as f:
                train_keys = set(json.load(f).keys())
        if tune_file.exists():
            with open(tune_file) as f:
                tune_keys = set(json.load(f).keys())

        src_file = all_file if all_file.exists() else train_file
        with open(src_file) as f:
            all_labels: dict = json.load(f)

        for filename, annotation in all_labels.items():
            img_path = _image_path(self.root, filename)
            if not img_path.exists():
                continue

            if self.split_override:
                split = self.split_override
            elif filename in train_keys:
                split = "train"
            elif filename in tune_keys:
                split = "val"
            else:
                split = "train"

            keypoints = _active_keypoints(annotation.get("labels", {}))

            yield self._make_entry(
                str(img_path), split,
                modality      = "image",
                instances     = [],
                study_id      = filename.split("-")[1],   # hash as pseudo study ID
                task_type     = "ssl_only",
                ssl_stream    = "image",
                is_promptable = False,
                has_mask      = False,
                source_meta   = {
                    "root":      str(self.root),
                    "doi":       self.DOI,
                    "filename":  filename,
                    "keypoints": keypoints,
                },
            )
