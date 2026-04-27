"""
data/adapters/breast/busv_adapter.py  ·  BUSV breast ultrasound video adapter
==============================================================================
BUSV (Breast Ultrasound Video): 188 videos.
  Classes: benign, malignant
  Format:  rawframes — one folder per video, frames as 000000.png, 000001.png...
  Source:  github.com/xbhlk/BUSV
  SonoDQS: silver

Dataset layout on disk
-----------------------
BUSV/
├── rawframes/
│   ├── benign/
│   │   ├── {video_id}/         ← hex hash, one per clip
│   │   │   ├── 000000.png
│   │   │   ├── 000001.png
│   │   │   └── ...
│   │   └── ...
│   └── malignant/
│       ├── {video_id}/
│       └── ...
├── imagenet_vid_train_15frames.json   ← train split video ids
└── imagenet_vid_val.json              ← val split video ids

Key observations
----------------
- modality_type = "video", ssl_stream = "video"
- Label comes from the parent folder (benign / malignant).
- Each video clip is a folder of sequentially numbered PNG frames.
- image_paths contains all frame paths in order.
- Split is read from the JSON files if available, else inferred.
- No segmentation masks — classification only.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

# ── Label ontology mapping ─────────────────────────────────────────────────
CLASSES = {
    "benign":    ("benign_lesion",    "breast_lesion_benign"),
    "malignant": ("malignant_lesion", "breast_lesion_malignant"),
}


class BUSVAdapter(BaseAdapter):
    """
    Adapter for the BUSV breast ultrasound video dataset.

    Parameters
    ----------
    root : str | Path
        Root directory containing rawframes/ and the JSON split files.
    split_override : str, optional
        If set, all entries get this split label.
    """

    DATASET_ID     = "BUSV"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "silver"
    DOI            = "https://github.com/xbhlk/BUSV"

    def __init__(
        self,
        root: str | Path,
        split_override: Optional[str] = None,
    ):
        super().__init__(root=root, split_override=split_override)
        self._split_map = self._load_split_map()

    # ── Private helpers ────────────────────────────────────────────────────

    def _load_split_map(self) -> dict[str, str]:
        """
        Build a mapping video_id → "train" | "val" from the JSON split files.
        Falls back to empty dict if files are missing.
        """
        split_map: dict[str, str] = {}

        for fname, split_label in [
            ("imagenet_vid_train_15frames.json", "train"),
            ("imagenet_vid_val.json",            "val"),
        ]:
            json_path = self.root / fname
            if not json_path.exists():
                continue
            try:
                with open(json_path) as f:
                    data = json.load(f)
                # The JSON may be a list of video_ids or a dict with a "videos" key
                if isinstance(data, list):
                    video_ids = data
                elif isinstance(data, dict):
                    # Try common keys
                    video_ids = (
                        data.get("videos") or
                        data.get("video_ids") or
                        list(data.keys())
                    )
                else:
                    video_ids = []
                for vid in video_ids:
                    # vid may be a string id or a dict with an "id" field
                    if isinstance(vid, dict):
                        vid = vid.get("id") or vid.get("video_id", "")
                    split_map[str(vid)] = split_label
            except Exception:
                pass

        return split_map

    # ── Public interface ───────────────────────────────────────────────────

    def iter_entries(self) -> Iterator[USManifestEntry]:
        """Yield one USManifestEntry per video clip."""

        rawframes = self.root / "rawframes"
        if not rawframes.exists():
            raise FileNotFoundError(
                f"BUSV: rawframes/ not found under {self.root}"
            )

        all_clips = []

        for cls_name, (label_raw, label_ontology) in CLASSES.items():
            cls_dir = rawframes / cls_name
            if not cls_dir.exists():
                continue
            for clip_dir in sorted(cls_dir.iterdir()):
                if clip_dir.is_dir():
                    all_clips.append((clip_dir, cls_name, label_raw, label_ontology))

        n = len(all_clips)

        for i, (clip_dir, cls_name, label_raw, label_ontology) in enumerate(all_clips):
            video_id = clip_dir.name

            # ── Collect frames in order ────────────────────────────────────
            frames = sorted(clip_dir.glob("*.png"))
            if not frames:
                continue

            frame_paths = [str(f) for f in frames]

            # ── Split ─────────────────────────────────────────────────────
            if self.split_override:
                split = self.split_override
            elif video_id in self._split_map:
                split = self._split_map[video_id]
            else:
                split = self._infer_split(video_id, i, n)

            # ── Instance ──────────────────────────────────────────────────
            instance = self._make_instance(
                instance_id    = video_id,
                label_raw      = label_raw,
                label_ontology = label_ontology,
                mask_path      = None,
                is_promptable  = False,
            )

            # ── Entry ──────────────────────────────────────────────────────
            yield self._make_entry(
                frame_paths,          # list of frame paths
                split,
                modality      = "video",
                instances     = [instance],
                has_mask      = False,
                task_type     = "classification",
                ssl_stream    = "video",
                is_promptable = False,
                is_cine       = True,
                num_frames    = len(frames),
                probe_type    = "linear",
                source_meta   = {
                    "video_id":  video_id,
                    "cls_name":  cls_name,
                    "n_frames":  len(frames),
                    "doi":       self.DOI,
                },
            )
