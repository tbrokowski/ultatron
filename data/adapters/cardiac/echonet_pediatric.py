"""
data/adapters/cardiac/echonet_pediatric.py  ·  EchoNet-Pediatric adapter
==========================================================================

EchoNet-Pediatric: 7,643 echocardiogram videos from Lucile Packard
  Children's Hospital (Stanford), patients aged 0-18.
  Views:   A4C (apical 4-chamber, n=3,176) and PSAX (parasternal short-axis, n=4,424)
  Labels:  ejection fraction (EF%), LV volume tracings
  Format:  .avi videos + FileList.csv + VolumeTracings.csv
  Split:   numeric column 0-9 (0-6 = train, 7 = val, 8-9 = test)

VolumeTracings.csv columns: FileName, X, Y, Frame
  Each video has expert LV tracings at end-systole and end-diastole.
  Coordinate rows share a Frame value identifying the source video frame.
  Some videos mark a missing phase with Frame = "No Systolic" or "No Diastolic".

Directory layout:
  {root}/pediatric_echo_avi/pediatric_echo_avi/{A4C,PSAX}/
      Videos/          *.avi
      FileList.csv     FileName, EF, Sex, Age, Weight, Height, Split
      VolumeTracings.csv  FileName, X, Y, Frame
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

# Numeric split values from the dataset → canonical names
_SPLIT_MAP = {str(i): "train" for i in range(7)}
_SPLIT_MAP.update({"7": "val", "8": "test", "9": "test"})

_VIEW_TO_CANONICAL = {
    "A4C":  "A4C",   # apical 4-chamber
    "PSAX": "PSAX",  # parasternal short-axis
}

_PHASE_MARKERS = frozenset({"No Systolic", "No Diastolic"})


def _parse_volume_tracings(rows: List[dict]) -> dict:
    """
    Parse VolumeTracings rows for one video into frame-aligned metadata.

    Returns ed_frame, es_frame, labeled_frame_indices, and per-frame
    contour keypoints suitable for manifest source_meta and instances.
    """
    by_frame: Dict[int, List[List[float]]] = defaultdict(list)
    markers: set[str] = set()

    for row in rows:
        frame_raw = str(row.get("Frame", "")).strip()
        if frame_raw in _PHASE_MARKERS:
            markers.add(frame_raw)
            continue
        try:
            frame = int(frame_raw)
        except (TypeError, ValueError):
            continue
        try:
            x = float(row["X"])
            y = float(row["Y"])
        except (TypeError, ValueError, KeyError):
            continue
        by_frame[frame].append([x, y])

    valid_frames = sorted(by_frame.keys())
    ed_frame: Optional[int] = None
    es_frame: Optional[int] = None

    if "No Systolic" in markers and len(valid_frames) == 1:
        ed_frame = valid_frames[0]
    elif "No Diastolic" in markers and len(valid_frames) == 1:
        es_frame = valid_frames[0]
    elif len(valid_frames) == 2:
        f0, f1 = valid_frames
        # More contour samples usually correspond to end-diastole (larger cavity).
        if len(by_frame[f0]) >= len(by_frame[f1]):
            ed_frame, es_frame = f0, f1
        else:
            ed_frame, es_frame = f1, f0

    volume_tracings = {
        str(fr): {"points": pts, "n_points": len(pts)}
        for fr, pts in by_frame.items()
    }

    return {
        "ed_frame":              ed_frame,
        "es_frame":              es_frame,
        "labeled_frame_indices": valid_frames,
        "volume_tracings":       volume_tracings,
        "has_tracings":          bool(valid_frames),
        "phase_markers":         sorted(markers),
    }


class EchoNetPediatricAdapter(BaseAdapter):
    """
    EchoNet-Pediatric adapter.  Yields video + frame-aligned image entries.

    Per labelled video:
      1 × video entry  — full cine clip with EF + tracing metadata
      0–2 × image entries — ED / ES frames with contour keypoints
                            (frame_idx in source_meta for dataloader alignment)

    Directory layout (two layers deep due to download structure):
        {root}/pediatric_echo_avi/pediatric_echo_avi/{A4C,PSAX}/
            Videos/           *.avi
            FileList.csv
            VolumeTracings.csv
    """

    DATASET_ID     = "EchoNet-Pediatric"
    ANATOMY_FAMILY = "cardiac"
    SONODQS        = "silver"
    DOI            = "https://doi.org/10.1038/s41591-023-02221-3"

    # Capstor nested layout
    _VIEW_DIRS = ("A4C", "PSAX")
    _DATA_PREFIX = Path("pediatric_echo_avi") / "pediatric_echo_avi"

    def _view_root(self, view: str) -> Path:
        return self.root / self._DATA_PREFIX / view

    def iter_entries(self) -> Iterator[USManifestEntry]:
        for view in self._VIEW_DIRS:
            vroot = self._view_root(view)
            if not vroot.exists():
                continue
            yield from self._iter_view(view, vroot)

    def _iter_view(self, view: str, vroot: Path) -> Iterator[USManifestEntry]:
        filelist_path  = vroot / "FileList.csv"
        tracings_path  = vroot / "VolumeTracings.csv"

        tracings: Dict[str, List[dict]] = {}
        if tracings_path.exists():
            with open(tracings_path) as f:
                for row in csv.DictReader(f):
                    tracings.setdefault(row["FileName"], []).append(row)

        if not filelist_path.exists():
            return

        with open(filelist_path) as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            fname = row["FileName"]
            if not fname.endswith(".avi"):
                fname += ".avi"
            vpath = vroot / "Videos" / fname
            if not vpath.exists():
                continue

            raw_split = row.get("Split", "0").strip()
            if self.split_override:
                split = self.split_override
            else:
                split = _SPLIT_MAP.get(raw_split, "train")

            ef     = float(row.get("EF", 0.0) or 0.0)
            age    = row.get("Age", "")
            sex    = row.get("Sex", "")
            weight = row.get("Weight", "")
            height = row.get("Height", "")
            study_id = fname.replace(".avi", "")

            tracing_meta = (
                _parse_volume_tracings(tracings[fname])
                if fname in tracings else {}
            )
            has_tracings = tracing_meta.get("has_tracings", False)

            base_meta = {
                "root":   str(self.root),
                "doi":    self.DOI,
                "view":   view,
                "ef":     ef,
                "age":    age,
                "sex":    sex,
                "weight": weight,
                "height": height,
            }

            instances: List[Instance] = []
            if has_tracings:
                instances.append(Instance(
                    instance_id    = f"{study_id}_{view}_lv_tracing",
                    label_raw      = "LV_contour",
                    label_ontology = "lv_segmentation",
                    anatomy_family = "cardiac",
                    is_promptable  = True,
                ))

            yield self._make_entry(
                str(vpath), split,
                modality           = "video",
                instances          = instances,
                study_id           = study_id,
                view_type          = _VIEW_TO_CANONICAL[view],
                is_cine            = True,
                has_temporal_order = True,
                fps                = 25.0,
                frame_indices      = tracing_meta.get("labeled_frame_indices") or None,
                task_type          = "regression",
                ssl_stream         = "both",
                is_promptable      = has_tracings,
                has_points         = has_tracings,
                source_meta        = {**base_meta, **tracing_meta},
            )

            # ED / ES single-frame entries for frame-aligned downstream tasks.
            for phase, frame_idx in (
                ("ED", tracing_meta.get("ed_frame")),
                ("ES", tracing_meta.get("es_frame")),
            ):
                if frame_idx is None:
                    continue
                points = tracing_meta.get("volume_tracings", {}).get(str(frame_idx), {})
                keypoints = points.get("points", [])
                img_instances = [
                    self._make_instance(
                        instance_id    = f"{study_id}_{view}_{phase}",
                        label_raw      = "LV_contour",
                        label_ontology = "lv_segmentation",
                        keypoints      = keypoints,
                        is_promptable  = True,
                    ),
                ]
                yield self._make_entry(
                    str(vpath), split,
                    modality      = "image",
                    instances     = img_instances,
                    study_id      = study_id,
                    series_id     = f"{study_id}_{view}_{phase}",
                    view_type     = _VIEW_TO_CANONICAL[view],
                    task_type     = "measurement",
                    ssl_stream    = "both",
                    is_promptable = True,
                    has_points    = bool(keypoints),
                    source_meta   = {
                        **base_meta,
                        "phase":     phase,
                        "frame_idx": frame_idx,
                        "ef":        ef,
                    },
                )
