"""
data/adapters/cardiac/echonet_pediatric.py  ·  EchoNet-Pediatric adapter
==========================================================================

EchoNet-Pediatric: 7,643 echocardiogram videos from Lucile Packard
  Children's Hospital (Stanford), patients aged 0-18.
  Views:   A4C (apical 4-chamber, n=3,176) and PSAX (parasternal short-axis, n=4,424)
  Labels:  ejection fraction (EF%), LV volume tracings
  Format:  .avi videos + FileList.csv + VolumeTracings.csv
  Split:   numeric column 0-9 (0-6 = train, 7 = val, 8-9 = test)

Directory layout:
  {root}/pediatric_echo_avi/pediatric_echo_avi/{A4C,PSAX}/
      Videos/          *.avi
      FileList.csv     FileName, EF, Sex, Age, Weight, Height, Split
      VolumeTracings.csv  FileName, X, Y, Frame
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry, Instance

# Numeric split values from the dataset → canonical names
_SPLIT_MAP = {str(i): "train" for i in range(7)}
_SPLIT_MAP.update({"7": "val", "8": "test", "9": "test"})

_VIEW_TO_CANONICAL = {
    "A4C":  "A4C",   # apical 4-chamber
    "PSAX": "PSAX",  # parasternal short-axis
}


class EchoNetPediatricAdapter(BaseAdapter):
    """
    EchoNet-Pediatric adapter.  Yields one video entry per AVI file across
    both A4C and PSAX view directories.

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

        # Load LV tracings keyed by filename
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

            # Numeric split column (string "0"-"9")
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

            instances = []
            if fname in tracings:
                instances.append(Instance(
                    instance_id    = f"{fname}_{view}",
                    label_raw      = "LV_contour",
                    label_ontology = "lv_segmentation",
                    anatomy_family = "cardiac",
                    is_promptable  = True,
                ))

            yield self._make_entry(
                str(vpath), split,
                modality           = "video",
                instances          = instances,
                study_id           = fname.replace(".avi", ""),
                view_type          = _VIEW_TO_CANONICAL[view],
                is_cine            = True,
                has_temporal_order = True,
                fps                = 25.0,
                task_type          = "regression",
                ssl_stream         = "both",
                is_promptable      = bool(instances),
                has_mask           = bool(instances),
                source_meta        = {
                    "root":   str(self.root),
                    "doi":    self.DOI,
                    "view":   view,
                    "ef":     ef,
                    "age":    age,
                    "sex":    sex,
                    "weight": weight,
                    "height": height,
                },
            )
