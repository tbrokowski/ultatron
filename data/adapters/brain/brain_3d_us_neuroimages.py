"""
data/adapters/brain_3d_us_neuroimages.py  ·  3D neonatal brain ultrasound volumes
====================================================================================

3D-US-Neuroimages-Dataset (Zenodo 14917169) contains transfontanellar 3D US
volumes exported as NRRD files.  Filenames follow ``{patient_id}_{YYYY}_{MM}_{DD}.nrrd``
with multiple longitudinal scans per patient.  The public release is unlabeled
(SSL-only); there are no mask or classification files on disk.

Each NRRD is one 3D sweep (~250 axial slices).  We emit one manifest entry per
sweep, route it to both image and video SSL streams, and group train/val/test
by patient_id to avoid temporal leakage across scans from the same neonate.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

_SCAN_STEM_RE = re.compile(
    r"^(?P<patient_id>\d+)_(?P<year>\d{4})_(?P<month>\d{2})_(?P<day>\d{2})$"
)


class ThreeDUSNeuroimagesAdapter(BaseAdapter):
    DATASET_ID = "3D-US-Neuroimages-Dataset"
    ANATOMY_FAMILY = "brain"
    SONODQS = "silver"
    DOI = "https://doi.org/10.5281/zenodo.14917169"

    def iter_entries(self) -> Iterator[USManifestEntry]:
        volumes = sorted(self.root.glob("*.nrrd"))
        if not volumes:
            return

        parsed = [self._parse_scan(vol_path) for vol_path in volumes]
        split_map = self._group_split_map(meta["patient_id"] for _, meta in parsed)

        for vol_path, meta in parsed:
            patient_id = meta["patient_id"]
            num_frames = self._nrrd_depth(vol_path)
            yield self._make_entry(
                str(vol_path),
                split=split_map.get(patient_id, "train"),
                modality="volume",
                study_id=patient_id,
                series_id=meta["scan_id"],
                is_3d=True,
                is_cine=True,
                has_temporal_order=True,
                num_frames=num_frames,
                view_type="neonatal_transfontanellar_3d",
                task_type="ssl_only",
                ssl_stream="both",
                is_promptable=False,
                source_meta={
                    "patient_id": patient_id,
                    "scan_id": meta["scan_id"],
                    "scan_date": meta["scan_date"],
                    "file_name": vol_path.name,
                    "num_z_slices": num_frames,
                },
            )

    def _group_split_map(self, group_ids) -> Dict[str, str]:
        groups = sorted(set(group_ids))
        return {
            group_id: self._infer_split(group_id, idx, len(groups))
            for idx, group_id in enumerate(groups)
        }

    @staticmethod
    def _parse_scan(vol_path: Path) -> Tuple[Path, dict]:
        stem = vol_path.stem
        match = _SCAN_STEM_RE.match(stem)
        if match:
            patient_id = match.group("patient_id")
            scan_date = (
                f"{match.group('year')}-"
                f"{match.group('month')}-"
                f"{match.group('day')}"
            )
        else:
            patient_id = stem.split("_", 1)[0]
            scan_date = None

        return vol_path, {
            "patient_id": patient_id,
            "scan_id": stem,
            "scan_date": scan_date,
        }

    @staticmethod
    def _nrrd_depth(vol_path: Path) -> int:
        """Read the first axis size from an NRRD header without loading voxels."""
        try:
            with vol_path.open("rb") as handle:
                for raw_line in handle:
                    line = raw_line.decode("latin-1", errors="replace").strip()
                    if not line:
                        break
                    if line.startswith("sizes:"):
                        parts = line.split(":", 1)[1].strip().split()
                        if parts:
                            return int(parts[0])
        except OSError:
            pass
        return 1
