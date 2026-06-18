"""
data/adapters/breast/midi_b.py  ·  MIDI-B breast ultrasound DICOM adapter
==========================================================================

MIDI-B (Medical Image De-Identification Benchmark, MICCAI 2024): multi-modality
DICOM collection hosted on TCIA. The breast subset on capstor contains US series
staged under dicoms/ with optional per-series zip archives.

Layout:
  {root}/
    series_uids.txt              one SeriesInstanceUID per line (optional)
    dicoms/
      {series_uid}.zip           downloaded archive (optional)
      {series_uid}/              extracted single-frame DICOM slices
        00000001.dcm
        ...
"""
from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)


class MidiBAdapter(BaseAdapter):
    DATASET_ID     = "midi-b"
    ANATOMY_FAMILY = "breast"
    SONODQS        = "unrated"
    DOI            = "https://www.cancerimagingarchive.net/collection/midi-b-test-midi-b-validation/"

    def __init__(self, root, split_override=None):
        super().__init__(root, split_override=split_override)
        self._dicoms_dir = self.root / "dicoms"

    @staticmethod
    def _ensure_extracted(dicoms_dir: Path) -> None:
        """Extract any series zip whose output directory is missing or empty."""
        for zip_path in sorted(dicoms_dir.glob("*.zip")):
            out_dir = dicoms_dir / zip_path.stem
            if out_dir.is_dir() and any(out_dir.glob("*.dcm")):
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(out_dir)
            log.info("MidiBAdapter: extracted %s", zip_path.name)

    def _load_series_uids(self) -> List[str]:
        uids_file = self.root / "series_uids.txt"
        if uids_file.exists():
            return [
                line.strip()
                for line in uids_file.read_text().splitlines()
                if line.strip()
            ]
        return sorted(
            p.name
            for p in self._dicoms_dir.iterdir()
            if p.is_dir() and any(p.glob("*.dcm"))
        )

    def _iter_series(self) -> Iterator[Tuple[str, List[Path]]]:
        for series_uid in self._load_series_uids():
            series_dir = self._dicoms_dir / series_uid
            if not series_dir.is_dir():
                continue
            dcms = sorted(series_dir.glob("*.dcm"))
            if dcms:
                yield series_uid, dcms

    def _series_split_map(self, series_uids: List[str]) -> Dict[str, str]:
        return {
            series_uid: self._infer_split(series_uid, idx, len(series_uids))
            for idx, series_uid in enumerate(series_uids)
        }

    def iter_entries(self) -> Iterator[USManifestEntry]:
        if not self._dicoms_dir.is_dir():
            log.warning("MidiBAdapter: dicoms/ not found under %s", self.root)
            return

        self._ensure_extracted(self._dicoms_dir)

        series = list(self._iter_series())
        if not series:
            if any(self._dicoms_dir.glob("*.zip")):
                log.warning(
                    "MidiBAdapter: DICOM zips present but no extracted "
                    "series found under %s",
                    self._dicoms_dir,
                )
            else:
                log.warning(
                    "MidiBAdapter: no DICOM series found under %s",
                    self._dicoms_dir,
                )
            return

        series_uids = [uid for uid, _ in series]
        split_map = self._series_split_map(series_uids)

        for series_uid, dcms in series:
            split = self.split_override or split_map[series_uid]
            n_frames = len(dcms)

            for frame_idx, dcm_path in enumerate(dcms):
                yield self._make_entry(
                    str(dcm_path),
                    split=split,
                    modality="image",
                    series_id=series_uid,
                    study_id=series_uid,
                    num_frames=n_frames,
                    has_temporal_order=n_frames > 1,
                    task_type="ssl_only",
                    ssl_stream="image",
                    is_promptable=False,
                    source_meta={
                        "series_uid": series_uid,
                        "frame_idx": frame_idx,
                        "n_frames": n_frames,
                        "modality_dicom": "US",
                    },
                )
