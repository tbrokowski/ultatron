"""
data/adapters/prostate/openpros.py  ·  OpenPros USCT adapter
=============================================================

OpenPros — limited-view prostate ultrasound computed tomography benchmark.
Wang et al., 2025 (https://arxiv.org/abs/2505.12261).

Layout on disk::

    {root}/
    ├── speed_of_sound/
    │   └── 3_0{i}/                     patient batch (i = 1..4)
    │       ├── 3_0{i}_P_{date}_data.npy   waveform  (N × 40 × 1000 × 161)
    │       └── 3_0{i}_P_{date}_sos.npy    SOS map    (N × 1  × 401  × 161)
    └── mask_mat_3_0{i}.zip             optional segmentation masks (.mat)

Each ``*_data.npy`` / ``*_sos.npy`` pair stores *N* stacked limited-view
configurations for one prostate phantom (typically N = 1140).  The adapter
emits one manifest entry per stacked sample so downstream loaders can index
into the shared NumPy files via ``source_meta["sample_idx"]``.

Split strategy: deterministic 80/10/10 by sorted ``(batch, sample_date)``,
matching the published 224K / 28K / 28K partition.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from data.adapters.base import BaseAdapter
from data.schema.manifest import USManifestEntry

log = logging.getLogger(__name__)

_DATA_SUFFIX = "_data.npy"
_SOS_SUFFIX = "_sos.npy"
_FILE_RE = re.compile(
    r"^(?P<batch>\d+_\d+)_P_(?P<sample_date>.+)_(?P<kind>data|sos)\.npy$",
    re.IGNORECASE,
)

# Published tensor shapes (sample axis excluded).
_DATA_TRAILING_SHAPE = (40, 1000, 161)
_SOS_TRAILING_SHAPE = (1, 401, 161)
_SOS_HEIGHT, _SOS_WIDTH = 401, 161


class OpenProsAdapter(BaseAdapter):
    DATASET_ID = "ProstateSeg"
    ANATOMY_FAMILY = "prostate"
    SONODQS = "gold"
    DOI = "https://arxiv.org/abs/2505.12261"
    DEFAULT_SPLIT_RATIO = (0.80, 0.10, 0.10)

    def iter_entries(self) -> Iterator[USManifestEntry]:
        sos_dir = self.root / "speed_of_sound"
        if not sos_dir.is_dir():
            log.warning(
                "OpenProsAdapter: speed_of_sound dir not found at %s", self.root
            )
            return

        pairs = self._collect_pairs(sos_dir)
        if not pairs:
            log.warning(
                "OpenProsAdapter: no *_data.npy/*_sos.npy pairs under %s", sos_dir
            )
            return

        study_keys = sorted({meta["study_key"] for _, _, meta in pairs})
        n_studies = len(study_keys)

        for data_path, sos_path, meta in pairs:
            try:
                n_samples = self._sample_count(data_path)
            except Exception as exc:
                log.warning(
                    "OpenProsAdapter: skipping %s (cannot read shape): %s",
                    data_path.name,
                    exc,
                )
                continue

            split = self.split_override or self._infer_split(
                meta["study_key"],
                study_keys.index(meta["study_key"]),
                n_studies,
            )

            for sample_idx in range(n_samples):
                yield self._make_sample_entry(
                    data_path=data_path,
                    sos_path=sos_path,
                    meta=meta,
                    sample_idx=sample_idx,
                    split=split,
                )

    @staticmethod
    def _collect_pairs(
        sos_dir: Path,
    ) -> List[Tuple[Path, Path, dict]]:
        """Return sorted (data_path, sos_path, meta) triples."""
        data_files = sorted(sos_dir.rglob(f"*{_DATA_SUFFIX}"))
        pairs: List[Tuple[Path, Path, dict]] = []

        for data_path in data_files:
            sos_path = data_path.with_name(
                data_path.name[: -len(_DATA_SUFFIX)] + _SOS_SUFFIX
            )
            if not sos_path.is_file():
                log.warning(
                    "OpenProsAdapter: missing SOS pair for %s", data_path.name
                )
                continue

            meta = _parse_file_stem(data_path.stem)
            if meta is None:
                log.warning(
                    "OpenProsAdapter: unexpected filename %s", data_path.name
                )
                continue

            pairs.append((data_path, sos_path, meta))

        return pairs

    @staticmethod
    def _sample_count(data_path: Path) -> int:
        arr = np.load(data_path, mmap_mode="r")
        try:
            if arr.ndim < 1:
                raise ValueError(f"expected >=1 dims, got shape {arr.shape}")
            return int(arr.shape[0])
        finally:
            del arr

    def _make_sample_entry(
        self,
        *,
        data_path: Path,
        sos_path: Path,
        meta: dict,
        sample_idx: int,
        split: str,
    ) -> USManifestEntry:
        study_id = meta["study_id"]
        series_id = f"{study_id}_{sample_idx:04d}"
        sample_key = f"{data_path}::{sample_idx}"

        entry = self._make_entry(
            [str(data_path), str(sos_path)],
            split=split,
            modality="volume",
            study_id=study_id,
            series_id=series_id,
            height=_SOS_HEIGHT,
            width=_SOS_WIDTH,
            num_frames=1,
            is_3d=True,
            task_type="regression",
            ssl_stream="image",
            is_promptable=False,
            has_mask=False,
            source_meta={
                "batch": meta["batch"],
                "sample_date": meta["sample_date"],
                "sample_idx": sample_idx,
                "data_shape": list(_DATA_TRAILING_SHAPE),
                "sos_shape": list(_SOS_TRAILING_SHAPE),
                "format": "openpros_numpy_waveform",
            },
        )
        entry.sample_id = USManifestEntry.make_sample_id(self.DATASET_ID, sample_key)
        return entry


def _parse_file_stem(stem: str) -> Optional[dict]:
    """
    Parse ``3_01_P_2021-03-16_data`` → batch/study metadata.

    The ``P_{date}`` token is a prostate-phantom identifier, not a calendar date.
    """
    m = _FILE_RE.match(f"{stem}.npy")
    if not m or m.group("kind").lower() != "data":
        return None

    batch = m.group("batch")
    sample_date = m.group("sample_date")
    study_id = f"{batch}_P_{sample_date}"
    return {
        "batch": batch,
        "sample_date": sample_date,
        "study_id": study_id,
        "study_key": study_id,
    }
