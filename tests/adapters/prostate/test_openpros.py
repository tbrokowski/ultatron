"""
tests/adapters/prostate/test_openpros.py
========================================

Unit tests for OpenProsAdapter.

Run with:
    pytest tests/adapters/prostate/test_openpros.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.adapters.prostate.openpros import OpenProsAdapter


N_SAMPLES = 3


def _build_openpros(root: Path) -> Path:
    batch_dir = root / "speed_of_sound" / "3_01"
    batch_dir.mkdir(parents=True)

    data = np.zeros((N_SAMPLES, 40, 1000, 161), dtype=np.float32)
    sos = np.zeros((N_SAMPLES, 1, 401, 161), dtype=np.float32)
    np.save(batch_dir / "3_01_P_2021-03-16_data.npy", data)
    np.save(batch_dir / "3_01_P_2021-03-16_sos.npy", sos)

    data2 = np.zeros((N_SAMPLES, 40, 1000, 161), dtype=np.float32)
    sos2 = np.zeros((N_SAMPLES, 1, 401, 161), dtype=np.float32)
    np.save(batch_dir / "3_01_P_2021-03-29_data.npy", data2)
    np.save(batch_dir / "3_01_P_2021-03-29_sos.npy", sos2)
    return root


@pytest.fixture(scope="module")
def openpros_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("openpros")
    return _build_openpros(root)


def test_openpros_adapter_yields_one_entry_per_sample(openpros_root):
    adapter = OpenProsAdapter(openpros_root)
    entries = list(adapter.iter_entries())

    assert len(entries) == 2 * N_SAMPLES

    sample_ids = {e.sample_id for e in entries}
    assert len(sample_ids) == len(entries)

    for e in entries:
        assert e.dataset_id == "ProstateSeg"
        assert e.anatomy_family == "prostate"
        assert e.modality_type == "volume"
        assert e.task_type == "regression"
        assert e.height == 401
        assert e.width == 161
        assert len(e.image_paths) == 1
        assert e.image_paths[0].endswith("_data.npy")
        assert e.source_meta["sos_path"].endswith("_sos.npy")
        assert e.source_meta["format"] == "openpros_numpy_waveform"
        assert e.source_meta["frame_idx"] == e.source_meta["sample_idx"]
        assert 0 <= e.source_meta["sample_idx"] < N_SAMPLES


def test_openpros_adapter_split_override(openpros_root):
    adapter = OpenProsAdapter(openpros_root, split_override="test")
    entries = list(adapter.iter_entries())
    assert entries
    assert all(e.split == "test" for e in entries)


def test_openpros_adapter_missing_sos_dir(tmp_path):
    adapter = OpenProsAdapter(tmp_path)
    entries = list(adapter.iter_entries())
    assert entries == []
