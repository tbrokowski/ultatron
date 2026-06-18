"""
tests/adapters/prostate/test_mri_us_biopsy.py
=============================================

Unit tests for ProstateMRIUSBiopsyAdapter.

Run with:
    pytest tests/adapters/prostate/test_mri_us_biopsy.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from data.adapters.prostate.mri_us_biopsy import ProstateMRIUSBiopsyAdapter


def _build_biopsy_tree(root: Path) -> Path:
    series_a = root / "ProstateX-0001" / "2014-01-01" / "series-us-001"
    series_b = root / "ProstateX-0002" / "2014-02-01" / "series-us-002"
    series_a.mkdir(parents=True)
    series_b.mkdir(parents=True)
    (series_a / "0001.dcm").write_bytes(b"dicom-a")
    (series_a / "0002.dcm").write_bytes(b"dicom-a2")
    (series_b / "0001.dcm").write_bytes(b"dicom-b")
    return root


@pytest.fixture(scope="module")
def biopsy_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("prostate_mri_us_biopsy")
    return _build_biopsy_tree(root)


def test_mri_us_biopsy_yields_one_entry_per_dicom(biopsy_root):
    adapter = ProstateMRIUSBiopsyAdapter(biopsy_root)
    entries = list(adapter.iter_entries())

    assert len(entries) == 3
    assert {e.dataset_id for e in entries} == {"Prostate-MRI-US-Biopsy"}
    assert all(e.anatomy_family == "prostate" for e in entries)
    assert all(e.modality_type == "image" for e in entries)
    assert all(e.task_type == "ssl_only" for e in entries)
    assert {e.study_id for e in entries} == {"ProstateX-0001", "ProstateX-0002"}


def test_mri_us_biopsy_empty_root(tmp_path):
    adapter = ProstateMRIUSBiopsyAdapter(tmp_path)
    assert list(adapter.iter_entries()) == []
