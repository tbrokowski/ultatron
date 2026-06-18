"""
tests/dataset_adapters/test_open_hip_dysplasia_adapters.py
===========================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=. pytest tests/dataset_adapters/test_open_hip_dysplasia_adapters.py -v
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def open_hip_root(tmp_path_factory):
    """
    Synthetic layout with radoss-org nesting:
      radoss-org/radoss-org-open-hip-dysplasia-8433611/
        radiopedia_ultrasound_2d/data/
          case_001.png + case_001_label.png + case_001.json
        hong_kong_poly_ultrasound_2d/data/
          standard_001.png + standard_001.json
    """
    root = tmp_path_factory.mktemp("open_hip")
    base = root / "radoss-org" / "radoss-org-open-hip-dysplasia-8433611"

    radio_dir = base / "radiopedia_ultrasound_2d" / "data"
    radio_dir.mkdir(parents=True)
    (radio_dir / "case_001.png").write_bytes(b"\x89PNG")
    (radio_dir / "case_001_label.png").write_bytes(b"\x89PNG")
    (radio_dir / "case_001.json").write_text(
        json.dumps({"R/L Graf Type": "IIa", "side": "left"})
    )

    hk_dir = base / "hong_kong_poly_ultrasound_2d" / "data"
    hk_dir.mkdir(parents=True)
    (hk_dir / "standard_001.png").write_bytes(b"\x89PNG")
    (hk_dir / "standard_001.json").write_text(json.dumps({"quality": "good"}))

    return root


class TestOpenHipDysplasiaAdapter:

    def test_import(self):
        from data.adapters.muscle.open_hip_dysplasia import OpenHipDysplasiaAdapter
        assert OpenHipDysplasiaAdapter.DATASET_ID     == "open-hip-dysplasia"
        assert OpenHipDysplasiaAdapter.ANATOMY_FAMILY == "joint"
        assert OpenHipDysplasiaAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "open-hip-dysplasia" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, open_hip_root):
        from data.adapters.muscle.open_hip_dysplasia import OpenHipDysplasiaAdapter
        entries = list(OpenHipDysplasiaAdapter(root=open_hip_root).iter_entries())
        assert len(entries) == 2

    def test_entry_schema(self, open_hip_root):
        from data.adapters.muscle.open_hip_dysplasia import OpenHipDysplasiaAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in OpenHipDysplasiaAdapter(root=open_hip_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id == "open-hip-dysplasia"
            assert e.anatomy_family in ANATOMY_FAMILIES
            assert e.modality_type == "image"
            assert e.ssl_stream == "image"

    def test_radiopedia_segmentation(self, open_hip_root):
        from data.adapters.muscle.open_hip_dysplasia import OpenHipDysplasiaAdapter
        entries = list(OpenHipDysplasiaAdapter(root=open_hip_root).iter_entries())
        radio = [e for e in entries if e.source_meta.get("subset") == "radiopedia"]
        assert len(radio) == 1
        assert radio[0].task_type == "segmentation"
        assert radio[0].has_mask is True
        assert radio[0].instances[0].mask_path is not None

    def test_hong_kong_ssl(self, open_hip_root):
        from data.adapters.muscle.open_hip_dysplasia import OpenHipDysplasiaAdapter
        entries = list(OpenHipDysplasiaAdapter(root=open_hip_root).iter_entries())
        hk = [e for e in entries if e.source_meta.get("subset") == "hong_kong"]
        assert len(hk) == 1
        assert hk[0].task_type == "ssl_only"
        assert hk[0].has_mask is False

    def test_build_manifest_for_dataset(self, open_hip_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "open_hip.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("open-hip-dysplasia", open_hip_root, writer)
        assert count == 2
        entries = load_manifest(out)
        assert all(e.dataset_id == "open-hip-dysplasia" for e in entries)
