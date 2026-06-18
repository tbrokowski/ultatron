"""
tests/dataset_adapters/test_bus_b_adapters.py  ·  BUSBAdapter contract tests
=============================================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_bus_b_adapters.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def bus_b_root(tmp_path_factory):
    """
    Synthetic BUS-B layout:
      original/
        000001.png  (benign → has mask)
        000002.png  (malignant → has mask)
        000003.png  (normal → no mask)
      GT/
        000001.png
        000002.png
    No DatasetB.xlsx → labels are unknown unless xlsx fixture is used.
    """
    root = tmp_path_factory.mktemp("BUS_B")
    (root / "original").mkdir()
    (root / "GT").mkdir()

    for stem in ["000001", "000002"]:
        (root / "original" / f"{stem}.png").write_bytes(b"\x89PNG")
        (root / "GT"       / f"{stem}.png").write_bytes(b"\x89PNG")

    # Normal image — no mask
    (root / "original" / "000003.png").write_bytes(b"\x89PNG")

    return root


def _write_dataset_b_xlsx(path: Path, rows: list[tuple[str, str]]) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(["Image", "Type", "Diagnosis"])
    for image_id, lesion_type in rows:
        ws.append([image_id, lesion_type, ""])
    wb.save(path)


@pytest.fixture(scope="module")
def bus_b_root_with_xlsx(tmp_path_factory):
    """BUS-B with DatasetB.xlsx matching the published column layout."""
    root = tmp_path_factory.mktemp("BUS_B_xlsx")
    (root / "original").mkdir()
    (root / "GT").mkdir()

    rows = [
        ("000001", "Benign"),
        ("000002", "Malignant"),
        ("000003", "Normal"),
    ]
    for stem, label in rows:
        (root / "original" / f"{stem}.png").write_bytes(b"\x89PNG")
        if label != "Normal":
            (root / "GT" / f"{stem}.png").write_bytes(b"\x89PNG")

    _write_dataset_b_xlsx(root / "DatasetB.xlsx", rows)
    return root


class TestBUSBAdapter:

    def test_import(self):
        from data.adapters.breast.bus_b_adapter import BUSBAdapter
        assert BUSBAdapter.DATASET_ID     == "BUS-B"
        assert BUSBAdapter.ANATOMY_FAMILY == "breast"
        assert BUSBAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "BUS-B" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, bus_b_root):
        from data.adapters.breast.bus_b_adapter import BUSBAdapter
        entries = list(BUSBAdapter(root=bus_b_root).iter_entries())
        assert len(entries) == 3  # 2 with mask + 1 normal

    def test_entry_schema(self, bus_b_root):
        from data.adapters.breast.bus_b_adapter import BUSBAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in BUSBAdapter(root=bus_b_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "BUS-B"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.probe_type      == "linear"

    def test_masked_entries_have_segmentation(self, bus_b_root):
        from data.adapters.breast.bus_b_adapter import BUSBAdapter
        entries = list(BUSBAdapter(root=bus_b_root).iter_entries())
        masked  = [e for e in entries if e.has_mask]
        assert len(masked) == 2
        for e in masked:
            assert e.task_type == "segmentation"
            assert len(e.instances) == 1
            assert e.instances[0].mask_path is not None
            assert Path(e.instances[0].mask_path).exists()
            assert e.instances[0].is_promptable is True

    def test_normal_entry_no_mask(self, bus_b_root):
        from data.adapters.breast.bus_b_adapter import BUSBAdapter
        entries  = list(BUSBAdapter(root=bus_b_root).iter_entries())
        no_mask  = [e for e in entries if not e.has_mask]
        assert len(no_mask) == 1
        e = no_mask[0]
        assert e.task_type == "classification"
        assert e.instances[0].mask_path is None

    def test_without_xlsx_labels_are_unknown_not_benign(self, bus_b_root):
        from data.adapters.breast.bus_b_adapter import BUSBAdapter
        entries = {
            e.instances[0].instance_id: e
            for e in BUSBAdapter(root=bus_b_root).iter_entries()
        }
        assert entries["000001"].instances[0].label_ontology == "breast_lesion"
        assert entries["000001"].instances[0].label_raw == "unknown"
        assert entries["000002"].instances[0].label_raw == "unknown"

    def test_xlsx_labels_benign_malignant_normal(self, bus_b_root_with_xlsx):
        from data.adapters.breast.bus_b_adapter import BUSBAdapter
        entries = {
            e.instances[0].instance_id: e
            for e in BUSBAdapter(root=bus_b_root_with_xlsx).iter_entries()
        }
        assert entries["000001"].instances[0].label_ontology == "breast_lesion_benign"
        assert entries["000002"].instances[0].label_ontology == "breast_lesion_malignant"
        assert entries["000003"].instances[0].label_ontology == "breast_normal"
        assert entries["000003"].task_type == "classification"
        assert not entries["000003"].has_mask

    def test_load_xlsx_stdlib(self, bus_b_root_with_xlsx):
        from data.adapters.breast.bus_b_adapter import _load_xlsx_labels

        labels = _load_xlsx_labels(bus_b_root_with_xlsx)
        assert labels["000001"] == "benign"
        assert labels["000002"] == "malignant"
        assert labels["000003"] == "normal"

    def test_label_ontology(self, bus_b_root_with_xlsx):
        from data.adapters.breast.bus_b_adapter import BUSBAdapter
        for e in BUSBAdapter(root=bus_b_root_with_xlsx).iter_entries():
            assert e.instances[0].label_ontology in {
                "breast_lesion_benign",
                "breast_lesion_malignant",
                "breast_normal",
            }

    def test_split_override(self, bus_b_root):
        from data.adapters.breast.bus_b_adapter import BUSBAdapter
        for e in BUSBAdapter(root=bus_b_root, split_override="val").iter_entries():
            assert e.split == "val"

    def test_sample_ids_unique(self, bus_b_root):
        from data.adapters.breast.bus_b_adapter import BUSBAdapter
        ids = [e.sample_id for e in BUSBAdapter(root=bus_b_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, bus_b_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "bus_b.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("BUS-B", bus_b_root, writer)
        assert count == 3
        entries = load_manifest(out)
        assert all(e.dataset_id == "BUS-B" for e in entries)
