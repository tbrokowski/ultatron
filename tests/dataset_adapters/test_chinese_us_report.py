"""
tests/dataset_adapters/test_chinese_us_report.py
=================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_chinese_us_report.py -v
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path


@pytest.fixture(scope="module")
def chinese_us_report_root(tmp_path_factory):
    """
    Synthetic Chinese US-Report layout:
      Mammary_report/ with 3 patients × 2 views
      new_Mammary2.json with train/val split
    """
    root = tmp_path_factory.mktemp("Chinese_US_Report")
    imgs_dir = root / "Mammary_report"
    imgs_dir.mkdir()

    records = [
        {"uid": 100, "finding": "双侧乳腺正常", "image_path": ["100_1.jpeg", "100_2.jpeg"], "labels": 1, "split": "train"},
        {"uid": 101, "finding": "左乳低回声结节", "image_path": ["101_1.jpeg", "101_2.jpeg"], "labels": 3, "split": "train"},
        {"uid": 102, "finding": "右乳肿块", "image_path": ["102_1.jpeg"], "labels": 5, "split": "val"},
    ]

    for rec in records:
        for fname in rec["image_path"]:
            (imgs_dir / fname).write_bytes(b"\xff\xd8")

    (root / "new_Mammary2.json").write_text(
        json.dumps({"train": [r for r in records if r["split"] == "train"],
                    "val":   [r for r in records if r["split"] == "val"]})
    )

    return root


class TestChineseUSReportBreastAdapter:

    def test_import(self):
        from data.adapters.breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter
        assert ChineseUSReportBreastAdapter.DATASET_ID     == "Chinese-US-Report-Breast"
        assert ChineseUSReportBreastAdapter.ANATOMY_FAMILY == "breast"
        assert ChineseUSReportBreastAdapter.SONODQS        == "bronze"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "Chinese-US-Report-Breast" in ADAPTER_REGISTRY

    def test_iter_entries_count(self, chinese_us_report_root):
        from data.adapters.breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter
        entries = list(ChineseUSReportBreastAdapter(root=chinese_us_report_root).iter_entries())
        assert len(entries) == 3

    def test_entry_schema(self, chinese_us_report_root):
        from data.adapters.breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in ChineseUSReportBreastAdapter(root=chinese_us_report_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id     == "Chinese-US-Report-Breast"
            assert e.anatomy_family in ANATOMY_FAMILIES
            assert e.modality_type  == "image"
            assert e.ssl_stream     == "image"
            assert e.split          in {"train", "val", "test"}
            assert e.task_type      == "weak_label"

    def test_split_from_json(self, chinese_us_report_root):
        from data.adapters.breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter
        entries = {
            e.source_meta["uid"]: e
            for e in ChineseUSReportBreastAdapter(root=chinese_us_report_root).iter_entries()
        }
        assert entries["100"].split == "train"
        assert entries["101"].split == "train"
        assert entries["102"].split == "val"

    def test_multi_view(self, chinese_us_report_root):
        from data.adapters.breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter
        entries = {
            e.source_meta["uid"]: e
            for e in ChineseUSReportBreastAdapter(root=chinese_us_report_root).iter_entries()
        }
        assert len(entries["100"].image_paths) == 2
        assert len(entries["102"].image_paths) == 1

    def test_finding_in_source_meta(self, chinese_us_report_root):
        from data.adapters.breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter
        for e in ChineseUSReportBreastAdapter(root=chinese_us_report_root).iter_entries():
            assert "finding" in e.source_meta
            assert len(e.source_meta["finding"]) > 0

    def test_no_mask(self, chinese_us_report_root):
        from data.adapters.breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter
        for e in ChineseUSReportBreastAdapter(root=chinese_us_report_root).iter_entries():
            assert not e.has_mask

    def test_split_override(self, chinese_us_report_root):
        from data.adapters.breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter
        for e in ChineseUSReportBreastAdapter(root=chinese_us_report_root, split_override="test").iter_entries():
            assert e.split == "test"

    def test_sample_ids_unique(self, chinese_us_report_root):
        from data.adapters.breast.chinese_us_report_adapter import ChineseUSReportBreastAdapter
        ids = [e.sample_id for e in ChineseUSReportBreastAdapter(root=chinese_us_report_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, chinese_us_report_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "chinese_us.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("Chinese-US-Report-Breast", chinese_us_report_root, writer)
        assert count == 3
        entries = load_manifest(out)
        assert all(e.dataset_id == "Chinese-US-Report-Breast" for e in entries)
