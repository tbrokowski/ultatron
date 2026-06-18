"""
tests/dataset_adapters/test_spinal_cord_injury_us_adapters.py
=============================================================
Self-contained synthetic fixture.

Run with:
    PYTHONPATH=/Users/nouralaoui/ultatron pytest tests/dataset_adapters/test_spinal_cord_injury_us_adapters.py -v
"""
from __future__ import annotations

import textwrap
import pytest
from pathlib import Path

_VOC_XML = textwrap.dedent("""\
    <annotation>
      <filename>{stem}.png</filename>
      <object>
        <name>spinal_cord</name>
        <bndbox>
          <xmin>10</xmin><ymin>20</ymin>
          <xmax>80</xmax><ymax>90</ymax>
        </bndbox>
      </object>
    </annotation>
""")


@pytest.fixture(scope="module")
def sci_root(tmp_path_factory):
    """
    Synthetic layout:
      Final dataset for object detection/
        train/  predict4_scaled-A0004_frame1.{png,xml}
                predict4_scaled-A0004_frame2.{png,xml}
        val/    predict4_scaled-A0091_frame1.{png,xml}
        test/   predict5_scaled-A0139_frame1.{png,xml}
      SegmentationDataset/
        train_images/ + train_masks/  predict4_scaled-A0004_frame1.png (×2)
        val_images/   + val_masks/    predict4_scaled-A0091_frame1.png
        test_images/  + test_masks/   predict5_scaled-A0139_frame1.png
    """
    root = tmp_path_factory.mktemp("SpinalCordInjuryUS")

    det_root = root / "Final dataset for object detection"
    for split, stems in [
        ("train", ["predict4_scaled-A0004_frame1", "predict4_scaled-A0004_frame2"]),
        ("val",   ["predict4_scaled-A0091_frame1"]),
        ("test",  ["predict5_scaled-A0139_frame1"]),
    ]:
        d = det_root / split
        d.mkdir(parents=True)
        for stem in stems:
            (d / f"{stem}.png").write_bytes(b"\x89PNG")
            (d / f"{stem}.xml").write_text(_VOC_XML.format(stem=stem))

    seg_root = root / "SegmentationDataset"
    for split, stems in [
        ("train", ["predict4_scaled-A0004_frame1", "predict4_scaled-A0004_frame2"]),
        ("val",   ["predict4_scaled-A0091_frame1"]),
        ("test",  ["predict5_scaled-A0139_frame1"]),
    ]:
        (seg_root / f"{split}_images").mkdir(parents=True)
        (seg_root / f"{split}_masks").mkdir(parents=True)
        for stem in stems:
            (seg_root / f"{split}_images" / f"{stem}.png").write_bytes(b"\x89PNG")
            (seg_root / f"{split}_masks"  / f"{stem}.png").write_bytes(b"\x89PNG")

    return root


class TestSpinalCordInjuryUSAdapter:

    def test_import(self):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        assert SpinalCordInjuryUSAdapter.DATASET_ID     == "SpinalCordInjuryUS"
        assert SpinalCordInjuryUSAdapter.ANATOMY_FAMILY == "spine"
        assert SpinalCordInjuryUSAdapter.SONODQS        == "silver"

    def test_in_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "SpinalCordInjuryUS" in ADAPTER_REGISTRY

    def test_total_entries(self, sci_root):
        """4 detection + 4 segmentation = 8 total."""
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        entries = list(SpinalCordInjuryUSAdapter(root=sci_root).iter_entries())
        assert len(entries) == 8

    def test_detection_only(self, sci_root):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        entries = list(SpinalCordInjuryUSAdapter(
            root=sci_root, include_segmentation=False
        ).iter_entries())
        assert len(entries) == 4
        for e in entries:
            assert e.task_type == "detection"
            assert e.source_meta["sub_dataset"] == "detection"

    def test_segmentation_only(self, sci_root):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        entries = list(SpinalCordInjuryUSAdapter(
            root=sci_root, include_detection=False
        ).iter_entries())
        assert len(entries) == 4
        for e in entries:
            assert e.task_type == "segmentation"
            assert e.has_mask  is True

    def test_entry_schema(self, sci_root):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        from data.schema.manifest import USManifestEntry, ANATOMY_FAMILIES
        for e in SpinalCordInjuryUSAdapter(root=sci_root).iter_entries():
            assert isinstance(e, USManifestEntry)
            assert e.dataset_id      == "SpinalCordInjuryUS"
            assert e.anatomy_family  in ANATOMY_FAMILIES
            assert e.modality_type   == "image"
            assert e.ssl_stream      == "image"
            assert e.split           in {"train", "val", "test"}
            assert e.curriculum_tier in {1, 2, 3}
            assert e.probe_type      == "linear"

    def test_boxless_detection_is_ssl_only(self, tmp_path):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        root = tmp_path / "sci_boxless"
        det_dir = root / "Final dataset for object detection" / "train"
        det_dir.mkdir(parents=True)
        stem = "predict11_scaled-A0067_frame1"
        (det_dir / f"{stem}.png").write_bytes(b"\x89PNG")
        (det_dir / f"{stem}.xml").write_text(
            "<annotation><filename>{}</filename></annotation>".format(stem)
        )
        entries = list(SpinalCordInjuryUSAdapter(
            root=root, include_segmentation=False
        ).iter_entries())
        assert len(entries) == 1
        assert entries[0].task_type == "ssl_only"
        assert entries[0].has_box is False
        assert entries[0].instances == []

    def test_detection_bbox_parsed(self, sci_root):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        det_entries = [
            e for e in SpinalCordInjuryUSAdapter(root=sci_root).iter_entries()
            if e.task_type == "detection"
        ]
        for e in det_entries:
            assert len(e.instances) == 1
            assert e.instances[0].label_raw      == "spinal_cord"
            assert e.instances[0].label_ontology == "spinal_cord"
            assert e.instances[0].bbox_xyxy      is not None
            assert e.has_box                     is True

    def test_segmentation_masks_exist(self, sci_root):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        seg_entries = [
            e for e in SpinalCordInjuryUSAdapter(root=sci_root).iter_entries()
            if e.task_type == "segmentation"
        ]
        for e in seg_entries:
            assert len(e.instances) == 1
            assert e.instances[0].mask_path is not None
            assert Path(e.instances[0].mask_path).exists()
            assert e.instances[0].is_promptable is True

    def test_subject_and_frame_in_meta(self, sci_root):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        for e in SpinalCordInjuryUSAdapter(root=sci_root).iter_entries():
            assert "subject_id" in e.source_meta
            assert "frame_idx"  in e.source_meta
            assert e.source_meta["subject_id"] is not None

    def test_splits_respected(self, sci_root):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        entries = list(SpinalCordInjuryUSAdapter(root=sci_root).iter_entries())
        splits  = {e.split for e in entries}
        assert splits == {"train", "val", "test"}

    def test_split_override(self, sci_root):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        for e in SpinalCordInjuryUSAdapter(root=sci_root, split_override="train").iter_entries():
            assert e.split == "train"

    def test_sample_ids_unique(self, sci_root):
        from data.adapters.muscle.spinal_cord_injury_us import SpinalCordInjuryUSAdapter
        ids = [e.sample_id for e in SpinalCordInjuryUSAdapter(root=sci_root).iter_entries()]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, sci_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "sci.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("SpinalCordInjuryUS", sci_root, writer)
        assert count == 8
        entries = load_manifest(out)
        assert all(e.dataset_id == "SpinalCordInjuryUS" for e in entries)
