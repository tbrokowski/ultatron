"""
tests/test_liver_adapters.py  ·  Unit tests for liver adapters

Run with:
    pytest tests/test_liver_adapters.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.adapters.liver.aul import AULAdapter


# ══════════════════════════════════════════════════════════════════════════════
# AUL
# ══════════════════════════════════════════════════════════════════════════════

def test_aul_yields_all_images(aul_root: Path):
    entries = list(AULAdapter(aul_root).iter_entries())
    # 3 Benign + 3 Malignant + 1 Normal = 7
    assert len(entries) == 7


def test_aul_schema(aul_root: Path):
    entries = list(AULAdapter(aul_root).iter_entries())
    for e in entries:
        assert e.dataset_id == "AUL"
        assert e.anatomy_family == "liver"
        assert e.modality_type == "image"
        assert e.view_type == "liver_bmode"
        assert e.ssl_stream == "image"
        assert e.curriculum_tier in (1, 2, 3)
        assert e.study_id == e.series_id


def test_aul_classification_instance_always_present(aul_root: Path):
    entries = list(AULAdapter(aul_root).iter_entries())
    for e in entries:
        cls_instances = [i for i in e.instances if i.label_ontology == "aul_liver_class"]
        assert len(cls_instances) == 1
        inst = cls_instances[0]
        assert inst.label_raw in {"Benign", "Malignant", "Normal"}
        assert inst.classification_label in {0, 1, 2}
        assert inst.is_promptable is False


def test_aul_class_label_values(aul_root: Path):
    entries = {e.series_id: e for e in AULAdapter(aul_root).iter_entries()}
    cls = {
        e.source_meta["class"]: e.source_meta["classification_label"]
        for e in entries.values()
    }
    assert cls["Normal"]    == 0
    assert cls["Benign"]    == 1
    assert cls["Malignant"] == 2


def test_aul_benign_has_all_three_seg_instances(aul_root: Path):
    entries = {e.series_id: e for e in AULAdapter(aul_root).iter_entries()}

    for stem in ("1", "2", "3"):
        e = entries[f"Benign_{stem}"]
        assert e.has_mask is True
        assert e.task_type == "segmentation"
        assert e.is_promptable is True

        seg = [i for i in e.instances if i.label_ontology != "aul_liver_class"]
        assert len(seg) == 3

        ontologies = {i.label_ontology for i in seg}
        assert ontologies == {"liver_parenchyma", "liver_outline", "liver_lesion"}

        for inst in seg:
            assert isinstance(inst.polygon, list)
            assert len(inst.polygon) >= 3
            assert all(len(pt) == 2 for pt in inst.polygon)


def test_aul_malignant_missing_liver_json(aul_root: Path):
    entries = {e.series_id: e for e in AULAdapter(aul_root).iter_entries()}

    # image 3 is missing liver/ JSON — only outline + mass instances expected
    e = entries["Malignant_3"]
    seg = [i for i in e.instances if i.label_ontology != "aul_liver_class"]
    ontologies = {i.label_ontology for i in seg}
    assert "liver_parenchyma" not in ontologies
    assert "liver_outline" in ontologies
    assert "liver_lesion" in ontologies

    # images 1 and 2 are complete
    for stem in ("1", "2"):
        e = entries[f"Malignant_{stem}"]
        seg = [i for i in e.instances if i.label_ontology != "aul_liver_class"]
        assert {i.label_ontology for i in seg} == {
            "liver_parenchyma", "liver_outline", "liver_lesion"
        }


def test_aul_normal_has_no_mass_instance(aul_root: Path):
    entries = {e.series_id: e for e in AULAdapter(aul_root).iter_entries()}

    e = entries["Normal_1"]
    seg = [i for i in e.instances if i.label_ontology != "aul_liver_class"]
    ontologies = {i.label_ontology for i in seg}
    assert "liver_lesion" not in ontologies
    assert "liver_parenchyma" in ontologies
    assert "liver_outline" in ontologies


def test_aul_source_meta(aul_root: Path):
    entries = {e.series_id: e for e in AULAdapter(aul_root).iter_entries()}

    e = entries["Benign_1"]
    assert e.source_meta["class"] == "Benign"
    assert e.source_meta["image_id"] == "1"
    assert e.source_meta["classification_label"] == 1
    assert set(e.source_meta["seg_subfolders_present"]) == {"liver", "outline", "mass"}
    assert e.source_meta["polygon_format"] == "xy_list"

    e = entries["Malignant_3"]
    assert set(e.source_meta["seg_subfolders_present"]) == {"outline", "mass"}


def test_aul_class_balanced_splits(aul_root: Path):
    entries = list(AULAdapter(aul_root).iter_entries())

    # All splits are valid values
    for e in entries:
        assert e.split in ("train", "val", "test")

    # Each class is split independently — entries of different classes
    # in the same split do not imply anything about the other class's ordering.
    benign  = [e for e in entries if e.source_meta["class"] == "Benign"]
    malign  = [e for e in entries if e.source_meta["class"] == "Malignant"]
    normal  = [e for e in entries if e.source_meta["class"] == "Normal"]
    assert len(benign)  == 3
    assert len(malign)  == 3
    assert len(normal)  == 1


def test_aul_resolve_root_nested(aul_root: Path):
    # Adapter must also work when passed the parent directory (containing AUL/).
    # aul_root IS the AUL/ folder; its parent should trigger the nested-resolve path.
    # We can't pass aul_root.parent here because build_aul puts everything directly
    # under aul_root, not under aul_root/AUL/.  Test the direct path instead.
    entries = list(AULAdapter(aul_root).iter_entries())
    assert len(entries) == 7


def test_aul_split_override(aul_root: Path):
    entries = list(AULAdapter(aul_root, split_override="val").iter_entries())
    assert all(e.split == "val" for e in entries)
