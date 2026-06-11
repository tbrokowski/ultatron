"""Unit tests for DDTI and TN5000 thyroid adapters."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.adapters.thyroid.ddti import DDTIAdapter
from data.adapters.thyroid.tn5000 import TN5000Adapter


def _validate_entries(entries, *, min_count: int = 1):
    assert len(entries) >= min_count
    for e in entries:
        assert e.sample_id
        assert e.dataset_id
        assert e.anatomy_family == "thyroid"
        assert e.split in ("train", "val", "test", "unlabeled")
        assert len(e.image_paths) >= 1


def build_ddti(root: Path) -> None:
    archive = root / "archive"
    archive.mkdir(parents=True, exist_ok=True)

    svg = json.dumps([
        {
            "points": [
                {"x": 10, "y": 10},
                {"x": 40, "y": 10},
                {"x": 40, "y": 40},
                {"x": 10, "y": 40},
            ],
            "regionType": "freehand",
        }
    ])
    (archive / "1.xml").write_text(
        "<case>"
        "<number>1</number><age>40</age><sex>F</sex>"
        "<composition>solid</composition>"
        "<echogenicity>hypoechogenicity</echogenicity>"
        "<margins>well defined</margins>"
        "<calcifications>none</calcifications>"
        "<tirads>3</tirads>"
        f"<mark><image>1</image><svg>{svg}</svg></mark>"
        "</case>"
    )
    (archive / "2.xml").write_text(
        "<case>"
        "<number>2</number><tirads>5</tirads>"
        f"<mark><image>1</image><svg>{svg}</svg></mark>"
        "</case>"
    )

    try:
        from PIL import Image
        Image.new("RGB", (64, 48)).save(archive / "1_1.jpg")
        Image.new("RGB", (64, 48)).save(archive / "2_1.jpg")
    except Exception:
        (archive / "1_1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
        (archive / "2_1.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)


def build_tn5000(root: Path) -> None:
    data = root / "Main data"
    img_dir = data / "JPEGImages"
    ann_dir = data / "Annotations"
    split_dir = data / "ImageSets" / "Main"
    for d in (img_dir, ann_dir, split_dir):
        d.mkdir(parents=True, exist_ok=True)

    for stem, label, split in (
        ("000001", "0", "train"),
        ("000002", "1", "val"),
        ("000003", "1", "test"),
    ):
        (split_dir / f"{split}.txt").write_text(
            ((split_dir / f"{split}.txt").read_text() if (split_dir / f"{split}.txt").exists() else "")
            + f"{stem}\n"
        )
        (ann_dir / f"{stem}.xml").write_text(
            f"<annotation><size><width>64</width><height>48</height></size>"
            f"<object><name>{label}</name><bndbox>"
            f"<xmin>10</xmin><ymin>10</ymin><xmax>40</xmax><ymax>30</ymax>"
            f"</bndbox></object></annotation>"
        )
        try:
            from PIL import Image
            Image.new("RGB", (64, 48)).save(img_dir / f"{stem}.jpg")
        except Exception:
            (img_dir / f"{stem}.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)


@pytest.fixture(scope="session")
def ddti_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("ddti")
    build_ddti(root)
    return root


@pytest.fixture(scope="session")
def tn5000_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("tn5000")
    build_tn5000(root)
    return root


def test_ddti_adapter(ddti_root: Path):
    entries = list(DDTIAdapter(ddti_root).iter_entries())
    _validate_entries(entries, min_count=2)

    by_case = {e.study_id: e for e in entries}
    assert set(by_case) == {"1", "2"}

    e1 = by_case["1"]
    assert e1.has_mask is True
    assert e1.task_type == "segmentation"
    assert e1.source_meta["tirads_raw"] == "3"

    seg = [i for i in e1.instances if i.label_ontology == "thyroid_nodule_boundary"]
    cls = [i for i in e1.instances if i.label_ontology == "thyroid_tirads"]
    assert len(seg) == 1 and seg[0].mask_path
    assert len(cls) == 1 and cls[0].classification_label == 2


def test_tn5000_adapter(tn5000_root: Path):
    entries = list(TN5000Adapter(tn5000_root).iter_entries())
    _validate_entries(entries, min_count=3)

    by_stem = {e.study_id: e for e in entries}
    assert set(by_stem) == {"000001", "000002", "000003"}

    e_train = by_stem["000001"]
    assert e_train.split == "train"
    assert e_train.has_mask is True
    assert e_train.has_box is True
    assert e_train.source_meta["nodule_class"] == "benign"

    seg = [i for i in e_train.instances if i.label_ontology == "thyroid_nodule_boundary"]
    cls = [i for i in e_train.instances if i.label_ontology == "thyroid_nodule_class"]
    assert len(seg) == 1 and seg[0].bbox_xyxy is not None
    assert len(cls) == 1 and cls[0].classification_label == 0
