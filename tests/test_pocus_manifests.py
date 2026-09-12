"""Tests for stratified POCUS manifests, clip fallbacks, and US-365K attributes."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.pocus.sampling import (
    clip_plan,
    equal_per_dataset,
    metadata_field_report,
    stratified_sample,
    summarise_clip_fallbacks,
)
from scripts.pocus.build_manifests import build
from tests.test_us_365k_adapter import _MIN_JPEG


def test_clip_plan_fallbacks():
    assert clip_plan(100)["stride"] == 2
    assert clip_plan(100)["fallback"] == "none"
    assert clip_plan(50)["stride"] == 1
    assert clip_plan(50)["fallback"] == "stride1"
    assert clip_plan(20)["pad_repeat"] is True
    assert clip_plan(20)["fallback"] == "pad_repeat"
    fb = summarise_clip_fallbacks([clip_plan(80), clip_plan(40), clip_plan(10), clip_plan(10)])
    assert fb["n_clips"] == 4
    assert fb["counts"]["pad_repeat"] == 2


def test_stratified_sample_proportional():
    rows = (
        [{"body_system": "cardiac", "organ": "heart", "i": i} for i in range(80)]
        + [{"body_system": "lung", "organ": "lung", "i": i} for i in range(20)]
    )
    sample = stratified_sample(rows, 20, seed=1234)
    assert len(sample) == 20
    n_c = sum(1 for r in sample if r["body_system"] == "cardiac")
    n_l = sum(1 for r in sample if r["body_system"] == "lung")
    assert n_c == 16
    assert n_l == 4


def test_equal_per_dataset_round_robin():
    rows = (
        [{"dataset_id": "CardiacUDC", "i": i} for i in range(3)]
        + [{"dataset_id": "COVID-BLUES", "i": i} for i in range(3)]
        + [{"dataset_id": "IUGC2024", "i": i} for i in range(3)]
    )
    mixed = equal_per_dataset(rows, seed=1234)
    ids = [r["dataset_id"] for r in mixed[:3]]
    assert set(ids) == {"CardiacUDC", "COVID-BLUES", "IUGC2024"}


def test_captions_only_report():
    rows = [{"image": "a.jpg", "caption": "ultrasound of the kidney"}] * 5
    rep = metadata_field_report(rows)
    assert rep["captions_only"] is True
    assert rep["has_structured_attributes"] is False

    rows2 = [{"caption": "x", "organ": "kidney", "body_system": "abdomen"}]
    rep2 = metadata_field_report(rows2)
    assert rep2["has_structured_attributes"] is True
    assert "organ" in rep2["structured_keys_present"]


def test_build_manifests_from_us365k_layout(tmp_path: Path):
    root = tmp_path / "US-365K"
    images = root / "images"
    meta = root / "metadata"
    images.mkdir(parents=True)
    meta.mkdir()
    (images / "sample_a.jpg").write_bytes(_MIN_JPEG)
    rec = {
        "image": str(images / "sample_a.jpg"),
        "caption": "Ultrasound of the kidney reveals a hypoechoic lesion.",
        "split": "train",
    }
    (meta / "train.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (meta / "val.jsonl").write_text(json.dumps({**rec, "split": "val"}) + "\n", encoding="utf-8")

    ns = type("A", (), {})()
    ns.out = tmp_path / "man"
    ns.us365k = root
    ns.cardiacudc = tmp_path / "missing-c"
    ns.covid_blues = tmp_path / "missing-b"
    ns.iugc2024 = tmp_path / "missing-i"
    ns.n_images = 1
    ns.n_prompts = 1
    ns.n_heldout = 1
    ns.n_video = 0
    ns.seed = 1234
    ns.production_manifest = None
    summary = build(ns)
    assert summary["counts"]["enc_images"] == 1
    assert (ns.out / "enc_images.jsonl").exists()
    assert summary["attribute_report"]["captions_only"] is True
    img = json.loads((ns.out / "enc_images.jsonl").read_text().splitlines()[0])
    assert img["ssl_stream"] == "image"
    assert img.get("image_paths")


def test_coerce_and_concat_manifests(tmp_path: Path):
    from data.schema.manifest import coerce_entry_dict, concat_manifests, load_manifest

    hf = coerce_entry_dict({
        "image": str(tmp_path / "a.jpg"),
        "caption": "kidney ultrasound",
        "dataset": "US-365K",
        "split": "train",
        "clip_plan": {"stride": 2},
    }, ssl_stream="image")
    assert hf["sample_id"]
    assert hf["image_paths"] == [str(tmp_path / "a.jpg")]
    assert hf["ssl_stream"] == "image"
    assert hf["source_meta"]["caption"] == "kidney ultrasound"

    img = tmp_path / "enc_images.jsonl"
    vid = tmp_path / "enc_videos.jsonl"
    img.write_text(json.dumps({"image": "/x.jpg", "caption": "c", "dataset": "US-365K"}) + "\n")
    vid.write_text(json.dumps({
        "video_path": "/y.mp4", "dataset": "COVID-BLUES", "num_frames": 80, "ssl_stream": "video",
    }) + "\n")
    dest = tmp_path / "combined.jsonl"
    stats = concat_manifests(dest, [(img, "image"), (vid, "video")])
    assert stats["n"] == 2
    assert stats["by_ssl_stream"]["image"] == 1
    assert stats["by_ssl_stream"]["video"] == 1
    entries = load_manifest(dest)
    assert len(entries) == 2
    streams = {e.ssl_stream for e in entries}
    assert streams == {"image", "video"}
