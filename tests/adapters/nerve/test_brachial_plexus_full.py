"""
tests/adapters/nerve/test_brachial_plexus_full.py
==================================================

Unit tests for BrachialPlexusFullAdapter.

Run with:
    pytest tests/adapters/nerve/test_brachial_plexus_full.py -v
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_MACHINES = ("Sonosite", "Butterfly", "eSaote")

# Sonosite: 2 videos (s005 with 3 frames, s071 with 2 frames)
# Butterfly: 1 video (b001 with 2 frames)
# eSaote: 1 video (e002 with 4 frames)
_MACHINE_VIDEOS = {
    "Sonosite":  [("s005", 3, "sc", "yes"), ("s071", 2, "sc", "")],
    "Butterfly": [("b001", 2, "sc", "yes")],
    "eSaote":    [("e002", 4, "sc", "")],
}


def _make_bb_line(frame_idx: int) -> str:
    bb = {"bounding_boxes": "[[10, 20, 30, 40]]", "tracker": "human"}
    return json.dumps({str(frame_idx): bb})


def _build_brachial(root: Path) -> Path:
    """Build synthetic brachial_plexus/ layout. Returns the adapter root."""
    data_dir = root / "data"
    for machine, videos in _MACHINE_VIDEOS.items():
        m_dir = data_dir / machine
        (m_dir / "videos").mkdir(parents=True)
        (m_dir / "ac_masks").mkdir(parents=True)
        (m_dir / "bb_annotations").mkdir(parents=True)

        csv_rows = []
        for vid_id, frame_cnt, nerve, needle in videos:
            # video file
            (m_dir / "videos" / f"{vid_id}.mp4").write_bytes(b"\x00")
            # ac_masks
            mask_dir = m_dir / "ac_masks" / vid_id
            mask_dir.mkdir(parents=True)
            for i in range(frame_cnt):
                (mask_dir / f"{vid_id}_{i:03d}.jpg").write_bytes(b"\x00")
            # bb_annotations
            bb_path = m_dir / "bb_annotations" / f"{vid_id}.txt"
            with bb_path.open("w") as f:
                for i in range(frame_cnt):
                    f.write(_make_bb_line(i) + "\n")
            csv_rows.append({
                "vid_name":       vid_id,
                "nerve":          nerve,
                "frame_cnt":      str(frame_cnt),
                "needle":         needle,
                "gac_iter":       "10",
            })

        csv_path = m_dir / "subjects_data.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["vid_name", "nerve", "frame_cnt", "needle", "gac_iter"])
            w.writeheader()
            w.writerows(csv_rows)

    return root


@pytest.fixture(scope="module")
def bp_root(tmp_path_factory):
    return _build_brachial(tmp_path_factory.mktemp("BrachialPlexusFull"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entries(root, **kwargs):
    from data.adapters.nerve.brachial_plexus_full import BrachialPlexusFullAdapter
    return list(BrachialPlexusFullAdapter(root, **kwargs).iter_entries())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBrachialPlexusFullMeta:

    def test_class_attributes(self):
        from data.adapters.nerve.brachial_plexus_full import BrachialPlexusFullAdapter
        assert BrachialPlexusFullAdapter.DATASET_ID     == "brachial-plexus-full"
        assert BrachialPlexusFullAdapter.ANATOMY_FAMILY == "nerve"
        assert BrachialPlexusFullAdapter.SONODQS        == "silver"

    def test_registered_in_global_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "brachial-plexus-full" in ADAPTER_REGISTRY


class TestBrachialPlexusFullEntryCount:

    def test_total_entries(self, bp_root):
        # 2 (Sonosite) + 1 (Butterfly) + 1 (eSaote) = 4
        assert len(_entries(bp_root)) == 4

    def test_entries_per_machine(self, bp_root):
        entries = _entries(bp_root)
        by_machine = {}
        for e in entries:
            m = e.source_meta["machine"]
            by_machine[m] = by_machine.get(m, 0) + 1
        assert by_machine["Sonosite"]  == 2
        assert by_machine["Butterfly"] == 1
        assert by_machine["eSaote"]    == 1


class TestBrachialPlexusFullSchema:

    def test_entry_fields(self, bp_root):
        from data.schema.manifest import ANATOMY_FAMILIES
        for e in _entries(bp_root):
            assert e.dataset_id         == "brachial-plexus-full"
            assert e.anatomy_family     in ANATOMY_FAMILIES
            assert e.modality_type      == "video"
            assert e.ssl_stream         == "video"
            assert e.split              == "train"
            assert e.task_type          == "segmentation"
            assert e.has_mask           is True
            assert e.has_box            is True
            assert e.has_temporal_order is True
            assert e.is_cine            is True
            assert e.label_raw          == ["brachial_plexus"]
            assert e.is_promptable      is False
            assert e.curriculum_tier    in {1, 2, 3}

    def test_image_paths_empty(self, bp_root):
        for e in _entries(bp_root):
            assert e.image_paths == []

    def test_one_instance_per_entry(self, bp_root):
        for e in _entries(bp_root):
            assert len(e.instances) == 1

    def test_instance_fields(self, bp_root):
        for e in _entries(bp_root):
            inst = e.instances[0]
            assert inst.label_raw      == "brachial_plexus"
            assert inst.label_ontology == "brachial_plexus"
            assert inst.is_promptable  is False


class TestBrachialPlexusFullNumFrames:

    def test_num_frames_from_csv(self, bp_root):
        entries = {e.study_id: e for e in _entries(bp_root)}
        assert entries["s005"].num_frames == 3
        assert entries["s071"].num_frames == 2
        assert entries["b001"].num_frames == 2
        assert entries["e002"].num_frames == 4

    def test_frame_indices_match_num_frames(self, bp_root):
        for e in _entries(bp_root):
            assert e.frame_indices == list(range(e.num_frames))


class TestBrachialPlexusFullStudyId:

    def test_study_id_is_video_stem(self, bp_root):
        study_ids = {e.study_id for e in _entries(bp_root)}
        assert study_ids == {"s005", "s071", "b001", "e002"}


class TestBrachialPlexusFullSourceMeta:

    def test_required_keys(self, bp_root):
        for e in _entries(bp_root):
            sm = e.source_meta
            assert "video_path"          in sm
            assert "mask_paths"          in sm
            assert "bb_annotations_path" in sm
            assert "machine"             in sm
            assert "nerve"               in sm
            assert "needle"              in sm

    def test_video_path_is_mp4(self, bp_root):
        for e in _entries(bp_root):
            assert e.source_meta["video_path"].endswith(".mp4")
            assert Path(e.source_meta["video_path"]).exists()

    def test_mask_paths_count_matches_frames(self, bp_root):
        for e in _entries(bp_root):
            assert len(e.source_meta["mask_paths"]) == e.num_frames

    def test_mask_paths_exist(self, bp_root):
        for e in _entries(bp_root):
            for mp in e.source_meta["mask_paths"]:
                assert Path(mp).exists()

    def test_bb_annotations_path_exists(self, bp_root):
        for e in _entries(bp_root):
            bb = e.source_meta["bb_annotations_path"]
            assert bb is not None
            assert Path(bb).exists()

    def test_machine_values(self, bp_root):
        machines = {e.source_meta["machine"] for e in _entries(bp_root)}
        assert machines == {"Sonosite", "Butterfly", "eSaote"}

    def test_needle_values(self, bp_root):
        entries = {e.study_id: e for e in _entries(bp_root)}
        assert entries["s005"].source_meta["needle"] == "yes"
        assert entries["s071"].source_meta["needle"] == ""
        assert entries["b001"].source_meta["needle"] == "yes"
        assert entries["e002"].source_meta["needle"] == ""


class TestBrachialPlexusFullSplit:

    def test_all_train(self, bp_root):
        for e in _entries(bp_root):
            assert e.split == "train"

    def test_split_override(self, bp_root):
        for e in _entries(bp_root, split_override="val"):
            assert e.split == "val"


class TestBrachialPlexusFullManifest:

    def test_sample_ids_unique(self, bp_root):
        ids = [e.sample_id for e in _entries(bp_root)]
        assert len(ids) == len(set(ids))

    def test_build_manifest_for_dataset(self, bp_root, tmp_path):
        from data.schema.manifest import ManifestWriter, load_manifest
        from data.adapters import build_manifest_for_dataset
        out = tmp_path / "bp.jsonl"
        with ManifestWriter(out) as writer:
            count = build_manifest_for_dataset("brachial-plexus-full", bp_root, writer)
        assert count == 4
        entries = load_manifest(out)
        assert all(e.dataset_id == "brachial-plexus-full" for e in entries)
