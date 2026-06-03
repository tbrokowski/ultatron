"""
tests/adapters/liver/test_lepset.py
====================================

Unit tests for LEPsetAdapter.

Run with:
    pytest tests/adapters/liver/test_lepset.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

try:
    from PIL import Image as PILImage
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ── Fixture ───────────────────────────────────────────────────────────────────
#
# Synthetic layout:
#   labeled/NPC/: 3 patients (p01, p02, p03), 2 frames each  → 6 entries
#   labeled/PC/:  3 patients (p04, p05, p06), 2 frames each  → 6 entries
#   unlabeled/:   3 JPGs                                      → 3 entries
#   Total: 15 entries

NPC_PATIENTS = ["p01", "p02", "p03"]
PC_PATIENTS  = ["p04", "p05", "p06"]
N_FRAMES     = 2
UNLABELED    = ["0", "1000", "2000"]


def _save_jpg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    if PIL_OK:
        PILImage.fromarray(arr).save(str(path), format="JPEG")
    else:
        path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)


def _build_lepset(root: Path) -> None:
    for patient_id in NPC_PATIENTS:
        patient_dir = root / "labeled" / "NPC" / patient_id
        for i in range(N_FRAMES):
            _save_jpg(patient_dir / f"{i}.jpg")

    for patient_id in PC_PATIENTS:
        patient_dir = root / "labeled" / "PC" / patient_id
        for i in range(N_FRAMES):
            _save_jpg(patient_dir / f"{i}.jpg")

    unlabeled_dir = root / "unlabeled"
    for stem in UNLABELED:
        _save_jpg(unlabeled_dir / f"{stem}.jpg")


@pytest.fixture(scope="module")
def lepset_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("lepset") / "LEPset"
    _build_lepset(root)
    return root


# ── Helpers ───────────────────────────────────────────────────────────────────

def _entries(root, **kwargs):
    from data.adapters.liver.lepset import LEPsetAdapter
    return list(LEPsetAdapter(root, **kwargs).iter_entries())


def _labeled(root, **kwargs):
    return [e for e in _entries(root, **kwargs) if e.task_type == "classification"]


def _unlabeled(root, **kwargs):
    return [e for e in _entries(root, **kwargs) if e.task_type == "ssl_only"]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestLEPsetMeta:

    def test_class_attributes(self):
        from data.adapters.liver.lepset import LEPsetAdapter
        assert LEPsetAdapter.DATASET_ID     == "LEPset"
        assert LEPsetAdapter.ANATOMY_FAMILY == "liver"
        assert LEPsetAdapter.SONODQS        == "gold"
        assert LEPsetAdapter.DOI            == "https://doi.org/10.5281/zenodo.8041285"

    def test_registered_in_global_registry(self):
        from data.adapters import ADAPTER_REGISTRY
        assert "LEPset" in ADAPTER_REGISTRY

    def test_missing_root_raises(self, tmp_path):
        from data.adapters.liver.lepset import LEPsetAdapter
        with pytest.raises(FileNotFoundError, match="LEPset"):
            LEPsetAdapter(tmp_path)

    def test_resolve_nested_root(self, tmp_path):
        """Adapter must work when passed the parent containing LEPset/."""
        _build_lepset(tmp_path / "LEPset")
        from data.adapters.liver.lepset import LEPsetAdapter
        entries = list(LEPsetAdapter(tmp_path).iter_entries())
        assert len(entries) == 15


class TestLEPsetEntryCount:

    def test_total_entries(self, lepset_root):
        assert len(_entries(lepset_root)) == 15

    def test_labeled_entry_count(self, lepset_root):
        assert len(_labeled(lepset_root)) == (len(NPC_PATIENTS) + len(PC_PATIENTS)) * N_FRAMES

    def test_unlabeled_entry_count(self, lepset_root):
        assert len(_unlabeled(lepset_root)) == len(UNLABELED)

    def test_npc_and_pc_frame_counts(self, lepset_root):
        entries = _labeled(lepset_root)
        npc = [e for e in entries if e.source_meta["class"] == "NPC"]
        pc  = [e for e in entries if e.source_meta["class"] == "PC"]
        assert len(npc) == len(NPC_PATIENTS) * N_FRAMES
        assert len(pc)  == len(PC_PATIENTS)  * N_FRAMES


class TestLEPsetSchema:

    def test_labeled_entry_fields(self, lepset_root):
        from data.schema.manifest import ANATOMY_FAMILIES
        for e in _labeled(lepset_root):
            assert e.dataset_id     == "LEPset"
            assert e.anatomy_family == "liver"
            assert e.anatomy_family in ANATOMY_FAMILIES
            assert e.modality_type  == "image"
            assert e.view_type      == "endoscopic_us"
            assert e.ssl_stream     == "image"
            assert e.has_mask       is False
            assert e.task_type      == "classification"
            assert e.is_promptable  is False
            assert e.study_id       == e.series_id
            assert e.curriculum_tier in (1, 2, 3)
            assert len(e.image_paths) == 1

    def test_unlabeled_entry_fields(self, lepset_root):
        for e in _unlabeled(lepset_root):
            assert e.dataset_id    == "LEPset"
            assert e.modality_type == "image"
            assert e.task_type     == "ssl_only"
            assert e.instances     == []
            assert e.has_mask      is False
            assert e.is_promptable is False

    def test_labeled_image_paths_exist(self, lepset_root):
        for e in _labeled(lepset_root):
            assert Path(e.image_paths[0]).exists()

    def test_unlabeled_image_paths_exist(self, lepset_root):
        for e in _unlabeled(lepset_root):
            assert Path(e.image_paths[0]).exists()


class TestLEPsetInstances:

    def test_exactly_one_instance_per_labeled_entry(self, lepset_root):
        for e in _labeled(lepset_root):
            assert len(e.instances) == 1

    def test_instance_ontology(self, lepset_root):
        for e in _labeled(lepset_root):
            inst = e.instances[0]
            assert inst.label_ontology       == "pancreatic_cancer_class"
            assert inst.is_promptable        is False

    def test_npc_classification_label(self, lepset_root):
        for e in _labeled(lepset_root):
            if e.source_meta["class"] == "NPC":
                assert e.instances[0].classification_label == 0
                assert e.instances[0].label_raw            == "NPC"

    def test_pc_classification_label(self, lepset_root):
        for e in _labeled(lepset_root):
            if e.source_meta["class"] == "PC":
                assert e.instances[0].classification_label == 1
                assert e.instances[0].label_raw            == "PC"

    def test_both_classes_present(self, lepset_root):
        cls_vals = {e.instances[0].classification_label for e in _labeled(lepset_root)}
        assert cls_vals == {0, 1}


class TestLEPsetStudyId:

    def test_study_id_format(self, lepset_root):
        for e in _labeled(lepset_root):
            cls   = e.source_meta["class"]
            pid   = e.source_meta["patient_id"]
            assert e.study_id == f"{cls}_{pid}"

    def test_all_frames_of_patient_share_study_id(self, lepset_root):
        from collections import defaultdict
        by_study: dict = defaultdict(list)
        for e in _labeled(lepset_root):
            by_study[e.study_id].append(e)
        for study_id, group in by_study.items():
            assert len(group) == N_FRAMES, f"{study_id} has {len(group)} frames"


class TestLEPsetSourceMeta:

    def test_labeled_source_meta_keys(self, lepset_root):
        for e in _labeled(lepset_root):
            sm = e.source_meta
            assert "class"                in sm
            assert "patient_id"           in sm
            assert "frame"                in sm
            assert "classification_label" in sm

    def test_source_meta_class_values(self, lepset_root):
        for e in _labeled(lepset_root):
            assert e.source_meta["class"] in ("NPC", "PC")

    def test_source_meta_label_matches_instance(self, lepset_root):
        for e in _labeled(lepset_root):
            assert e.source_meta["classification_label"] == e.instances[0].classification_label

    def test_unlabeled_source_meta(self, lepset_root):
        for e in _unlabeled(lepset_root):
            assert "frame" in e.source_meta


class TestLEPsetSplit:

    def test_all_splits_valid(self, lepset_root):
        for e in _entries(lepset_root):
            assert e.split in ("train", "val", "test")

    def test_patient_level_split_consistency(self, lepset_root):
        """All frames from the same patient must be in the same split."""
        from collections import defaultdict
        by_study: dict = defaultdict(set)
        for e in _labeled(lepset_root):
            by_study[e.study_id].add(e.split)
        for study_id, splits in by_study.items():
            assert len(splits) == 1, (
                f"Patient {study_id} has frames in multiple splits: {splits}"
            )

    def test_unlabeled_always_train(self, lepset_root):
        for e in _unlabeled(lepset_root):
            assert e.split == "train"

    def test_split_override_labeled(self, lepset_root):
        entries = _entries(lepset_root, split_override="val")
        assert all(e.split == "val" for e in entries)

    def test_split_override_does_not_change_task_type(self, lepset_root):
        entries = _entries(lepset_root, split_override="test")
        npc_pc  = [e for e in entries if e.task_type == "classification"]
        ssl     = [e for e in entries if e.task_type == "ssl_only"]
        assert len(npc_pc) == (len(NPC_PATIENTS) + len(PC_PATIENTS)) * N_FRAMES
        assert len(ssl)    == len(UNLABELED)
