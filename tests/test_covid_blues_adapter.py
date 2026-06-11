from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.lung.covid_blues import COVIDBLUESAdapter


@pytest.fixture
def covid_blues_root(tmp_path: Path) -> Path:
    root = tmp_path / "COVID-BLUES"
    videos = root / "lus_videos"
    meta = root / "metadata"
    videos.mkdir(parents=True)
    meta.mkdir(parents=True)

    # dummy mp4 (empty file is enough for path existence)
    (videos / "patient_1_R1.mp4").write_bytes(b"\x00")
    (videos / "patient_2_L3.mp4").write_bytes(b"\x00")

    records = [
        {
            "video_file": "patient_1_R1.mp4",
            "video_path": str(videos / "patient_1_R1.mp4"),
            "patient_id": "1",
            "blue_point": "R1",
            "lung_side": "right",
            "duplicate_idx": None,
            "severity_score": 1.0,
            "a_lines": True,
            "b_lines": True,
            "comments": "thickened pleura line",
            "cov_test": 1,
            "covid_positive": True,
            "split": "train",
        },
        {
            "video_file": "patient_2_L3.mp4",
            "video_path": str(videos / "patient_2_L3.mp4"),
            "patient_id": "2",
            "blue_point": "L3",
            "lung_side": "left",
            "duplicate_idx": None,
            "severity_score": 0.0,
            "a_lines": True,
            "b_lines": False,
            "comments": "",
            "cov_test": 0,
            "covid_positive": False,
            "split": "val",
        },
    ]
    with (meta / "video_labels.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    return root


def test_covid_blues_adapter_reads_formatted_labels(covid_blues_root: Path):
    entries = list(COVIDBLUESAdapter(covid_blues_root).iter_entries())

    assert len(entries) == 2
    by_patient = {e.study_id: e for e in entries}

    e1 = by_patient["1"]
    assert e1.dataset_id == "COVID-BLUES"
    assert e1.anatomy_family == "lung"
    assert e1.modality_type == "video"
    assert e1.task_type == "classification"
    assert e1.split == "train"
    assert e1.source_meta["severity_score"] == 1.0
    assert e1.source_meta["covid_positive"] is True
    assert e1.source_meta["blue_point"] == "R1"
    assert "COVID-19 PCR positive" in e1.source_meta["report_text"]

    e2 = by_patient["2"]
    assert e2.split == "val"
    assert e2.source_meta["covid_positive"] is False
    assert e2.instances[0].label_ontology == "lung_covid_negative"
