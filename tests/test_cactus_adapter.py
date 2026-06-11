from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.cardiac.cactus import CACTUSAdapter

_MIN_JPEG = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08"
    b"\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e"
    b"\x1d\x1a\x1c\x1c $.\' \",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0"
    b"\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00"
    b"\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01"
    b"\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01"
    b"\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04"
    b"\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15"
    b"R\xd1\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJ"
    b"STUVWXYZcdefghijstuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94"
    b"\x95\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3"
    b"\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2"
    b"\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9"
    b"\xea\xf1\xf2\xf3\xf4\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01"
    b"\x00\x00?\x00\xfb\xd0\xff\xd9"
)


@pytest.fixture
def cactus_root(tmp_path: Path) -> Path:
    root = tmp_path / "CACTUS"
    data = root / "Cactus Dataset"
    images = data / "Images Dataset"
    grades = data / "Grades"
    videos = data / "Videos" / "Training"
    images.mkdir(parents=True)
    grades.mkdir(parents=True)
    videos.mkdir(parents=True)

    (images / "A4C").mkdir()
    (images / "Random").mkdir()
    (images / "A4C" / "sample_a4c.jpg").write_bytes(_MIN_JPEG)
    (images / "Random" / "sample_random.jpg").write_bytes(_MIN_JPEG)

    with (grades / "A4C_grades.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Subfolder Name", "Grade"])
        writer.writerow(["sample_a4c.jpg", "A4C", 7])

    with (grades / "Random_grades.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Image Name", "Subfolder Name", "Grade"])
        writer.writerow(["sample_random.jpg", "Random", 0])

    (videos / "Video1.mp4").write_bytes(b"fake-mp4")
    with (videos / "Video1.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "grade"])
        writer.writerow(["frame_0", 5])

    return root


def test_cactus_adapter_yields_graded_images_and_training_video(cactus_root: Path):
    entries = list(CACTUSAdapter(cactus_root).iter_entries())

    image_entries = [e for e in entries if e.modality_type == "image"]
    video_entries = [e for e in entries if e.modality_type == "video"]

    assert len(image_entries) == 2
    assert len(video_entries) == 1

    by_view = {e.source_meta["view"]: e for e in image_entries}
    assert by_view["A4C"].task_type == "classification"
    assert by_view["A4C"].source_meta["quality_grade"] == 7
    assert by_view["A4C"].instances[0].classification_label == 0
    assert by_view["Random"].source_meta["quality_grade"] == 0
    assert by_view["Random"].view_type == "non_cardiac"

    assert video_entries[0].task_type == "ssl_only"
    assert video_entries[0].source_meta["n_labeled_frames"] == 1


def test_cactus_adapter_resolves_nested_root(cactus_root: Path):
    adapter = CACTUSAdapter(cactus_root)
    assert adapter.data_root.name == "Cactus Dataset"
    assert adapter.images_root.is_dir()
