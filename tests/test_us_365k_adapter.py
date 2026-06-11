from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.adapters.multi_organ.us_365k import US365KAdapter

# Minimal valid JPEG (1x1)
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
def us_365k_root(tmp_path: Path) -> Path:
    root = tmp_path / "US-365K"
    images = root / "images"
    meta = root / "metadata"
    images.mkdir(parents=True)
    meta.mkdir(parents=True)

    records = [
        {
            "image": "sample_a.jpg",
            "caption": "Ultrasound of the kidney reveals a hypoechoic lesion.",
            "split": "train",
        },
        {
            "image": "sample_b.jpg",
            "caption": "Sonographic assessment of the breast demonstrates a nodule.",
            "split": "val",
        },
    ]
    for rec in records:
        (images / rec["image"]).write_bytes(_MIN_JPEG)

    with (meta / "train.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({**records[0], "image": str(images / records[0]["image"])}) + "\n")
    with (meta / "val.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({**records[1], "image": str(images / records[1]["image"])}) + "\n")

    return root


def test_us_365k_adapter_yields_text_labeled_entries(us_365k_root: Path):
    entries = list(US365KAdapter(us_365k_root).iter_entries())

    assert len(entries) == 2
    for entry in entries:
        assert entry.dataset_id == "US-365K"
        assert entry.anatomy_family == "multi"
        assert entry.task_type == "weak_label"
        assert entry.modality_type == "image"
        assert entry.instances == []
        assert entry.source_meta["report_text"] == entry.source_meta["caption"]
        assert len(entry.source_meta["report_text"]) > 10
        assert Path(entry.image_paths[0]).exists()

    splits = {e.split for e in entries}
    assert splits == {"train", "val"}
