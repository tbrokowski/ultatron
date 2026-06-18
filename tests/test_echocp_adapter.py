"""EchoCP adapter — PFO label alignment with rest/valsalva volumes."""
from __future__ import annotations

import gzip
from pathlib import Path

import pytest


def _write_nii_gz(path: Path) -> None:
    path.write_bytes(gzip.compress(b"\x00" * 8))


def _write_labels_xlsx(path: Path, rows: list[tuple]) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.append(("Idx", "Action", "PFO level (o indicates No PFO)"))
    for row in rows:
        ws.append(row)
    wb.save(path)


@pytest.fixture
def echocp_root(tmp_path: Path) -> Path:
    """
    Minimal EchoCP layout:
      001 rest+valsalva labelled (short action codes in xlsx)
      002 rest only labelled (full action name)
      003 image with no xlsx row → ssl_only
    """
    ds = tmp_path / "EchoCP_dataset"
    ds.mkdir()
    for stem in ("001_r", "001_v", "002_r", "003_r"):
        _write_nii_gz(ds / f"{stem}_image.nii.gz")
        _write_nii_gz(ds / f"{stem}_label.nii.gz")

    _write_labels_xlsx(
        tmp_path / "echoCP_diagnosis_label.xlsx",
        [
            (1, "r", 1),
            (1, "v", 0),
            (2, "rest", 0),
        ],
    )
    return tmp_path


def test_echocp_labels_align_with_volume_filenames(echocp_root: Path):
    from data.adapters.cardiac.echocp import EchoCPAdapter

    entries = list(EchoCPAdapter(root=echocp_root).iter_entries())
    by_stem = {Path(e.image_paths[0]).name.replace("_image.nii.gz", ""): e for e in entries}

    rest = by_stem["001_r"]
    assert rest.task_type == "classification"
    assert rest.instances[0].classification_label == 1
    assert rest.source_meta["pfo_level"] == 1
    assert rest.source_meta["action"] == "rest"

    valsalva = by_stem["001_v"]
    assert valsalva.task_type == "classification"
    assert valsalva.instances[0].classification_label == 0
    assert valsalva.source_meta["pfo_level"] == 0

    full_action = by_stem["002_r"]
    assert full_action.task_type == "classification"
    assert full_action.instances[0].classification_label == 0

    unlabelled = by_stem["003_r"]
    assert unlabelled.task_type == "ssl_only"
    assert unlabelled.instances == []
    assert unlabelled.source_meta["pfo_level"] == -1


def test_echocp_load_labels_parses_o_as_no_pfo(tmp_path: Path):
    from data.adapters.cardiac.echocp import _load_labels

    _write_labels_xlsx(
        tmp_path / "echoCP_diagnosis_label.xlsx",
        [(4, "rest", "o"), (4, "v", "1")],
    )
    labels = _load_labels(tmp_path)
    assert labels[("004", "rest")] == 0
    assert labels[("004", "valsalva")] == 1
