"""
tests/conftest.py  ·  Shared fixtures creating synthetic mini-datasets
"""
from __future__ import annotations
import csv, json, os
from pathlib import Path
import numpy as np
import pytest
try:
    from PIL import Image
    PIL_OK = True
except ImportError:
    PIL_OK = False


# ── helpers ───────────────────────────────────────────────────────────────────
def _gray(h=64, w=64):
    rng = np.random.default_rng(42)
    return (rng.random((h, w)) * 255).astype(np.uint8)

def _mask(h=64, w=64):
    m = np.zeros((h, w), dtype=np.uint8)
    m[16:48, 16:48] = 255
    return m

def _save_png(arr, path: Path):
    if PIL_OK:
        Image.fromarray(arr).save(path)
    else:
        # fallback: write raw bytes (tests will skip if PIL not available)
        path.write_bytes(b"\x89PNG\r\n" + arr.tobytes()[:64])

def _save_mhd(arr, path: Path):
    try:
        import SimpleITK as sitk
        sitk.WriteImage(sitk.GetImageFromArray(arr.astype(np.float32)), str(path))
    except ImportError:
        raw = path.with_suffix(".zraw")
        raw.write_bytes(arr.tobytes())
        path.write_text(
            f"ObjectType = Image\nNDims = 2\nDimSize = {arr.shape[1]} {arr.shape[0]}\n"
            f"ElementType = MET_UCHAR\nElementDataFile = {raw.name}\n"
        )


# ── dataset layout builders ───────────────────────────────────────────────────
def build_camus(root: Path, n=4):
    root.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        d = root / f"patient{i:04d}"; d.mkdir()
        for v in ("2CH", "4CH"):
            for p in ("ED", "ES"):
                pfx = f"patient{i:04d}_{v}_{p}"
                _save_mhd(_gray(), d / f"{pfx}.mhd")
                _save_mhd(_mask(), d / f"{pfx}_gt.mhd")

def build_busi(root: Path, n=4):
    for cls in ("normal", "benign", "malignant"):
        d = root / cls; d.mkdir(parents=True, exist_ok=True)
        for i in range(1, n + 1):
            _save_png(_gray(), d / f"image({i}).png")
            if cls != "normal":
                _save_png(_mask(), d / f"image({i})_mask.png")

def build_generic_seg(root: Path, n=8):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    stems = [f"s{i:04d}" for i in range(n)]
    for s in stems:
        _save_png(_gray(), root / "images" / f"{s}.png")
        _save_png(_mask(), root / "masks"  / f"{s}.png")
    splits = {"train": stems[:5], "val": stems[5:7], "test": stems[7:]}
    (root / "splits.json").write_text(json.dumps(splits))

def build_covidx(root: Path, n=4):
    (root / "data").mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        lines = []
        for i in range(n):
            fname = f"{split}_{i:04d}.gif"
            (root / "data" / fname).write_bytes(b"\x00" * 64)
            lines.append(f"{fname} {i % 3}")
        (root / f"{split}.txt").write_text("\n".join(lines))

def build_fetal_planes(root: Path, n=12):
    (root / "Images").mkdir(parents=True, exist_ok=True)
    planes = ["Fetal abdomen","Fetal brain","Fetal femur","Fetal thorax","Maternal cervix","Other"]
    rows = []
    for i in range(n):
        stem = f"img_{i:04d}"
        _save_png(_gray(), root / "Images" / f"{stem}.png")
        rows.append({"Image_name": stem, "Plane": planes[i % len(planes)],
                     "Patient_num": str(i // 2), "Train": "1" if i < 9 else "0"})
    with open(root / "FETAL_PLANES_DB_data.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Image_name","Plane","Patient_num","Train"], delimiter=";")
        w.writeheader(); w.writerows(rows)

def build_hc18(root: Path):
    for d, n in (("training", 6), ("test", 3)):
        (root / d).mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _save_png(_gray(), root / d / f"{i:03d}.png")
            if d == "training":
                _save_png(_mask(), root / d / f"{i:03d}_Annotation.png")

def build_lus_multicenter(root: Path, n=5):
    for cls in ("a_lines", "b_lines"):
        d = root / cls; d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            _save_png(_gray(), d / f"scan_{i:04d}.png")

def build_bus_bra(root: Path, n=8):
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        fname = f"img_{i:04d}.png"
        _save_png(_gray(), root / "images" / fname)
        _save_png(_mask(), root / "masks"  / fname)
        rows.append({"image_filename": fname, "pathology": ["benign","malignant"][i%2], "birads": str(i%5+1)})
    with open(root / "annotations.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_filename","pathology","birads"])
        w.writeheader(); w.writerows(rows)


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    return tmp_path_factory.mktemp("us_data")

@pytest.fixture(scope="session")
def camus_root(data_root):
    r = data_root / "CAMUS"; build_camus(r); return r

@pytest.fixture(scope="session")
def busi_root(data_root):
    r = data_root / "BUSI"; build_busi(r); return r

@pytest.fixture(scope="session")
def tn3k_root(data_root):
    r = data_root / "TN3K"; build_generic_seg(r); return r

@pytest.fixture(scope="session")
def covidx_root(data_root):
    r = data_root / "COVIDx-US"; build_covidx(r); return r

@pytest.fixture(scope="session")
def fetal_planes_root(data_root):
    r = data_root / "FETAL_PLANES_DB"; build_fetal_planes(r); return r

@pytest.fixture(scope="session")
def hc18_root(data_root):
    r = data_root / "HC18"; build_hc18(r); return r

@pytest.fixture(scope="session")
def lus_multicenter_root(data_root):
    r = data_root / "LUS-multicenter-2025"; build_lus_multicenter(r); return r

@pytest.fixture(scope="session")
def bus_bra_root(data_root):
    r = data_root / "BUS-BRA"; build_bus_bra(r); return r

@pytest.fixture(scope="session")
def generic_seg_root(data_root):
    r = data_root / "GenericSeg"; build_generic_seg(r); return r
