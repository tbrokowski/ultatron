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
    """Build TN3K-compatible layout: image/ and label/ subdirectories."""
    (root / "image").mkdir(parents=True, exist_ok=True)
    (root / "label").mkdir(parents=True, exist_ok=True)
    stems = [f"s{i:04d}" for i in range(n)]
    for s in stems:
        _save_png(_gray(), root / "image" / f"{s}.jpg")
        _save_png(_mask(), root / "label" / f"{s}.png")
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


def build_echonet_dynamic(root: Path, n=4):
    """Synthetic EchoNet-Dynamic with AVI files and FileList.csv."""
    vid_dir = root / "Videos"
    vid_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        fname = f"echo_{i:04d}.avi"
        (vid_dir / fname).write_bytes(b"\x00" * 64)
        split_name = ["TRAIN", "VAL", "TEST"][i % 3]
        rows.append({"FileName": fname, "EF": f"{55.0 + i:.2f}",
                     "ESV": f"{40.0 + i:.2f}", "EDV": f"{100.0 + i:.2f}",
                     "Split": split_name, "FrameHeight": "112",
                     "FrameWidth": "112", "FPS": "30", "NumberOfFrames": "50"})
    with open(root / "FileList.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # Empty VolumeTracings
    with open(root / "VolumeTracings.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=["FileName","X","Y","Frame","Phase"]).writeheader()


def build_echonet_pediatric(root: Path, n=4):
    """Synthetic EchoNet-Pediatric with A4C and PSAX view subdirectories."""
    import hashlib
    base = root / "pediatric_echo_avi" / "pediatric_echo_avi"
    for view_idx, view in enumerate(("A4C", "PSAX")):
        vdir = base / view
        (vdir / "Videos").mkdir(parents=True, exist_ok=True)
        rows, tracing_rows = [], []
        for i in range(n):
            fname = f"syn_{view}_{i:04d}.avi"
            (vdir / "Videos" / fname).write_bytes(b"\x00" * 64)
            split_val = str(i % 9)   # numeric split 0-8
            rows.append({
                "FileName": fname, "EF": f"{55.0 + i:.2f}",
                "Sex": "M" if i % 2 else "F",
                "Age": str(i + 1), "Weight": "15.0", "Height": "90.0",
                "Split": split_val,
            })
            tracing_rows.append({"FileName": fname, "X": "58", "Y": "58", "Frame": str(i + 1)})
        with open(vdir / "FileList.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["FileName","EF","Sex","Age","Weight","Height","Split"])
            w.writeheader(); w.writerows(rows)
        with open(vdir / "VolumeTracings.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["FileName","X","Y","Frame"])
            w.writeheader(); w.writerows(tracing_rows)


def build_ted(root: Path, n=4):
    """Synthetic TED dataset with .mhd cine sequences."""
    db = root / "database"
    for i in range(1, n + 1):
        pid  = f"patient{i:03d}"
        pdir = db / pid
        pdir.mkdir(parents=True, exist_ok=True)
        _save_mhd(_gray(), pdir / f"{pid}_4CH_sequence.mhd")
        _save_mhd(_mask(), pdir / f"{pid}_4CH_sequence_gt.mhd")
        cfg_text = (
            f"ED: 1\nES: {5 + i}\nNbFrame: {15 + i}\n"
            f"Sex: {'M' if i % 2 else 'F'}\nAge: {40 + i}\n"
            f"ImageQuality: Medium\nEF: {55 - i}\n"
        )
        (pdir / f"{pid}_4CH_info.cfg").write_text(cfg_text)


def build_unity(root: Path, n=6):
    """Synthetic Unity dataset with hash-bucketed PNGs and keypoint labels."""
    import hashlib
    labels_all = {}
    train_set  = {}
    tune_set   = {}
    for i in range(n):
        h      = hashlib.sha256(f"frame{i}".encode()).hexdigest()
        prefix = "01"
        fname  = f"{prefix}-{h}-{i:04d}.png"
        bucket1, bucket2 = h[:2], h[2:4]
        img_dir = root / "png-cache" / prefix / bucket1 / bucket2
        img_dir.mkdir(parents=True, exist_ok=True)
        _save_png(_gray(), img_dir / fname)
        entry = {"labels": {
            "mv-ant-hinge": {"type": "point", "x": "100.0", "y": "200.0"},
            "mv-post-hinge": {"type": "point", "x": "150.0", "y": "210.0"},
            "lv-apex-endo":  {"type": "off",   "x": "",      "y": ""},
        }}
        labels_all[fname] = entry
        if i < n - 2:
            train_set[fname] = entry
        else:
            tune_set[fname] = entry
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("labels-all.json",   labels_all),
        ("labels-train.json", train_set),
        ("labels-tune.json",  tune_set),
    ):
        (labels_dir / name).write_text(__import__("json").dumps(data))


def build_mimic_lvvol_a4c(root: Path, n=4):
    """Synthetic MIMIC-IV-Echo-LVVol-A4C with stub DICOMs and FileList.csv."""
    base = root / "physionet.org" / "files" / "mimic-iv-echo-ext-lvvol-a4c" / "1.0.0"
    dcm_dir = base / "dicom"
    dcm_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        study_id = f"9000{i:04d}"
        (dcm_dir / f"{study_id}.dcm").write_bytes(b"\x00" * 128)
        rows.append({
            "patient_id":          f"1{i:07d}",
            "study_date":          "21470526",
            "study_time":          "142855",
            "parent_dicom_path":   f"files/p1/p1{i:07d}/s{study_id}/{study_id}_0001.dcm",
            "study_id":            study_id,
            "modality":            "US",
            "manufacturer":        "GE",
            "manufacturer_model":  "Vivid E95",
            "rows":                "708",
            "columns":             "1016",
            "number_of_frames":    "60",
            "frame_rate":          "30",
            "duration_seconds":    "2.0",
            "file_size_mb":        "5.0",
            "photometric_interpretation": "YBR_FULL_422",
            "LVEDV_A4C":  f"{100.0 + i:.2f}",
            "LVESV_A4C":  f"{35.0 + i:.2f}",
            "LVEF_A4C":   f"{65.0 - i:.2f}",
            "LVEDV_BP":   f"{98.0 + i:.2f}",
            "LVESV_BP":   f"{34.0 + i:.2f}",
            "LVEF_BP":    f"{65.3 - i:.2f}",
        })
    fields = list(rows[0].keys())
    with open(base / "FileList.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


# ── fixtures ──────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def data_root(tmp_path_factory):
    return tmp_path_factory.mktemp("us_data")

@pytest.fixture(scope="session")
def camus_root(data_root):
    r = data_root / "CAMUS"; build_camus(r); return r

@pytest.fixture(scope="session")
def echonet_root(data_root):
    r = data_root / "EchoNet-Dynamic"; build_echonet_dynamic(r); return r

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

@pytest.fixture(scope="session")
def echonet_pediatric_root(data_root):
    r = data_root / "EchoNet-Pediatric"; build_echonet_pediatric(r); return r

@pytest.fixture(scope="session")
def ted_root(data_root):
    r = data_root / "TED"; build_ted(r); return r

@pytest.fixture(scope="session")
def unity_root(data_root):
    r = data_root / "Unity"; build_unity(r); return r

@pytest.fixture(scope="session")
def mimic_lvvol_a4c_root(data_root):
    r = data_root / "MIMIC-IV-Echo-LVVol-A4C"; build_mimic_lvvol_a4c(r); return r


def build_aul(root: Path):
    """
    Synthetic AUL dataset.

    Benign:    3 images, all 3 segmentation types (liver, outline, mass)
    Malignant: 3 images, liver JSON missing for image 3
    Normal:    1 image,  liver + outline only (no mass/ folder)
    Total: 7 images
    """
    polygon = [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]]
    polygon_json = json.dumps(polygon)

    classes = {
        "Benign":    {"stems": ["1", "2", "3"], "seg_types": ["liver", "outline", "mass"]},
        "Malignant": {"stems": ["1", "2", "3"], "seg_types": ["liver", "outline", "mass"],
                      "skip_liver": {"3"}},
        "Normal":    {"stems": ["1"],            "seg_types": ["liver", "outline"]},
    }

    for cls_name, cfg in classes.items():
        img_dir = root / cls_name / "image"
        img_dir.mkdir(parents=True, exist_ok=True)
        for seg_type in cfg["seg_types"]:
            (root / cls_name / "segmentation" / seg_type).mkdir(parents=True, exist_ok=True)

        for stem in cfg["stems"]:
            # JPEG image
            img_path = img_dir / f"{stem}.jpg"
            if PIL_OK:
                from PIL import Image as PILImage
                PILImage.fromarray(_gray(8, 8)).save(img_path, format="JPEG")
            else:
                img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)

            # Segmentation JSONs
            skip_liver = cfg.get("skip_liver", set())
            for seg_type in cfg["seg_types"]:
                if seg_type == "liver" and stem in skip_liver:
                    continue
                json_path = root / cls_name / "segmentation" / seg_type / f"{stem}.json"
                json_path.write_text(polygon_json)


@pytest.fixture(scope="session")
def aul_root(data_root):
    r = data_root / "AUL"; build_aul(r); return r


def build_us105(root: Path):
    """
    Synthetic 105US dataset.

    4 images: 001, 002, 003, 004
    Masks for 001, 002, 003 only — 004 has no mask (ssl_only path).
    """
    img_dir  = root / "105 US Images"
    mask_dir = root / "105 US Masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for stem in ("001", "002", "003", "004"):
        _save_png(_gray(8, 8), img_dir / f"{stem}.png")

    gray_mask = _gray(8, 8)
    for stem in ("001", "002", "003"):
        _save_png(gray_mask, mask_dir / f"{stem} G man.png")

    # readme.txt present in real dataset — adapter must ignore it
    (img_dir / "readme.txt").write_text("placeholder")


@pytest.fixture(scope="session")
def us105_root(data_root):
    r = data_root / "105US"; build_us105(r); return r


@pytest.fixture
def tmp_manifest_with_masks(tmp_path):
    """A tiny manifest file with train/val entries and mask data."""
    from data.schema.manifest import USManifestEntry, Instance, ManifestWriter
    entries = []
    for i in range(6):
        split = "train" if i < 4 else "val"
        img   = tmp_path / f"img_{i:04d}.png"
        mask  = tmp_path / f"mask_{i:04d}.png"
        img.write_bytes(b"\x00" * 64)
        mask.write_bytes(b"\x00" * 64)
        inst = Instance(
            instance_id=f"inst_{i}",
            label_raw="thyroid_nodule",
            label_ontology="thyroid_nodule_boundary",
            anatomy_family="thyroid",
            mask_path=str(mask),
        )
        e = USManifestEntry(
            sample_id=f"seg_{i:04d}",
            dataset_id="TEST_SEG",
            anatomy_family="thyroid",
            modality_type="image",
            split=split,
            image_paths=[str(img)],
            has_mask=True,
            instances=[inst],
        )
        entries.append(e)
    out = tmp_path / "test_manifest.jsonl"
    with ManifestWriter(out) as w:
        for e in entries:
            w.write(e)
    return out
