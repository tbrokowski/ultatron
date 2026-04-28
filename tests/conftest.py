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

def _save_mha(arr, path: Path, *, vector_rgb: bool = False):
    """Write a minimal uncompressed uint8 MHA file for adapter/loader tests."""
    arr = np.asarray(arr, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    if vector_rgb:
        if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
            raise ValueError("vector_rgb expects H x W x C input")
        h, w, c = arr.shape
        header = (
            "ObjectType = Image\n"
            "NDims = 2\n"
            f"DimSize = {w} {h}\n"
            "ElementType = MET_UCHAR\n"
            f"ElementNumberOfChannels = {c}\n"
            "ElementSpacing = 1 1\n"
            "Offset = 0 0\n"
            "ElementDataFile = LOCAL\n"
        )
        payload = arr.tobytes(order="C")
    else:
        if arr.ndim == 2:
            h, w = arr.shape
            dim_size = f"{w} {h}"
            spacing = "1 1"
            offset = "0 0"
        elif arr.ndim == 3:
            z, h, w = arr.shape
            dim_size = f"{w} {h} {z}"
            spacing = "1 1 1"
            offset = "0 0 0"
        else:
            raise ValueError("arr must be 2-D or 3-D")
        header = (
            "ObjectType = Image\n"
            f"NDims = {arr.ndim}\n"
            f"DimSize = {dim_size}\n"
            "ElementType = MET_UCHAR\n"
            f"ElementSpacing = {spacing}\n"
            f"Offset = {offset}\n"
            "ElementDataFile = LOCAL\n"
        )
        payload = arr.tobytes(order="C")
    path.write_bytes(header.encode("ascii") + payload)


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
    planes = [
        ("Patient00001_Plane1_1_of_2", "Other",           "Not A Brain",       "1", "Op. 1", "Voluson E6"),
        ("Patient00001_Plane2_2_of_2", "Fetal brain",     "Trans-thalamic",    "1", "Op. 1", "Voluson E6"),
        ("Patient00002_Plane3_1_of_1", "Fetal abdomen",   "Not A Brain",       "1", "Op. 2", "Aloka"),
        ("Patient00003_Plane4_1_of_1", "Fetal femur",     "Not A Brain",       "1", "Op. 3", "Voluson S10"),
        ("Patient00004_Plane5_1_of_1", "Fetal thorax",    "Not A Brain",       "0", "Other", "Other"),
        ("Patient00005_Plane6_1_of_1", "Maternal cervix", "Not A Brain",       "0", "Op. 2", "Aloka"),
    ]
    rows = []
    for idx, (stem, plane, brain_plane, train_flag, operator, machine) in enumerate(planes, start=1):
        if PIL_OK:
            Image.fromarray(np.zeros((8, 8, 4), dtype=np.uint8)).save(
                root / "Images" / f"{stem}.png"
            )
        else:
            _save_png(_gray(8, 8), root / "Images" / f"{stem}.png")
        rows.append({
            "Image_name": stem,
            "Patient_num": str(idx if idx > 2 else 1),
            "Plane": plane,
            "Brain_plane": brain_plane,
            "Operator": operator,
            "US_Machine": machine,
            "Train ": f"{train_flag} ",
        })

    # This metadata row should be ignored because its image is absent.
    rows.append({
        "Image_name": "Patient99999_Plane1_1_of_1",
        "Patient_num": "99999",
        "Plane": "Other",
        "Brain_plane": "Not A Brain",
        "Operator": "Other",
        "US_Machine": "Other",
        "Train ": "1 ",
    })
    with open(root / "FETAL_PLANES_DB_data.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Image_name", "Patient_num", "Plane", "Brain_plane",
                "Operator", "US_Machine", "Train ",
            ],
            delimiter=";",
        )
        w.writeheader(); w.writerows(rows)

def build_fpus23(root: Path):
    """
    Synthetic FPUS23 archive with both sub-datasets:
      - Dataset/four_poses: 3 streams, 2 frames each, CVAT XML annotations.
      - Dataset_Plane: 4 classes, 2 images each.
    """
    archive = root / "archive"
    poses_root = archive / "Dataset" / "four_poses"
    boxes_root = archive / "Dataset" / "boxes" / "annotation"
    annos_root = archive / "Dataset" / "annos" / "annotation"
    plane_root = archive / "Dataset_Plane"

    streams = [
        ("stream_hdvb_aroundabd_h", "hdvb", "h"),
        ("stream_huvb_aroundhead_v", "huvb", "v"),
        ("stream_hdvf_longrec_h", "hdvf", "h"),
    ]

    rgb = np.stack([_gray(8, 8)] * 3, axis=-1)
    for stream_name, pose, probe in streams:
        stream_dir = poses_root / stream_name
        box_dir = boxes_root / stream_name
        anno_dir = annos_root / stream_name
        stream_dir.mkdir(parents=True, exist_ok=True)
        box_dir.mkdir(parents=True, exist_ok=True)
        anno_dir.mkdir(parents=True, exist_ok=True)

        for i in range(2):
            _save_png(rgb, stream_dir / f"frame_{i:06d}.png")

        boxes_xml = f"""<annotations>
  <image id="0" name="frame_000000.png" width="8" height="8">
    <box label="abdomen" xtl="1.0" ytl="1.0" xbr="5.0" ybr="6.0" />
    <box label="arm" xtl="2.0" ytl="2.0" xbr="4.0" ybr="4.0" />
    <tag label="Orientation"><attribute name="Pose">{pose}</attribute></tag>
    <tag label="Probe"><attribute name="orientation">{probe}</attribute></tag>
    <tag label="location"><attribute name="View_fetus">abdomen</attribute></tag>
  </image>
  <image id="1" name="frame_000001.png" width="8" height="8">
    <tag label="Orientation"><attribute name="Pose">{pose}</attribute></tag>
    <tag label="Probe"><attribute name="orientation">{probe}</attribute></tag>
  </image>
</annotations>
"""
        annos_xml = f"""<annotations>
  <image id="0" name="frame_000000.png" width="8" height="8">
    <tag label="Orientation"><attribute name="Pose">{pose}</attribute></tag>
  </image>
  <image id="1" name="frame_000001.png" width="8" height="8">
    <tag label="Orientation"><attribute name="Pose">{pose}</attribute></tag>
  </image>
</annotations>
"""
        (box_dir / "annotations.xml").write_text(boxes_xml)
        (anno_dir / "annotations.xml").write_text(annos_xml)

    for class_name in ("AC_PLANE", "BPD_PLANE", "FL_PLANE", "NO_PLANE"):
        class_dir = plane_root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(2):
            _save_png(rgb, class_dir / f"{class_name.lower()}_{i}.png")

def build_focus(root: Path):
    """
    Synthetic FOCUS dataset across official split folders.

    training/001 has both masks.
    training/002 has only cardiac mask to test zero-filled missing thorax.
    validation/003 has only thorax mask.
    testing/004 has no masks but has rectangle annotations.
    """
    samples = [
        ("training", "001", ("cardiac", "thorax"), True),
        ("training", "002", ("cardiac",), True),
        ("validation", "003", ("thorax",), True),
        ("testing", "004", (), False),
    ]

    for split_dir, stem, masks, grayscale in samples:
        base = root / split_dir
        img_dir = base / "images"
        mask_dir = base / "annfiles_mask"
        ellipse_dir = base / "annfiles_ellipse"
        rectangle_dir = base / "annfiles_rectangle"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        ellipse_dir.mkdir(parents=True, exist_ok=True)
        rectangle_dir.mkdir(parents=True, exist_ok=True)

        if grayscale:
            _save_png(_gray(8, 8), img_dir / f"{stem}.png")
        else:
            rgb = np.stack([_gray(8, 8)] * 3, axis=-1)
            _save_png(rgb, img_dir / f"{stem}.png")

        for mask_name in masks:
            mask = np.zeros((8, 8), dtype=np.uint8)
            if mask_name == "cardiac":
                mask[1:4, 1:4] = 255
            else:
                mask[4:7, 4:7] = 255
            _save_png(mask, mask_dir / f"{stem}-{mask_name}.png")

        (ellipse_dir / f"{stem}.txt").write_text(
            "[4, 4, 2, 3, 15] cardiac\n"
            "[5, 5, 3, 2, 20] thorax\n"
        )
        (rectangle_dir / f"{stem}.txt").write_text(
            "[1, 1, 4, 1, 4, 5, 1, 5] cardiac 0\n"
            "[3, 3, 7, 3, 7, 7, 3, 7] thorax 1\n"
        )

def build_psfhs(root: Path):
    """
    Synthetic PSFHS layout nested under PSFHS/.

    00001 uses a vector RGB MHA image.
    00002 uses a 3-plane MHA image to exercise channel-first handling.
    00003 has no label mask to exercise the ssl_only path.
    """
    inner = root / "PSFHS"
    img_dir = inner / "image_mha"
    label_dir = inner / "label_mha"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    rgb_hwc = np.zeros((8, 8, 3), dtype=np.uint8)
    rgb_hwc[..., 0] = 10
    rgb_hwc[..., 1] = 80
    rgb_hwc[..., 2] = 200
    _save_mha(rgb_hwc, img_dir / "00001.mha", vector_rgb=True)

    rgb_chw = np.zeros((3, 8, 8), dtype=np.uint8)
    rgb_chw[0] = 20
    rgb_chw[1] = 90
    rgb_chw[2] = 220
    _save_mha(rgb_chw, img_dir / "00002.mha")

    _save_mha(rgb_hwc, img_dir / "00003.mha", vector_rgb=True)

    for stem in ("00001", "00002"):
        label = np.zeros((8, 8), dtype=np.uint8)
        label[1:4, 1:4] = 1
        label[4:7, 4:7] = 2
        _save_mha(label, label_dir / f"{stem}.mha")

def build_hc18(root: Path):
    """
    Synthetic HC18 dataset using the real naming convention.

    Training patients and sweeps:
      001_HC    patient 001, sweep 1  — image + annotation + CSV row
      001_2HC   patient 001, sweep 2  — image + annotation + CSV row
      002_HC    patient 002           — image + annotation + CSV row
      003_HC    patient 003           — image + annotation + CSV row (no mask for ssl path)
    Test set:
      004_HC, 005_HC  — images only + pixel-size CSV
    """
    train_dir = root / "training_set"
    test_dir  = root / "test_set"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_samples = [
        ("001_HC",  True,  0.154, 178.3),
        ("001_2HC", True,  0.154, 179.1),
        ("002_HC",  True,  0.161, 185.0),
        ("003_HC",  False, 0.160, None),   # no annotation → ssl_only
    ]
    for stem, has_ann, _px, _hc in train_samples:
        _save_png(_gray(), train_dir / f"{stem}.png")
        if has_ann:
            _save_png(_mask(), train_dir / f"{stem}_Annotation.png")

    with open(root / "training_set_pixel_size_and_HC.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "pixel size(mm)", "head circumference (mm)"])
        w.writeheader()
        for stem, _ann, px, hc in train_samples:
            w.writerow({"filename": f"{stem}.png",
                        "pixel size(mm)": str(px) if px else "",
                        "head circumference (mm)": str(hc) if hc else ""})

    test_samples = [("004_HC", 0.158), ("005_HC", 0.162)]
    for stem, px in test_samples:
        _save_png(_gray(), test_dir / f"{stem}.png")

    with open(root / "test_set_pixel_size.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "pixel size(mm)"])
        w.writeheader()
        for stem, px in test_samples:
            w.writerow({"filename": f"{stem}.png", "pixel size(mm)": str(px)})

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
def fpus23_root(data_root):
    r = data_root / "FPUS23"; build_fpus23(r); return r

@pytest.fixture(scope="session")
def focus_root(data_root):
    r = data_root / "FOCUS"; build_focus(r); return r

@pytest.fixture(scope="session")
def psfhs_root(data_root):
    r = data_root / "PSFHS-parent"; build_psfhs(r); return r

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


def build_acouslic(root: Path):
    """
    Synthetic ACOUSLIC-AI dataset.

    4 sweeps across 3 subjects:
      sweep-aaa  subject 01  ac_mm=250.0  image+mask
      sweep-bbb  subject 01  ac_mm=252.0  image+mask  (same patient, two sweeps)
      sweep-ccc  subject 02  ac_mm=270.0  image+mask
      sweep-ddd  subject 03  no ac data   image only   (mask absent → ssl_only)
    """
    train_set = root / "acouslic-ai-train-set"
    img_dir  = train_set / "images" / "stacked_fetal_ultrasound"
    mask_dir = train_set / "masks"  / "stacked_fetal_abdomen"
    circ_dir = train_set / "circumferences"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    circ_dir.mkdir(parents=True, exist_ok=True)

    sweeps = [
        ("sweep-aaa", "01", "250.0", ""),
        ("sweep-bbb", "01", "",      "252.0"),
        ("sweep-ccc", "02", "270.0", ""),
        ("sweep-ddd", "03", "",      ""),
    ]
    for uuid, _sid, _ac1, _ac2 in sweeps:
        (img_dir / f"{uuid}.mha").write_bytes(b"MHA-stub")

    for uuid, _sid, _ac1, _ac2 in sweeps[:3]:  # sweep-ddd has no mask
        (mask_dir / f"{uuid}.mha").write_bytes(b"MHA-stub")

    rows = [
        {"uuid": uuid, "subject_id": sid,
         "sweep_1_ac_mm": ac1, "sweep_2_ac_mm": ac2,
         "sweep_3_ac_mm": "", "sweep_4_ac_mm": "",
         "sweep_5_ac_mm": "", "sweep_6_ac_mm": ""}
        for uuid, sid, ac1, ac2 in sweeps
    ]
    fields = ["uuid", "subject_id",
              "sweep_1_ac_mm", "sweep_2_ac_mm", "sweep_3_ac_mm",
              "sweep_4_ac_mm", "sweep_5_ac_mm", "sweep_6_ac_mm"]
    with open(circ_dir / "fetal_abdominal_circumferences_per_sweep.csv",
              "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


@pytest.fixture(scope="session")
def acouslic_root(data_root):
    r = data_root / "ACOUSLIC-AI"
    build_acouslic(r)
    return r


def build_fass(root: Path):
    """
    Synthetic FASS dataset inside the long-named subdirectory.

    6 images across 3 patients (P01, P02, P03), 2 images each.
    All 6 have paired NPY files except the last one (ssl_only path).
    """
    inner = root / "Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images"
    img_dir  = inner / "IMAGES"
    mask_dir = inner / "ARRAY_FORMAT"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    samples = [
        ("P01_IMG1", True),
        ("P01_IMG2", True),
        ("P02_IMG1", True),
        ("P02_IMG2", True),
        ("P03_IMG1", True),
        ("P03_IMG2", False),  # no NPY → ssl_only
    ]

    h, w = 4, 4
    for stem, has_npy in samples:
        _save_png(_gray(h, w), img_dir / f"{stem}.png")
        if has_npy:
            data = {
                "image": np.zeros((h, w, 3), dtype=np.uint8),
                "structures": {
                    "artery":  np.zeros((h, w), dtype=np.uint8),
                    "liver":   np.zeros((h, w), dtype=np.uint8),
                    "stomach": np.zeros((h, w), dtype=np.uint8),
                    "vein":    np.zeros((h, w), dtype=np.uint8),
                },
            }
            np.save(str(mask_dir / f"{stem}.npy"), np.array(data, dtype=object))


@pytest.fixture(scope="session")
def fass_root(data_root):
    r = data_root / "FASS"
    build_fass(r)
    return r


def build_fh_ps_aop(root: Path):
    """
    Synthetic FH-PS-AOP dataset inside the long-named subdirectory.

    5 image/mask pairs (00001–00005) plus one image with no mask (00006)
    to exercise the ssl_only path.
    """
    inner    = root / "Pubic Symphysis-Fetal Head Segmentation and Angle of Progression"
    img_dir  = inner / "image_mha"
    mask_dir = inner / "label_mha"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 7):
        stem = f"{i:05d}"
        (img_dir / f"{stem}.mha").write_bytes(b"MHA-stub")
        if i < 6:  # 00006 has no mask → ssl_only
            (mask_dir / f"{stem}.mha").write_bytes(b"MHA-stub")


@pytest.fixture(scope="session")
def fh_ps_aop_root(data_root):
    r = data_root / "FH-PS-AOP"
    build_fh_ps_aop(r)
    return r


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
