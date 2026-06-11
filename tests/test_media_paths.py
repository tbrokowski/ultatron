"""
tests/test_media_paths.py  ·  Archive paths, Roboflow names, HDF5 frames
"""
from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest
from PIL import Image

from data.pipeline.dataset import (
    image_path_extension,
    load_image,
    media_path_exists,
)


def test_image_path_extension_roboflow_suffix():
    path = "/tmp/sample_jpg.rf.7db9dd96475893e8d211ae8e11248d14.jpg"
    assert image_path_extension(path) == ".jpg"


def test_media_path_exists_for_zip_member(tmp_path):
    img = Image.new("RGB", (8, 6), color=(10, 20, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    payload = buf.getvalue()

    zpath = tmp_path / "images.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("folder/sample.jpg", payload)

    member_path = f"{zpath}::folder/sample.jpg"
    assert media_path_exists(member_path)
    loaded = load_image(member_path)
    assert loaded.shape == (6, 8, 3)
    assert loaded.dtype == np.uint8


def test_media_path_exists_for_extracted_zip_fallback(tmp_path):
    img = Image.new("RGB", (5, 4), color=(1, 2, 3))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")

    zpath = tmp_path / "image_Data.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("BEH00001/Image1.jpg", buf.getvalue())

    extract_dir = tmp_path / "image_Data" / "BEH00001"
    extract_dir.mkdir(parents=True)
    (extract_dir / "Image1.jpg").write_bytes(buf.getvalue())

    member_path = f"{zpath}::BEH00001/Image1.jpg"
    assert media_path_exists(member_path)
    loaded = load_image(member_path)
    assert loaded.shape == (4, 5, 3)


def test_load_image_h5_frames(tmp_path):
    pytest.importorskip("h5py")
    import h5py

    h5_path = tmp_path / "scan.h5"
    frames = np.stack([
        np.zeros((4, 4), dtype=np.uint8),
        np.full((4, 4), 200, dtype=np.uint8),
    ], axis=0)
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("frames", data=frames)

    img = load_image(str(h5_path), frame_idx=1)
    assert img.shape == (4, 4, 3)
    assert int(img.max()) == 200
