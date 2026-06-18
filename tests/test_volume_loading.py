"""
tests/test_volume_loading.py  ·  Medical volume format loading tests
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("SimpleITK")
import SimpleITK as sitk

from data.pipeline.dataset import load_image


def test_load_image_extracts_slice_from_nrrd_volume(tmp_path):
    pixels = np.zeros((5, 8, 8), dtype=np.float32)
    pixels[2, 2:6, 2:6] = 200
    nrrd_path = tmp_path / "volume.nrrd"
    sitk.WriteImage(sitk.GetImageFromArray(pixels), str(nrrd_path))

    img = load_image(str(nrrd_path), frame_idx=2)
    assert img.ndim in (2, 3)
    assert int(img.max()) > 150


def test_load_image_extracts_slice_from_minc_volume(tmp_path):
    pixels = np.zeros((5, 8, 8), dtype=np.float32)
    pixels[2, 2:6, 2:6] = 200
    mnc_path = tmp_path / "volume.mnc"
    sitk.WriteImage(sitk.GetImageFromArray(pixels), str(mnc_path))

    img = load_image(str(mnc_path), frame_idx=2)
    assert img.ndim in (2, 3)
    assert int(img.max()) > 150
