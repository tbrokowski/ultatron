"""Tests for NIfTI volume decoding."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def _write_minimal_nifti(path: Path, data: np.ndarray) -> None:
    """Write a tiny NIfTI-1 file with uint8 voxels in Fortran layout."""
    assert data.ndim in (2, 3)
    ndim = data.ndim
    header = bytearray(348)
    struct.pack_into("<i", header, 0, 348)          # sizeof_hdr
    struct.pack_into("<h", header, 40, ndim)        # dim[0]
    for i, d in enumerate(data.shape, start=1):
        struct.pack_into("<h", header, 40 + 2 * i, d)
    struct.pack_into("<h", header, 70, 2)          # DT_UINT8
    struct.pack_into("<f", header, 108, 352.0)     # vox_offset
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\x00" * 4)  # pad to vox_offset=352
        f.write(data.astype(np.uint8, order="F").tobytes(order="F"))


def test_read_nifti_fortran_order(tmp_path):
    """NIfTI dim[1]-fastest layout must not be scrambled by C-order reshape."""
    from data.pipeline.dataset import _read_nifti_array

    nx, ny, nz = 6, 5, 4
    vol = np.zeros((nx, ny, nz), dtype=np.uint8, order="F")
    vol[2, 3, 1] = 200

    path = tmp_path / "vol.nii"
    _write_minimal_nifti(path, vol)

    arr = _read_nifti_array(str(path))
    assert arr.shape == (nz, ny, nx)
    assert arr[1, 3, 2] == 200
    assert arr[0].max() == 0
    assert arr[2].max() == 0


def test_load_image_skips_blank_leading_slice(tmp_path):
    """Volumes with zero-padded slice 0 should fall back to the middle plane."""
    from data.pipeline.dataset import load_image

    nx, ny, nz = 3, 4, 5
    vol = np.zeros((nx, ny, nz), dtype=np.uint8, order="F")
    vol[1, 2, 2] = 180

    path = tmp_path / "padded.nii"
    _write_minimal_nifti(path, vol)

    img = load_image(str(path), frame_idx=0)
    assert img.max() > 0
    assert img.min() == 0


def test_select_volume_slice_collapses_3d_npy_volume():
    """3-D .npy volumes must reduce to 2-D before display."""
    from data.pipeline.dataset import _select_volume_slice

    vol = np.zeros((88, 118, 81), dtype=np.uint8)
    vol[44, 59, 40] = 200

    plane = _select_volume_slice(vol, frame_idx=-1)
    assert plane.ndim == 2
    assert plane.shape == (118, 81)
    assert plane.max() == 200
