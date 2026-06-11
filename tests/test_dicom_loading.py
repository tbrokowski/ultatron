"""
tests/test_dicom_loading.py  ·  DICOM pixel decoding tests
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian

from data.pipeline.dataset import (
    _dicom_frames,
    _dicom_photometric_to_rgb,
    _normalize_dicom_frames_to_uint8,
    _split_dicom_pixel_array,
    load_image,
)


def _write_dicom(path, pixel_array, *, photometric="MONOCHROME2", samples_per_pixel=1):
    import pydicom
    from pydicom.uid import generate_uid

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = samples_per_pixel
    if samples_per_pixel > 1:
        ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns = pixel_array.shape[-2], pixel_array.shape[-1]
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    if pixel_array.ndim == 3 and pixel_array.shape[0] > 1 and samples_per_pixel == 1:
        ds.NumberOfFrames = pixel_array.shape[0]
    elif pixel_array.ndim == 4:
        ds.NumberOfFrames = pixel_array.shape[0]
    ds.PixelData = pixel_array.astype(np.uint8).tobytes()
    pydicom.dcmwrite(str(path), ds)


def test_split_dicom_pixel_array_multiframe_monochrome():
    arr = np.arange(12, dtype=np.uint8).reshape(3, 2, 2)
    frames = _split_dicom_pixel_array(arr)
    assert len(frames) == 3
    assert frames[1].shape == (2, 2)


def test_monochrome1_inversion():
    frame = np.array([[0, 255], [64, 191]], dtype=np.float32)
    ds = Dataset()
    ds.PhotometricInterpretation = "MONOCHROME1"
    rgb = _dicom_photometric_to_rgb(frame, ds)
    assert rgb[0, 0, 0] == 255
    assert rgb[0, 1, 0] == 0


def test_ybr_full_422_uses_luma_channel():
    """YBR ultrasound cines must use the Y (luma) channel, not raw YBR as RGB."""
    ybr = np.zeros((4, 4, 3), dtype=np.float32)
    ybr[..., 0] = 100   # Y — B-mode luma
    ybr[..., 1] = 130   # Cb — would appear as green if shown as G
    ybr[..., 2] = 11    # Cr — would appear as blue if shown as B

    ds = Dataset()
    ds.PhotometricInterpretation = "YBR_FULL_422"
    rgb = _dicom_photometric_to_rgb(ybr, ds)
    assert rgb.shape == (4, 4, 3)
    assert np.allclose(rgb[..., 0], 100)
    assert np.allclose(rgb[..., 1], 100)
    assert np.allclose(rgb[..., 2], 100)


def test_clip_level_normalization_is_shared_across_frames():
    frames = [
        np.zeros((2, 2, 3), dtype=np.float32),
        np.full((2, 2, 3), 100.0, dtype=np.float32),
    ]
    out = _normalize_dicom_frames_to_uint8(frames)
    assert out[0].mean() == 0
    assert out[1].mean() == 255


def test_load_image_picks_middle_frame_for_multiframe_volume(tmp_path):
    pixels = np.zeros((5, 8, 8), dtype=np.uint8)
    pixels[0] = 0
    pixels[2] = 200
    pixels[4] = 50
    dcm_path = tmp_path / "volume.dcm"
    _write_dicom(dcm_path, pixels, photometric="MONOCHROME2")

    img = load_image(str(dcm_path))
    assert img.ndim in (2, 3)
    peak = int(img.max())
    assert peak > 150


def test_dicom_frames_returns_rgb_uint8(tmp_path):
    pixels = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    dcm_path = tmp_path / "rgb.dcm"
    _write_dicom(dcm_path, pixels, photometric="RGB", samples_per_pixel=3)

    frames = _dicom_frames(str(dcm_path))
    assert len(frames) >= 1
    assert frames[0].dtype == np.uint8
    assert frames[0].shape == (16, 16, 3)
