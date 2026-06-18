"""Shared helpers for brain 3D volume adapters."""
from __future__ import annotations

from pathlib import Path


def nifti_depth(vol_path: Path) -> int:
    """Return the first-axis size of a NIfTI volume without loading voxels."""
    try:
        import nibabel as nib

        return int(nib.load(str(vol_path)).shape[0])
    except Exception:
        return 1


def dicom_num_frames(dcm_path: Path) -> int:
    """Return NumberOfFrames from a DICOM header, defaulting to 1."""
    try:
        import pydicom

        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        return max(frames, 1)
    except Exception:
        return 1


def dicom_series_description(dcm_path: Path) -> str | None:
    try:
        import pydicom

        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        desc = getattr(ds, "SeriesDescription", None)
        return str(desc).strip() if desc else None
    except Exception:
        return None
