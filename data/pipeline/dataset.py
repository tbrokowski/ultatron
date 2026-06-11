"""
dataset.py  ·  Unified Ultatron dataset classes
====================================================
This file contains the dataset classes for the Ultatron foundation model.

"""
from __future__ import annotations

import io
import logging
import os
import random
import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.schema.manifest import USManifestEntry, load_manifest
from data.pipeline.transforms import (
    ImageSSLTransform, ImageSSLTransformConfig,
    VideoSSLTransform, VideoSSLTransformConfig,
)


# ── Format helpers (no external deps beyond stdlib + numpy) ──────────────────

_ARCHIVE_SEP = "::"
_RF_SUFFIX_RE = re.compile(
    r"\.rf\.[0-9a-f]+\.(jpg|jpeg|png|bmp|tif|tiff)$", re.IGNORECASE,
)


def split_archive_path(path: str) -> tuple[str, Optional[str]]:
    """Split ``/path/archive.zip::member/in.zip`` into archive path and member."""
    if _ARCHIVE_SEP in path:
        archive, member = path.split(_ARCHIVE_SEP, 1)
        return archive, member
    return path, None


def _extracted_archive_fallback(archive: str, member: str) -> Optional[str]:
    """If ``archive.zip`` was partially extracted to ``archive/``, use that file."""
    zp = Path(archive)
    candidate = zp.parent / zp.stem / member
    if candidate.is_file():
        return str(candidate)
    return None


def _resolve_zip_member(zf: zipfile.ZipFile, member: str) -> str:
    names = zf.namelist()
    if member in names:
        return member
    alt = member.lstrip("/")
    if alt in names:
        return alt
    basename = Path(member).name
    matches = [n for n in names if Path(n).name == basename]
    if len(matches) == 1:
        return matches[0]
    suffix_matches = [
        n for n in names
        if n.endswith(member) or n.endswith("/" + member.lstrip("/"))
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    raise FileNotFoundError(f"Archive member not found: {member!r} in {zf.filename}")


def media_path_exists(path: str) -> bool:
    """Return True when a plain file or ``zip::member`` path is readable."""
    archive, member = split_archive_path(path)
    if member is not None:
        fallback = _extracted_archive_fallback(archive, member)
        if fallback is not None:
            return True
        zp = Path(archive)
        if not zp.is_file():
            return False
        try:
            with zipfile.ZipFile(zp) as zf:
                resolved = _resolve_zip_member(zf, member)
                # Probe readability for encrypted entries.
                with zf.open(resolved) as fh:
                    fh.read(1)
            return True
        except (FileNotFoundError, zipfile.BadZipFile, OSError, RuntimeError):
            return False
    return Path(archive).exists()


def resolve_media_path(
    path: str,
    root_remap: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Return the first readable variant of *path* (store/scratch/remapped)."""
    candidates: List[str] = []
    remapped = path
    for old, new in (root_remap or {}).items():
        if remapped.startswith(old):
            remapped = remapped.replace(old, new, 1)
            break
    candidates.append(remapped)
    if remapped != path:
        candidates.append(path)
    for old, new in (root_remap or {}).items():
        for p in list(candidates):
            if p.startswith(old):
                candidates.append(p.replace(old, new, 1))
            elif p.startswith(new):
                candidates.append(p.replace(new, old, 1))
    for p in dict.fromkeys(candidates):
        if media_path_exists(p):
            return p
    return None


def image_path_extension(path: str) -> str:
    """Return the loader extension, normalising Roboflow and compound suffixes."""
    _, member = split_archive_path(path)
    name = member if member is not None else path
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    if lower.endswith(".tar.gz"):
        return ".tar.gz"
    rf_match = _RF_SUFFIX_RE.search(name)
    if rf_match:
        return "." + rf_match.group(1).lower()
    suffixes = Path(name).suffixes
    return suffixes[-1].lower() if suffixes else ""


def _read_archive_bytes(path: str) -> tuple[Optional[bytes], str]:
    """Return ``(bytes, logical_name)`` for zip members, else ``(None, path)``."""
    archive, member = split_archive_path(path)
    if member is None:
        return None, path

    fallback = _extracted_archive_fallback(archive, member)
    if fallback is not None:
        return None, fallback

    with zipfile.ZipFile(archive) as zf:
        resolved = _resolve_zip_member(zf, member)
        try:
            return zf.read(resolved), resolved
        except RuntimeError as exc:
            if "password" in str(exc).lower() or "encrypted" in str(exc).lower():
                fallback = _extracted_archive_fallback(archive, member)
                if fallback is not None:
                    return None, fallback
            raise


def _frame_to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert a single 2-D or 3-D frame to ``(H, W, 3)`` uint8 RGB."""
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = arr.astype(np.float32)
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmax > vmin and vmax <= 1.0:
            arr = arr * 255.0
        elif vmax > vmin and arr.max() > 255:
            arr = (arr - vmin) / (vmax - vmin) * 255.0
        img = np.clip(arr, 0, 255).astype(np.uint8)
        return np.stack([img, img, img], axis=-1)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        out = arr[..., :3].astype(np.float32)
        mx = float(out.max())
        if mx <= 1.0:
            out = out * 255.0
        elif mx > 255:
            vmin, vmax = float(out.min()), float(out.max())
            out = (out - vmin) / (vmax - vmin + 1e-8) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    raise ValueError(f"Unsupported frame shape: {arr.shape}")


def _read_h5_frames(path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    try:
        import h5py
    except ImportError:
        raise RuntimeError("h5py required for HDF5 files")

    with h5py.File(path, "r") as f:
        key = "frames" if "frames" in f else list(f.keys())[0]
        arr = np.asarray(f[key])

    if arr.ndim == 4 and arr.shape[-1] in (1, 3, 4):
        raw_frames = [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 3:
        raw_frames = [arr[i] for i in range(arr.shape[0])]
    elif arr.ndim == 2:
        raw_frames = [arr]
    else:
        raise ValueError(f"Unsupported HDF5 array shape: {arr.shape}")

    frames = [_frame_to_rgb_uint8(f) for f in raw_frames]
    if max_frames and len(frames) > max_frames:
        step = max(1, len(frames) // max_frames)
        frames = frames[::step][:max_frames]
    return frames


def _read_nifti_array(path: str) -> np.ndarray:
    """Read a NIfTI1 (.nii / .nii.gz) file and return the raw voxel array.

    Returns an ndarray with shape transposed to (T-or-Z, Y, X) for 3-D volumes
    or (Y, X) for 2-D images.

    NIfTI voxel data is laid out with dim[1] varying fastest (Fortran /
    column-major order).  A C-order reshape scrambles the volume and produces
    striped artefacts when slicing — use ``order='F'`` before transposing to
    (Z, Y, X).
    """
    import gzip, struct as _struct
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        raw = f.read()
    endian = "<" if _struct.unpack_from("<i", raw, 0)[0] == 348 else ">"
    ndim   = _struct.unpack_from(f"{endian}h", raw, 40)[0]
    shape  = tuple(_struct.unpack_from(f"{endian}{ndim}h", raw, 42))
    datatype = _struct.unpack_from(f"{endian}h", raw, 70)[0]
    _dt_map  = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32,
                64: np.float64, 256: np.int8, 512: np.uint16, 768: np.uint32}
    dtype = np.dtype(_dt_map.get(datatype, np.float32)).newbyteorder(endian)
    vox_offset = int(_struct.unpack_from(f"{endian}f", raw, 108)[0])
    arr = np.frombuffer(raw[vox_offset:], dtype=dtype).reshape(shape, order="F")
    return arr.T  # (X, Y[, Z]) → (Z, Y, X) or (Y, X)


def _read_mhd_array(path: str) -> np.ndarray:
    """Read a MetaImage (.mhd/.mha) file and return the raw voxel array.

    Returns shape (nz, ny, nx) for 3-D or (ny, nx) for 2-D.
    The caller is responsible for selecting the desired slice/frame.
    """
    import struct as _struct
    header: dict = {}
    with open(path, "rb") as f:
        for line in f:
            line = line.decode("ascii", errors="ignore").strip()
            if "=" not in line:
                continue
            key, _, val = line.partition("=")
            header[key.strip()] = val.strip()
            if key.strip() == "ElementDataFile":
                break

    ndims     = int(header.get("NDims", 3))
    dim_sizes = [int(v) for v in header.get("DimSize", "").split()]
    elem_type = header.get("ElementType", "MET_UCHAR").upper()
    _et_map   = {"MET_UCHAR": np.uint8, "MET_CHAR": np.int8,
                 "MET_USHORT": np.uint16, "MET_SHORT": np.int16,
                 "MET_UINT": np.uint32, "MET_INT": np.int32,
                 "MET_FLOAT": np.float32, "MET_DOUBLE": np.float64}
    dtype     = np.dtype(_et_map.get(elem_type, np.uint8))
    msb       = header.get("BinaryDataByteOrderMSB", "False").lower() == "true"
    if msb:
        dtype = dtype.newbyteorder(">")

    data_file = header.get("ElementDataFile", "LOCAL")
    if data_file == "LOCAL":
        # Inline .mha — data follows the header in the same file
        with open(path, "rb") as f:
            content = f.read()
        # Find the end of the header (last "ElementDataFile = LOCAL\n")
        sentinel = b"ElementDataFile = LOCAL"
        idx = content.rfind(sentinel)
        raw = content[idx + len(sentinel):].lstrip(b"\r\n")
    else:
        raw_path = Path(path).parent / data_file
        compressed = header.get("CompressedData", "False").lower() == "true"
        if compressed:
            import gzip
            with gzip.open(str(raw_path), "rb") as f:
                raw = f.read()
        else:
            with open(str(raw_path), "rb") as f:
                raw = f.read()

    # DimSize is (nx, ny[, nz, ...]) — reshape in reverse for C-order (nz, ny, nx)
    shape = tuple(reversed(dim_sizes))
    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return arr


def _read_nrrd_array(path: str) -> np.ndarray:
    """Read an NRRD (.nrrd) file and return the raw voxel array.

    Returns shape (nz, ny, nx) for 3-D or (ny, nx) for 2-D (SimpleITK axis order).
    """
    import SimpleITK as sitk
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


# ── Image / video loading ─────────────────────────────────────────────────────

def _read_dicom_dataset(path: str):
    try:
        import pydicom
        from pydicom.dataset import FileMetaDataset
        from pydicom.uid import ImplicitVRLittleEndian
    except ImportError:
        raise RuntimeError("pydicom required for DICOM files")

    ds = pydicom.dcmread(path, force=True)
    if not hasattr(ds, "file_meta") or ds.file_meta is None:
        ds.file_meta = FileMetaDataset()
    if "TransferSyntaxUID" not in ds.file_meta:
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    return ds


def _rescale_dicom_pixels(arr: np.ndarray, ds) -> np.ndarray:
    """Apply RescaleSlope / RescaleIntercept when present."""
    slope = float(getattr(ds, "RescaleSlope", 1) or 1)
    intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
    if slope != 1.0 or intercept != 0.0:
        return arr.astype(np.float32) * slope + intercept
    return arr.astype(np.float32) if arr.dtype != np.uint8 else arr.astype(np.float32)


def _dicom_photometric_to_rgb(frame: np.ndarray, ds) -> np.ndarray:
    """Convert a single DICOM frame to RGB float32 (H, W, 3).

    Echocardiography cines (MIMIC-ECHO, LVVol-A4C, etc.) are stored as
    ``YBR_FULL_422`` with the B-mode image in the **Y (luma) channel**.  If YBR
    triplets are displayed as RGB without conversion, Cb maps to green and Cr to
    magenta — the green-background artefact seen in thumbnails.  For all YBR
    photometric types we therefore replicate the luma channel to RGB.
    """
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2")).upper()

    if frame.ndim == 2:
        if photometric == "MONOCHROME1":
            frame = frame.max() - frame
        rgb = np.stack([frame, frame, frame], axis=-1)
        return rgb

    if frame.ndim == 3 and frame.shape[-1] in (3, 4):
        ch = frame[..., :3].astype(np.float32)

        if photometric.startswith("YBR"):
            y = ch[..., 0]
            return np.stack([y, y, y], axis=-1)

        if photometric == "RGB":
            return ch

        try:
            try:
                from pydicom.pixels import convert_color_space
            except ImportError:
                from pydicom.pixel_data_handlers.util import convert_color_space
            u8 = np.clip(ch, 0, 255).astype(np.uint8)
            return convert_color_space(u8, photometric, "RGB").astype(np.float32)
        except Exception:
            return ch

    raise ValueError(f"Unsupported DICOM frame shape: {frame.shape}")


def _normalize_dicom_frames_to_uint8(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Normalize all frames using clip-level min/max to avoid per-frame flicker."""
    if not frames:
        return frames
    stacked = np.stack([f.astype(np.float32) for f in frames], axis=0)
    vmin, vmax = float(stacked.min()), float(stacked.max())
    if vmax <= vmin:
        return [np.zeros_like(f, dtype=np.uint8) for f in frames]
    scaled = (stacked - vmin) / (vmax - vmin) * 255.0
    return [scaled[i].astype(np.uint8) for i in range(scaled.shape[0])]


def _split_dicom_pixel_array(arr: np.ndarray) -> List[np.ndarray]:
    """Split a DICOM pixel_array into a list of per-frame arrays."""
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return [arr[..., :3]]
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):
        return [arr[i, ..., :3] for i in range(arr.shape[0])]
    raise ValueError(f"Unsupported DICOM pixel array shape: {arr.shape}")


def _dicom_frames(path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Load DICOM cine/volume as RGB uint8 frames with correct color space and
    clip-level normalization.
    """
    ds = _read_dicom_dataset(path)
    raw = _rescale_dicom_pixels(np.asarray(ds.pixel_array), ds)
    raw_frames = _split_dicom_pixel_array(raw)
    rgb_frames = [_dicom_photometric_to_rgb(f, ds) for f in raw_frames]
    frames = _normalize_dicom_frames_to_uint8(rgb_frames)

    if max_frames and len(frames) > max_frames:
        step = max(1, len(frames) // max_frames)
        frames = frames[::step][:max_frames]
    return frames


def load_image(path: str, frame_idx: int = 0) -> np.ndarray:
    """
    Load an image and return a uint8 numpy array preserving available channels.

    Returns
    -------
    (H, W, 3) uint8  for colour/RGB sources (PNG, JPG, NPY with 3 channels, etc.)
    (H, W)    uint8  for inherently single-channel sources (DICOM, MHD, NPY 2D)

    ``to_canonical_tensor`` in transforms.py handles both shapes correctly:
    2D arrays are channel-repeated to (3, H, W); (H, W, 3) arrays are permuted
    to (3, H, W) without modification.
    """
    raw_bytes, logical = _read_archive_bytes(path)
    disk_path = logical
    ext = image_path_extension(logical)

    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
        from PIL import Image
        src = io.BytesIO(raw_bytes) if raw_bytes is not None else disk_path
        return np.array(Image.open(src).convert("RGB"), dtype=np.uint8)

    if raw_bytes is not None:
        raise ValueError(f"Unsupported image format inside archive: {ext}")

    if ext in (".npy",):
        arr = np.load(disk_path)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = arr[..., :3]
            return arr.astype(np.uint8) if arr.max() > 1 else (arr * 255).astype(np.uint8)
        arr = _select_volume_slice(np.asarray(arr), frame_idx=frame_idx)
        return arr.astype(np.uint8) if arr.max() > 1 else (arr * 255).astype(np.uint8)

    if ext in (".npz",):
        d   = np.load(disk_path)
        arr = np.asarray(d[list(d.keys())[0]])
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = arr[..., :3]
            return arr.astype(np.uint8) if arr.max() > 1 else (arr * 255).astype(np.uint8)
        arr = _select_volume_slice(arr, frame_idx=frame_idx)
        return arr.astype(np.uint8) if arr.max() > 1 else (arr * 255).astype(np.uint8)

    if ext in (".mhd", ".mha"):
        arr = _select_volume_slice(_read_mhd_array(disk_path), frame_idx=frame_idx)
        arr = arr.astype(np.float32)
        img = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return img

    if ext in (".dcm",):
        frames = _dicom_frames(disk_path, max_frames=None)
        if not frames:
            raise ValueError(f"No frames decoded from DICOM: {disk_path}")
        idx = len(frames) // 2 if frame_idx < 0 else min(frame_idx, len(frames) - 1)
        frame = frames[idx]
        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            return frame[..., :3]
        if frame.ndim > 2:
            return _select_volume_slice(frame, frame_idx=0)
        return frame

    if ext in (".nii.gz", ".nii"):
        arr = _select_volume_slice(_read_nifti_array(disk_path), frame_idx=frame_idx)
        arr = arr.astype(np.float32)
        img = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return img

    if ext in (".nrrd",):
        arr = _select_volume_slice(_read_nrrd_array(disk_path), frame_idx=frame_idx)
        arr = arr.astype(np.float32)
        img = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return img

    if ext in (".h5", ".hdf5"):
        frames = _read_h5_frames(disk_path, max_frames=None)
        if not frames:
            raise ValueError(f"No frames decoded from HDF5: {disk_path}")
        return frames[min(frame_idx, len(frames) - 1)]

    raise ValueError(f"Unsupported image format: {ext}")


def load_video_frames(path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Load video frames and return a list of (H, W, 3) uint8 RGB numpy arrays.

    Colour is preserved so that Doppler and colour-overlay clips are not
    inadvertently collapsed to grayscale.  ``to_canonical_tensor`` in
    transforms.py converts each frame to a (3, H, W) float32 tensor.
    """
    ext = image_path_extension(path)

    if ext in (".dcm",):
        return _dicom_frames(path, max_frames=max_frames)

    if ext in (".h5", ".hdf5"):
        return _read_h5_frames(path, max_frames=max_frames)

    if ext in (".nii.gz", ".nii"):
        arr = _read_nifti_array(path)
        if arr.ndim <= 2:
            return [_frame_to_rgb_uint8(arr)]
        depth = arr.shape[0]
        if max_frames is None or max_frames >= depth:
            indices = list(range(depth))
        else:
            step = max(1, depth // max_frames)
            indices = list(range(0, depth, step))[:max_frames]
        return [_frame_to_rgb_uint8(_select_volume_slice(arr, frame_idx=i)) for i in indices]

    if ext in (".mhd", ".mha"):
        arr = _read_mhd_array(path)
        if arr.ndim <= 2:
            return [_frame_to_rgb_uint8(arr)]
        depth = arr.shape[0]
        if max_frames is None or max_frames >= depth:
            indices = list(range(depth))
        else:
            step = max(1, depth // max_frames)
            indices = list(range(0, depth, step))[:max_frames]
        return [_frame_to_rgb_uint8(_select_volume_slice(arr, frame_idx=i)) for i in indices]

    if ext in (".nrrd",):
        arr = _read_nrrd_array(path)
        if arr.ndim <= 2:
            return [_frame_to_rgb_uint8(arr)]
        depth = arr.shape[0]
        if max_frames is None or max_frames >= depth:
            indices = list(range(depth))
        else:
            step = max(1, depth // max_frames)
            indices = list(range(0, depth, step))[:max_frames]
        return [_frame_to_rgb_uint8(_select_volume_slice(arr, frame_idx=i)) for i in indices]

    if ext in (".avi", ".mp4", ".mov", ".mkv", ".gif"):
        try:
            from decord import VideoReader, cpu
            vr      = VideoReader(path, ctx=cpu(0))
            indices = list(range(len(vr))) if max_frames is None \
                      else list(range(0, len(vr), max(1, len(vr) // max_frames)))[:max_frames]
            # decord returns (T, H, W, 3) uint8 RGB
            return list(vr.get_batch(indices).asnumpy())
        except ImportError:
            pass

        try:
            import cv2
            cap    = cv2.VideoCapture(path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if max_frames and len(frames) >= max_frames: break
            cap.release()
            return frames
        except ImportError:
            pass

        try:
            import torchvision.io as tvio
            vframes, _, _ = tvio.read_video(path, output_format="TCHW")
            # vframes: (T, C, H, W) uint8 — convert each to (H, W, C)
            frames = [vframes[i].permute(1, 2, 0).numpy() for i in range(len(vframes))]
            if max_frames: frames = frames[:max_frames]
            return frames
        except Exception as e:
            raise RuntimeError(f"Cannot load video {path}: {e}")

    raise ValueError(f"Unsupported video format: {ext}")


_PICKLED_MASK_PATH_MARKERS = (
    "Fetal Abdominal Structures Segmentation Dataset Using Ultrasonic Images",
    "ARRAY_FORMAT",
)


def _load_numpy_mask(path: str):
    try:
        return np.load(path, allow_pickle=False)
    except ValueError as exc:
        if "Object arrays cannot be loaded" not in str(exc):
            raise
        if not all(marker in path for marker in _PICKLED_MASK_PATH_MARKERS):
            raise ValueError(
                f"Refusing to load pickled NumPy mask outside trusted legacy mask paths: {path}"
            ) from exc
        return np.load(path, allow_pickle=True)


_VOLUME_SLICE_EXTS = (".nii.gz", ".nii", ".mhd", ".mha", ".nrrd")


def _select_volume_slice(arr: np.ndarray, frame_idx: int = 0) -> np.ndarray:
    """Select a 2-D plane from a volume; ``frame_idx < 0`` picks the middle."""
    arr = np.asarray(arr)
    if arr.ndim <= 2:
        return arr

    # Already a displayable RGB / greyscale image (H, W, C).
    if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        if arr.shape[-1] == 1:
            return arr[..., 0]
        return arr

    depth = arr.shape[0]
    idx = depth // 2 if frame_idx < 0 else min(frame_idx, depth - 1)
    plane = arr[idx]
    # Some brain US volumes (e.g. ReMIND2Reg) pad slice 0 with zeros.
    if frame_idx == 0 and plane.size > 0 and float(plane.max()) == float(plane.min()):
        idx = depth // 2
        plane = arr[idx]
    # Recurse for 4-D stacks (e.g. NIfTI with extra dim, multi-channel volumes).
    if plane.ndim > 2:
        return _select_volume_slice(plane, frame_idx=0)
    return plane


def _select_mask_plane(
    arr: np.ndarray,
    frame_idx: int = 0,
    mask_channel: Optional[int] = None,
    layout: str = "frame_first",
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim <= 2:
        return arr

    if mask_channel is not None:
        if layout == "channel_first":
            if mask_channel >= arr.shape[0]:
                raise ValueError(f"mask_channel={mask_channel} out of range for shape={arr.shape}")
            return arr[mask_channel]
        if layout == "channel_last":
            if mask_channel >= arr.shape[-1]:
                raise ValueError(f"mask_channel={mask_channel} out of range for shape={arr.shape}")
            return arr[..., mask_channel]
        raise ValueError(f"mask_channel requires channel layout, got layout={layout!r}")

    if layout != "frame_first":
        raise ValueError(f"frame_idx selection requires frame_first layout, got layout={layout!r}")

    return _select_volume_slice(arr, frame_idx=frame_idx)


def _select_npz_array(path: str, data) -> np.ndarray:
    keys = sorted(data.files)
    for key in ("mask", "masks", "segmentation", "label", "labels"):
        if key in keys:
            return data[key]
    if len(keys) == 1:
        return data[keys[0]]
    raise ValueError(
        f"Ambiguous NumPy mask archive {path}: expected one array or a key named "
        f"'mask', 'masks', 'segmentation', 'label', or 'labels'; found {keys}"
    )


def load_mask(path: str, frame_idx: int = 0, mask_channel: Optional[int] = None) -> np.ndarray:
    ext = image_path_extension(path)
    if ext in (".npy",):
        loaded = _load_numpy_mask(path)
        if loaded.shape == () and loaded.dtype == object:
            loaded = loaded.item()
        selected_structure = False
        if isinstance(loaded, dict):
            if "structures" in loaded:
                structures = loaded["structures"]
                keys = sorted(structures.keys())
                key = keys[mask_channel] if mask_channel is not None and mask_channel < len(keys) else keys[0]
                arr = np.asarray(structures[key])
                selected_structure = True
            elif "mask" in loaded:
                arr = np.asarray(loaded["mask"])
            else:
                raise ValueError(f"Unsupported NumPy mask dict keys for {path}: {sorted(loaded.keys())}")
        else:
            arr = np.asarray(loaded)
        # NumPy mask stacks in the current manifest use channel-first layout
        # when mask_channel is present. Other mask formats below treat
        # mask_channel as a categorical label value after frame selection.
        selected_channel_stack = mask_channel is not None and arr.ndim > 2
        layout = "channel_first" if selected_channel_stack else "frame_first"
        arr = _select_mask_plane(
            arr,
            frame_idx=frame_idx,
            mask_channel=mask_channel if selected_channel_stack else None,
            layout=layout,
        )
        if mask_channel is not None and not selected_channel_stack and not selected_structure:
            return (arr == mask_channel).astype(np.uint8)
        return (arr > 0).astype(np.uint8)
    if ext in (".npz",):
        data = np.load(path)
        arr = _select_npz_array(path, data)
        selected_channel_stack = mask_channel is not None and arr.ndim > 2
        layout = "channel_first" if selected_channel_stack else "frame_first"
        arr = _select_mask_plane(
            arr,
            frame_idx=frame_idx,
            mask_channel=mask_channel if selected_channel_stack else None,
            layout=layout,
        )
        if mask_channel is not None and not selected_channel_stack:
            return (arr == mask_channel).astype(np.uint8)
        return (arr > 0).astype(np.uint8)
    if ext in (".nii.gz", ".nii"):
        arr = _read_nifti_array(path)
        arr = _select_mask_plane(arr, frame_idx=frame_idx)
        if mask_channel is not None:
            return (arr == mask_channel).astype(np.uint8)
        return (arr > 0).astype(np.uint8)
    if ext in (".mhd", ".mha"):
        arr = _read_mhd_array(path)
        arr = _select_mask_plane(arr, frame_idx=frame_idx)
        if mask_channel is not None:
            return (arr == mask_channel).astype(np.uint8)
        return (arr > 0).astype(np.uint8)
    from PIL import Image
    raw_bytes, _ = _read_archive_bytes(path)
    src = io.BytesIO(raw_bytes) if raw_bytes is not None else path
    mask = np.array(Image.open(src).convert("L"), dtype=np.uint8)
    if mask_channel is not None:
        return (mask == mask_channel).astype(np.uint8)
    return (mask > 127).astype(np.uint8)


# ── Base dataset ──────────────────────────────────────────────────────────────

class USFoundationDataset(Dataset):
    def __init__(self, entries: List[USManifestEntry],
                 root_remap: Optional[Dict[str, str]] = None):
        self.entries    = entries
        self.root_remap = root_remap or {}

    def _remap_path(self, p: str) -> str:
        for old, new in self.root_remap.items():
            if p.startswith(old):
                return p.replace(old, new, 1)
        return p

    def _volume_slice_index(self, path: str, entry: USManifestEntry) -> int:
        meta = entry.source_meta or {}
        if "frame_idx" in meta and meta["frame_idx"] is not None:
            return int(meta["frame_idx"])
        if entry.modality_type != "volume":
            return 0

        ext = image_path_extension(path)
        if ext in (".nii.gz", ".nii"):
            arr = _read_nifti_array(path)
            if arr.ndim == 3:
                return random.randint(0, arr.shape[0] - 1)
        elif ext in (".mhd", ".mha"):
            arr = _read_mhd_array(path)
            if arr.ndim == 3:
                return random.randint(0, arr.shape[0] - 1)
        elif ext in (".nrrd",):
            arr = _read_nrrd_array(path)
            if arr.ndim == 3:
                return random.randint(0, arr.shape[0] - 1)
        elif ext == ".dcm":
            frames = _dicom_frames(path, max_frames=None)
            if frames:
                return random.randint(0, len(frames) - 1)
        return 0

    def _load_frame(self, entry: USManifestEntry, frame_idx: int = 0) -> np.ndarray:
        path = self._remap_path(entry.image_paths[frame_idx])
        vol_frame_idx = self._volume_slice_index(path, entry)
        return load_image(path, frame_idx=vol_frame_idx)

    def _load_clip(self, entry: USManifestEntry,
                   max_frames: Optional[int] = None) -> List[np.ndarray]:
        paths = [self._remap_path(p) for p in entry.image_paths]
        if len(paths) == 1:
            ext = image_path_extension(paths[0])
            if ext in (".avi", ".mp4", ".mov", ".mkv", ".gif", ".dcm", ".h5", ".hdf5"):
                return load_video_frames(paths[0], max_frames)
            if ext in _VOLUME_SLICE_EXTS:
                arr = (
                    _read_nifti_array(paths[0]) if ext in (".nii.gz", ".nii")
                    else _read_mhd_array(paths[0]) if ext in (".mhd", ".mha")
                    else _read_nrrd_array(paths[0])
                )
                if arr.ndim <= 2:
                    return [load_image(paths[0])]
                depth = arr.shape[0]
                if entry.frame_indices:
                    indices = [
                        i for i in entry.frame_indices
                        if 0 <= i < depth
                    ]
                elif max_frames is None or max_frames >= depth:
                    indices = list(range(depth))
                else:
                    step = max(1, depth // max_frames)
                    indices = list(range(0, depth, step))[:max_frames]
                if max_frames and len(indices) > max_frames:
                    step = max(1, len(indices) // max_frames)
                    indices = indices[::step][:max_frames]
                return [load_image(paths[0], frame_idx=i) for i in indices]
        frames = [load_image(p) for p in paths]
        if max_frames and len(frames) > max_frames:
            step   = len(frames) // max_frames
            frames = frames[::step][:max_frames]
        return frames

    def _load_mask_tensor(self, inst, frame_idx: int = 0) -> Optional[Tensor]:
        if inst.mask_path is None: return None
        mp = self._remap_path(inst.mask_path)
        if not media_path_exists(mp): return None
        return torch.from_numpy(
            load_mask(mp, frame_idx=frame_idx, mask_channel=getattr(inst, "mask_channel", None))
        ).float().unsqueeze(0)

    def __len__(self):  return len(self.entries)
    def __getitem__(self, idx): raise NotImplementedError


# ── Image SSL Dataset ─────────────────────────────────────────────────────────

class ImageSSLDataset(USFoundationDataset):
    """
    Returns per-sample dicts with native-resolution crops and padding masks.

    Keys
    ----
    global_crops    list[Tensor(3,H_i,W_i)]   n_global crops, variable size
    global_pmasks   list[Tensor(ph_i,pw_i)]   True=real patch per crop
    local_crops     list[Tensor(3,h_j,w_j)]   n_local crops
    local_pmasks    list[Tensor(ph_j,pw_j)]
    patch_mask      Tensor(ph_0,pw_0)  bool   freq-energy mask on global[0]
    dataset_id      str
    anatomy_family  str
    tier            int
    sample_id       str
    seg_mask        Tensor(1,H,W) or None
    cls_label       int or -1
    task_type       str
    is_promptable   bool
    """

    def __init__(
        self,
        entries: List[USManifestEntry],
        cfg: ImageSSLTransformConfig = ImageSSLTransformConfig(),
        root_remap: Optional[Dict] = None,
        alpha: float = 1.0,
    ):
        super().__init__(entries, root_remap)
        self.transform = ImageSSLTransform(cfg)
        self.alpha     = alpha

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        # 32 retries: resilient to partially-staged datasets where a meaningful
        # fraction of files may be missing while most entries are valid.
        for _attempt in range(32):
            try:
                return self._load_image_item(idx)
            except (FileNotFoundError, OSError) as exc:
                if _attempt == 0:
                    logging.warning(
                        "ImageSSLDataset: missing file at idx=%d: %s — resampling.", idx, exc
                    )
                idx = random.randrange(len(self))
        raise RuntimeError("ImageSSLDataset: too many consecutive missing-file errors.")

    def _load_image_item(self, idx: int) -> Dict[str, Any]:
        e = self.entries[idx]

        source_frame_idx = -1  # -1 for static images; actual index for frames sampled from video
        if e.modality_type in ("video", "pseudo_video", "volume"):
            if len(e.image_paths) == 1 and \
                    Path(self._remap_path(e.image_paths[0])).suffix.lower() \
                    in (".avi", ".mp4", ".mov", ".mkv", ".gif"):
                # For Phase 3 alignment, we need frame indices to be compatible
                # with the video stream's temporal sampling indices. Avoid
                # subsampling here; sample from the full decoded frame list.
                frames = self._load_clip(e, max_frames=None)
                source_frame_idx = torch.randint(len(frames), (1,)).item()
                img    = frames[source_frame_idx]
            else:
                source_frame_idx = torch.randint(max(1, len(e.image_paths)), (1,)).item()
                img = self._load_frame(e, source_frame_idx)
        else:
            img = self._load_frame(e, 0)

        views = self.transform(img, alpha=self.alpha)

        seg_mask, cls_label = None, -1
        _meta_frame_idx = (e.source_meta or {}).get("frame_idx", 0) or 0
        for inst in e.instances:
            if inst.mask_path:
                seg_mask = self._load_mask_tensor(inst, frame_idx=_meta_frame_idx)
                break
            if inst.classification_label is not None:
                cls_label = inst.classification_label

        return {
            "global_crops":    views["global"],        # list of (3,H_i,W_i)
            "global_pmasks":   views["global_pmask"],  # list of (ph_i,pw_i)
            "local_crops":     views["local"],         # list of (3,h_j,w_j)
            "local_pmasks":    views["local_pmask"],   # list of (ph_j,pw_j)
            "patch_mask":      views["mask"],          # (ph_0,pw_0) freq mask
            "dataset_id":      e.dataset_id,
            "anatomy_family":  e.anatomy_family,
            "tier":            e.curriculum_tier,
            "sample_id":       e.sample_id,
            "study_id":        e.study_id or "",
            "source_frame_idx": source_frame_idx,
            "seg_mask":        seg_mask,
            "cls_label":       cls_label,
            "task_type":       e.task_type,
            "is_promptable":   e.is_promptable,
        }


# ── Video SSL Dataset ─────────────────────────────────────────────────────────

class VideoSSLDataset(USFoundationDataset):
    """
    Returns per-sample dicts with native-resolution clips.

    Keys
    ----
    full_clip       Tensor(T,3,H,W)
    visible_clip    Tensor(T,3,H,W)
    tube_mask       Tensor(T,ph,pw) bool
    padding_mask    Tensor(ph,pw)   bool  True=real patch
    n_frames        int
    dataset_id, anatomy_family, tier, sample_id, task_type, fps, is_cine
    """

    def __init__(
        self,
        entries: List[USManifestEntry],
        cfg: VideoSSLTransformConfig = VideoSSLTransformConfig(),
        patch_size: int = 16,
        root_remap: Optional[Dict] = None,
        mask_ratio: Optional[float] = None,
    ):
        super().__init__(entries, root_remap)
        self.transform  = VideoSSLTransform(cfg, patch_size)
        self.mask_ratio = mask_ratio

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for _attempt in range(32):
            try:
                return self._load_video_item(idx)
            except (FileNotFoundError, OSError) as exc:
                if _attempt == 0:
                    logging.warning(
                        "VideoSSLDataset: missing file at idx=%d: %s — resampling.", idx, exc
                    )
                idx = random.randrange(len(self))
        raise RuntimeError("VideoSSLDataset: too many consecutive missing-file errors.")

    def _load_video_item(self, idx: int) -> Dict[str, Any]:
        e      = self.entries[idx]
        frames = self._load_clip(e)
        views  = self.transform(frames, mask_ratio=self.mask_ratio)

        return {
            "full_clip":             views["full"],
            "visible_clip":          views["visible"],
            "tube_mask":             views["tube_mask"],
            "padding_mask":          views["padding_mask"],
            "dataset_id":            e.dataset_id,
            "anatomy_family":        e.anatomy_family,
            "tier":                  e.curriculum_tier,
            "sample_id":             e.sample_id,
            "study_id":              e.study_id or "",
            "source_frame_indices":  views["sampled_frame_indices"],  # List[int] into loaded frames
            "n_frames":              views["full"].shape[0],
            "task_type":             e.task_type,
            "fps":                   e.fps or 25.0,
            "is_cine":               e.is_cine,
        }


# ── Paired SSL Dataset (Phase 3) ──────────────────────────────────────────────

class PairedSSLDataset(USFoundationDataset):
    """
    Phase 3 paired stream: each __getitem__ returns BOTH an image view
    and a video view from the *same* clip entry (ssl_stream='both').

    Because both views originate from the same source, the image frame is
    guaranteed to be one of the temporally-sampled video frames, giving an
    exact frame_offset=t with weight=1.0 for every sample in the batch.

    Return dict keys
    ----------------
    image        : dict  — compatible with ImageSSLCollator
    video        : dict  — compatible with VideoSSLCollator
    frame_offset : int   — temporal slot index in the video clip that the
                           image frame was drawn from
    """

    def __init__(
        self,
        entries: List[USManifestEntry],
        img_cfg: ImageSSLTransformConfig = ImageSSLTransformConfig(),
        vid_cfg: VideoSSLTransformConfig = VideoSSLTransformConfig(),
        patch_size: int = 16,
        root_remap: Optional[Dict] = None,
        img_alpha: float = 1.0,
        vid_mask_ratio: Optional[float] = None,
    ):
        super().__init__(entries, root_remap)
        self.img_transform  = ImageSSLTransform(img_cfg)
        self.vid_transform  = VideoSSLTransform(vid_cfg, patch_size)
        self.img_alpha      = img_alpha
        self.vid_mask_ratio = vid_mask_ratio

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        for _attempt in range(32):
            try:
                return self._load_paired_item(idx)
            except (FileNotFoundError, OSError) as exc:
                if _attempt == 0:
                    logging.warning(
                        "PairedSSLDataset: missing file at idx=%d: %s — resampling.", idx, exc
                    )
                idx = random.randrange(len(self))
        raise RuntimeError("PairedSSLDataset: too many consecutive missing-file errors.")

    def _load_paired_item(self, idx: int) -> Dict[str, Any]:
        e = self.entries[idx]

        # Load the full clip once for both modalities.
        frames = self._load_clip(e, max_frames=None)

        # ── Video view ────────────────────────────────────────────────────────
        vid_views = self.vid_transform(frames, mask_ratio=self.vid_mask_ratio)
        sampled_indices: List[int] = vid_views["sampled_frame_indices"]
        T = vid_views["full"].shape[0]

        # ── Image view: draw from one of the video's temporal slots ──────────
        # This guarantees exact frame overlap for the alignment pair.
        t = int(torch.randint(T, (1,)).item())
        source_frame_idx = sampled_indices[t]
        img = frames[source_frame_idx]
        img_views = self.img_transform(img, alpha=self.img_alpha)

        return {
            "image": {
                "global_crops":    img_views["global"],
                "global_pmasks":   img_views["global_pmask"],
                "local_crops":     img_views["local"],
                "local_pmasks":    img_views["local_pmask"],
                "patch_mask":      img_views["mask"],
                "dataset_id":      e.dataset_id,
                "anatomy_family":  e.anatomy_family,
                "tier":            e.curriculum_tier,
                "sample_id":       e.sample_id,
                "study_id":        e.study_id or "",
                "source_frame_idx": source_frame_idx,
                "seg_mask":        None,
                "cls_label":       -1,
                "task_type":       e.task_type,
                "is_promptable":   e.is_promptable,
            },
            "video": {
                "full_clip":            vid_views["full"],
                "visible_clip":         vid_views["visible"],
                "tube_mask":            vid_views["tube_mask"],
                "padding_mask":         vid_views["padding_mask"],
                "n_frames":             T,
                "dataset_id":           e.dataset_id,
                "anatomy_family":       e.anatomy_family,
                "tier":                 e.curriculum_tier,
                "sample_id":            e.sample_id,
                "study_id":             e.study_id or "",
                "source_frame_indices": sampled_indices,
                "fps":                  e.fps or 25.0,
                "is_cine":              e.is_cine,
                "task_type":            e.task_type,
            },
            "frame_offset": t,
        }
