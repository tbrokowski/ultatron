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

def _read_nifti_array(path: str) -> np.ndarray:
    """Read a NIfTI1 (.nii / .nii.gz) file and return the raw voxel array.

    Returns an ndarray with shape transposed to (T-or-Z, Y, X) for 3-D volumes
    or (Y, X) for 2-D images (NIfTI stores data as X-fastest, so we transpose).
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
    arr = np.frombuffer(raw[vox_offset:], dtype=dtype).reshape(shape)
    return arr.T  # (X, Y[, T]) → (T, Y, X) or (Y, X)


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
    n_channels = int(header.get("ElementNumberOfChannels", "1"))
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
        raw_start = idx + len(sentinel)
        if content[raw_start:raw_start + 2] == b"\r\n":
            raw_start += 2
        elif content[raw_start:raw_start + 1] in (b"\n", b"\r"):
            raw_start += 1
        raw = content[raw_start:]
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

    # DimSize is (nx, ny[, nz, ...]) — reshape scalar images in reverse for
    # C-order (nz, ny, nx). Vector 2-D MetaImages store channels as the
    # fastest-varying component, so expose them as (ny, nx, C).
    if n_channels > 1 and ndims == 2:
        shape = tuple(reversed(dim_sizes)) + (n_channels,)
    else:
        shape = tuple(reversed(dim_sizes))
    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return arr


# ── Image / video loading ─────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
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
    suffixes = Path(path).suffixes
    # Compound extensions like .nii.gz → treat as ".nii.gz"
    ext = "".join(suffixes[-2:]).lower() if len(suffixes) >= 2 else (suffixes[-1].lower() if suffixes else "")

    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)

    if ext in (".npy",):
        arr = np.load(path)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[..., :3]   # drop alpha, keep RGB
        return arr.astype(np.uint8) if arr.max() > 1 else (arr * 255).astype(np.uint8)

    if ext in (".npz",):
        d   = np.load(path)
        arr = d[list(d.keys())[0]]
        return arr.astype(np.uint8)

    if ext in (".mhd", ".mha"):
        arr = _read_mhd_array(path)
        if arr.ndim == 3:
            if arr.shape[-1] in (3, 4):
                arr = arr[..., :3]
            elif arr.shape[0] in (3, 4):
                arr = np.moveaxis(arr[:3], 0, -1)
            else:
                arr = arr[0]
        if np.issubdtype(arr.dtype, np.integer) and arr.min() >= 0 and arr.max() <= 255:
            img = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.float32)
            img = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return img

    if ext in (".dcm",):
        try:
            import pydicom
            ds  = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(float)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            return arr   # (H, W) for monochrome DICOM; to_canonical_tensor expands to RGB
        except ImportError:
            raise RuntimeError("pydicom required for DICOM files")

    if ext in (".nii.gz", ".nii"):
        arr = _read_nifti_array(path)
        if arr.ndim == 3:
            arr = arr[0]
        arr = arr.astype(np.float32)
        img = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
        return img

    raise ValueError(f"Unsupported image format: {ext}")


def load_video_frames(path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """
    Load video frames and return a list of (H, W, 3) uint8 RGB numpy arrays.

    Colour is preserved so that Doppler and colour-overlay clips are not
    inadvertently collapsed to grayscale.  ``to_canonical_tensor`` in
    transforms.py converts each frame to a (3, H, W) float32 tensor.
    """
    ext = Path(path).suffix.lower()

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


def load_mask(path: str, frame_idx: int = 0) -> np.ndarray:
    suffixes = Path(path).suffixes
    ext = "".join(suffixes[-2:]).lower() if len(suffixes) >= 2 else (suffixes[-1].lower() if suffixes else "")
    if ext in (".nii.gz", ".nii"):
        arr = _read_nifti_array(path)
        if arr.ndim == 3:
            arr = arr[min(frame_idx, arr.shape[0] - 1)]
        return (arr > 0).astype(np.uint8)
    if ext in (".mhd", ".mha"):
        arr = _read_mhd_array(path)
        if arr.ndim == 3:
            arr = arr[min(frame_idx, arr.shape[0] - 1)]
        return (arr > 0).astype(np.uint8)
    from PIL import Image
    mask = np.array(Image.open(path).convert("L"), dtype=np.uint8)
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

    def _load_frame(self, entry: USManifestEntry, frame_idx: int = 0) -> np.ndarray:
        return load_image(self._remap_path(entry.image_paths[frame_idx]))

    def _load_clip(self, entry: USManifestEntry,
                   max_frames: Optional[int] = None) -> List[np.ndarray]:
        paths = [self._remap_path(p) for p in entry.image_paths]
        if len(paths) == 1 and Path(paths[0]).suffix.lower() in \
                (".avi", ".mp4", ".mov", ".mkv", ".gif"):
            return load_video_frames(paths[0], max_frames)
        frames = [load_image(p) for p in paths]
        if max_frames and len(frames) > max_frames:
            step   = len(frames) // max_frames
            frames = frames[::step][:max_frames]
        return frames

    def _load_mask_tensor(self, inst, frame_idx: int = 0) -> Optional[Tensor]:
        if inst.mask_path is None: return None
        mp = self._remap_path(inst.mask_path)
        if not Path(mp).exists(): return None
        return torch.from_numpy(load_mask(mp, frame_idx=frame_idx)).float().unsqueeze(0)

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
        for _attempt in range(8):
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
        for _attempt in range(8):
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
        for _attempt in range(8):
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
