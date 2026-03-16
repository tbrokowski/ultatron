"""
dataset.py  ·  Unified Ultatron dataset classes
====================================================
This file contains the dataset classes for the Ultatron foundation model.

"""
from __future__ import annotations

import io
import os
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
    ext = Path(path).suffix.lower()

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
        try:
            import SimpleITK as sitk
            img = sitk.GetArrayFromImage(sitk.ReadImage(path))
            if img.ndim == 3: img = img[0]
            img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)
            return img
        except ImportError:
            raise RuntimeError("SimpleITK required for .mhd files")

    if ext in (".dcm",):
        try:
            import pydicom
            ds  = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(float)
            arr = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255).astype(np.uint8)
            return arr   # (H, W) for monochrome DICOM; to_canonical_tensor expands to RGB
        except ImportError:
            raise RuntimeError("pydicom required for DICOM files")

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


def load_mask(path: str) -> np.ndarray:
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

    def _load_mask_tensor(self, inst) -> Optional[Tensor]:
        if inst.mask_path is None: return None
        mp = self._remap_path(inst.mask_path)
        if not Path(mp).exists(): return None
        return torch.from_numpy(load_mask(mp)).float().unsqueeze(0)

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
        e = self.entries[idx]

        if e.modality_type in ("video", "pseudo_video", "volume"):
            if len(e.image_paths) == 1 and \
                    Path(self._remap_path(e.image_paths[0])).suffix.lower() \
                    in (".avi", ".mp4", ".mov", ".mkv", ".gif"):
                frames = self._load_clip(e, max_frames=64)
                img    = frames[torch.randint(len(frames), (1,)).item()]
            else:
                img = self._load_frame(e, torch.randint(max(1, len(e.image_paths)), (1,)).item())
        else:
            img = self._load_frame(e, 0)

        views = self.transform(img, alpha=self.alpha)

        seg_mask, cls_label = None, -1
        for inst in e.instances:
            if inst.mask_path:
                seg_mask = self._load_mask_tensor(inst)
                break
            if inst.classification_label is not None:
                cls_label = inst.classification_label

        return {
            "global_crops":   views["global"],        # list of (3,H_i,W_i)
            "global_pmasks":  views["global_pmask"],  # list of (ph_i,pw_i)
            "local_crops":    views["local"],         # list of (3,h_j,w_j)
            "local_pmasks":   views["local_pmask"],   # list of (ph_j,pw_j)
            "patch_mask":     views["mask"],          # (ph_0,pw_0) freq mask
            "dataset_id":     e.dataset_id,
            "anatomy_family": e.anatomy_family,
            "tier":           e.curriculum_tier,
            "sample_id":      e.sample_id,
            "seg_mask":       seg_mask,
            "cls_label":      cls_label,
            "task_type":      e.task_type,
            "is_promptable":  e.is_promptable,
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
        e      = self.entries[idx]
        frames = self._load_clip(e)
        views  = self.transform(frames, mask_ratio=self.mask_ratio)

        return {
            "full_clip":     views["full"],
            "visible_clip":  views["visible"],
            "tube_mask":     views["tube_mask"],
            "padding_mask":  views["padding_mask"],
            "dataset_id":    e.dataset_id,
            "anatomy_family":e.anatomy_family,
            "tier":          e.curriculum_tier,
            "sample_id":     e.sample_id,
            "n_frames":      views["full"].shape[0],
            "task_type":     e.task_type,
            "fps":           e.fps or 25.0,
            "is_cine":       e.is_cine,
        }
