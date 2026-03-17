"""
transforms.py  ·  Ultrasound-specific augmentations
====================================================

Native-resolution multi-crop
-----------------------------
Images are extracted at native pixel scale and padded to the nearest
patch_size multiple. 

Channel handling  
---------------------------
All images and video frames are converted to 3-channel RGB tensors at
ingestion by to_canonical_tensor().  

Ultrasound images may be:
  - Greyscale  (B-mode, CAMUS, most echo datasets) → R=G=B=grey
  - RGB        (Doppler colour flow, colour overlays, DICOM with colourmap)
               → passed through unchanged

Masking strategy  
-----------------------
ImageSSLTransformConfig and VideoSSLTransformConfig have a mask_strategy
field controlling how the student's masked view is produced:

  "freq"     Frequency-domain band masking (physics-motivated for ultrasound).
             The student sees a spectrally degraded image with mid/high-freq
             bands zeroed in Fourier space.  The spatial_mask is derived from
             which patches lost the most spectral energy.
             Default for both image and video branches.

  "spatial"  Random spatial patch masking (baseline approach).
             Random patches are zeroed in pixel space.  The spatial_mask
             is the set of zeroed patches.

  "both"     Frequency masking first, then spatial masking on top.
             Step 1: Apply frequency band masking → degraded image.
             Step 2: Additionally zero out a subset of spatial patches.
             The spatial_mask is the union of freq-energy patches and the
             additionally-zeroed patches.
             Harder pretext task: student must predict both spectrally
             degraded AND spatially missing regions.

All three strategies produce the same output dict shape 
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch import Tensor


# ── Constants ─────────────────────────────────────────────────────────────────

MASK_STRATEGY_FREQ    = "freq"
MASK_STRATEGY_SPATIAL = "spatial"
MASK_STRATEGY_BOTH    = "both"
_VALID_STRATEGIES     = (MASK_STRATEGY_FREQ, MASK_STRATEGY_SPATIAL, MASK_STRATEGY_BOTH)


# ── Config dataclasses ────────────────────────────────────────────────────────

@dataclass
class FreqMaskConfig:
    r_inner_min: float = 0.10
    r_outer_max: float = 0.95
    band_width_min: float = 0.15
    band_width_max: float = 0.40
    mask_ratio: float = 0.40
    use_alp_bias: bool = True
    perturb_phase: bool = False
    phase_perturb_sigma: float = 0.10
    n_bands: int = 1


@dataclass
class ImageSSLTransformConfig:
    # Multi-crop scales
    n_global_crops: int = 2
    global_crop_scale: Tuple[float, float] = (0.4, 1.0)
    n_local_crops: int = 6
    local_crop_scale: Tuple[float, float] = (0.05, 0.4)

    patch_size: int = 16
    mask_ratio: float = 0.40

    # Native-resolution constraints
    max_global_crop_px: int = 512
    min_crop_px: int = 32

    # Photometric
    brightness: float = 0.4
    contrast: float = 0.4
    apply_speckle: bool = True
    speckle_sigma: float = 0.1
    apply_blur: bool = True
    blur_sigma: Tuple[float, float] = (0.1, 2.0)
    apply_solarize: bool = True
    solarize_thresh: float = 0.5

    # Geometry
    allow_hflip: bool = False
    allow_vflip: bool = False
    max_rotation_deg: float = 15.0

    # ── Masking strategy ──────────────────────────────────────────────────────
    # "freq"    : frequency-domain band masking (default)
    # "spatial" : random spatial patch masking
    # "both"    : frequency masking then spatial masking on top
    mask_strategy: str = MASK_STRATEGY_FREQ

    # Freq masking config (used when mask_strategy in {"freq", "both"})
    freq_mask: FreqMaskConfig = field(default_factory=FreqMaskConfig)

    # Spatial mask ratio used when mask_strategy in {"spatial", "both"}.
    # In "both" mode this is the *additional* spatial mask ratio applied on
    # top of the frequency mask.  The union of both masks is the final mask.
    spatial_mask_ratio: float = 0.40

    def __post_init__(self):
        if self.mask_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"mask_strategy must be one of {_VALID_STRATEGIES}, "
                f"got {self.mask_strategy!r}"
            )


@dataclass
class VideoSSLTransformConfig:
    n_frames: int = 16
    max_n_frames: int = 64
    temporal_stride: int = 2

    tube_mask_ratio: float = 0.75
    tube_size: int = 2
    min_mask_ratio: float = 0.4
    max_mask_ratio: float = 0.9

    crop_scale: Tuple[float, float] = (0.5, 1.0)
    max_crop_px: int = 512
    min_crop_px: int = 64

    allow_hflip: bool = False
    apply_speckle: bool = True
    speckle_sigma: float = 0.08

    # ── Masking strategy ──────────────────────────────────────────────────────
    # "freq"    : frequency-domain band masking per temporal group (tube_size frames)
    # "spatial" : spatial block masking per temporal group
    # "both"    : freq + spatial applied independently, union tube mask (default)
    mask_strategy: str = MASK_STRATEGY_BOTH

    freq_mask: FreqMaskConfig = field(default_factory=lambda: FreqMaskConfig(
        mask_ratio=0.75,
    ))

    # Spatial block mask ratio for the spatial component in "spatial" and "both" modes.
    # In "both" mode this is INDEPENDENT of tube_mask_ratio — the spatial component
    # samples its own block regions; the union of both becomes the final tube_mask.
    spatial_mask_ratio: float = 0.40

    def __post_init__(self):
        if self.mask_strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"mask_strategy must be one of {_VALID_STRATEGIES}, "
                f"got {self.mask_strategy!r}"
            )


# ── Channel utilities (used by transforms AND imported by backbone files) ──────

def to_canonical_tensor(img) -> Tensor:
    """
    Convert any image input to a float32 RGB tensor (3, H, W) in [0, 1].

    All images are forced to 3-channel RGB at this stage 

    Channel rules
    -------------
    Greyscale (1-channel or 2D)  → channel-repeat to RGB  (R=G=B=grey)
    RGB (3-channel)              → pass through unchanged
    RGBA (4-channel)             → alpha channel dropped, first 3 kept

    Returns
    -------
    Tensor  (3, H, W)  float32  [0, 1]
    """
    # ── Tensor input ──────────────────────────────────────────────────────────
    if isinstance(img, Tensor):
        if img.ndim == 2:
            # (H, W) greyscale → (3, H, W)
            t = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()
            return t.unsqueeze(0).expand(3, -1, -1).contiguous()
        if img.ndim == 3:
            t = img.float() / 255.0 if img.dtype == torch.uint8 else img.float()
            # (C, H, W) layout
            if t.shape[0] == 1:
                return t.expand(3, -1, -1).contiguous()   # grey → RGB
            if t.shape[0] == 3:
                return t                                    # already RGB
            if t.shape[0] == 4:
                return t[:3].contiguous()                  # drop alpha
            # (H, W, C) layout — permute first
            if t.shape[2] in (1, 3, 4):
                t = t.permute(2, 0, 1)
                if t.shape[0] == 1:
                    return t.expand(3, -1, -1).contiguous()
                return t[:3].contiguous()
        return img.float()

    # ── PIL Image ─────────────────────────────────────────────────────────────
    if hasattr(img, "mode"):
        # Convert everything to RGB via PIL — handles all modes (L, P, CMYK, etc.)
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()   # (3, H, W)

    # ── numpy array ───────────────────────────────────────────────────────────
    arr = np.asarray(img)
    if arr.ndim == 2:
        # (H, W) greyscale
        t = torch.from_numpy(arr.astype(np.float32) / 255.0)
        return t.unsqueeze(0).expand(3, -1, -1).contiguous()
    if arr.ndim == 3:
        if arr.shape[2] in (1, 3, 4):
            # (H, W, C) → (C, H, W)
            t = torch.from_numpy(arr[:, :, :3].astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).contiguous()    # (3, H, W) or (1, H, W)
            if t.shape[0] == 1:
                return t.expand(3, -1, -1).contiguous()
            return t
        if arr.shape[0] in (1, 3, 4):
            # (C, H, W) layout
            t = torch.from_numpy(arr.astype(np.float32) / 255.0)
            if t.shape[0] == 1:
                return t.expand(3, -1, -1).contiguous()
            return t[:3].contiguous()

    raise ValueError(f"Cannot convert to RGB tensor: shape={np.asarray(img).shape}")


def ensure_rgb(x: Tensor) -> Tensor:
    """
    Ensure a (C, H, W) or (B, C, H, W) or (B, T, C, H, W) tensor has C=3.

    If C=1 (greyscale) → channel-repeat to 3 (no learnable params, no copy).
    If C=3 (RGB)       → pass through unchanged.
    If C=4             → drop alpha channel, return first 3.

    This is a pure tensor utility.  Backbone files import it from here
    so the grey→RGB adaptation lives in one place.
    """
    # Determine which axis is the channel dimension
    # Support (C,H,W), (B,C,H,W), (B,T,C,H,W)
    ndim = x.ndim
    if ndim == 3:
        c_dim = 0
    elif ndim == 4:
        c_dim = 1
    elif ndim == 5:
        c_dim = 2
    else:
        raise ValueError(f"ensure_rgb: unexpected tensor ndim={ndim}")

    c = x.shape[c_dim]
    if c == 3:
        return x                            # already RGB
    if c == 1:
        return x.expand(*[-1 if i != c_dim else 3
                           for i in range(ndim)]).contiguous()
    if c == 4:
        # Drop alpha
        idx = [slice(None)] * ndim
        idx[c_dim] = slice(0, 3)
        return x[tuple(idx)].contiguous()
    raise ValueError(f"ensure_rgb: unexpected channel count C={c}")


def add_speckle_noise(x: Tensor, sigma: float = 0.1) -> Tensor:
    """Multiplicative Gaussian noise  y = x * (1 + σ·N(0,1))."""
    return torch.clamp(x * (1 + torch.randn_like(x) * sigma), 0.0, 1.0)


def pad_to_patch_multiple(x: Tensor, patch_size: int) -> Tuple[Tensor, Tensor]:
    """
    Pad a (C, H, W) crop so H and W are multiples of patch_size.
    Padding is zero, applied on right and bottom only.

    Returns
    -------
    padded   : (C, H_pad, W_pad)
    tok_mask : (ph, pw) bool  True=real patch
    """
    C, H, W = x.shape
    pH = math.ceil(H / patch_size) * patch_size
    pW = math.ceil(W / patch_size) * patch_size
    ph, pw = pH // patch_size, pW // patch_size

    if pH == H and pW == W:
        return x, torch.ones(ph, pw, dtype=torch.bool)

    padded   = F.pad(x, (0, pW - W, 0, pH - H), value=0.0)
    tok_mask = torch.zeros(ph, pw, dtype=torch.bool)
    real_ph  = math.ceil(H / patch_size)
    real_pw  = math.ceil(W / patch_size)
    tok_mask[:real_ph, :real_pw] = True
    return padded, tok_mask


# ── Native-resolution crop extraction ────────────────────────────────────────

def _native_crop(
    x: Tensor,                       # (C, H, W)
    scale: Tuple[float, float],
    patch_size: int,
    min_px: int,
    max_px: int,
    aspect_range: Tuple[float, float] = (3/4, 4/3),
    n_tries: int = 10,
) -> Tensor:
    """
    Extract a random-scale, random-position crop at native pixel resolution.
    No resize.  Snapped to patch_size multiples, clamped to [min_px, max_px].
    Channel dimension C is preserved.
    """
    C, H, W = x.shape
    area    = H * W

    for _ in range(n_tries):
        target_area = random.uniform(*scale) * area
        aspect      = random.uniform(*aspect_range)
        ch = int(round(math.sqrt(target_area * aspect)))
        cw = int(round(math.sqrt(target_area / aspect)))
        ch = max(patch_size, min(max_px, round(ch / patch_size) * patch_size))
        cw = max(patch_size, min(max_px, round(cw / patch_size) * patch_size))
        if ch <= H and cw <= W and ch >= min_px and cw >= min_px:
            top  = random.randint(0, H - ch)
            left = random.randint(0, W - cw)
            return x[:, top:top + ch, left:left + cw]

    # Fallback: whole image, snapped and capped
    ch = max(patch_size, min(max_px, round(H / patch_size) * patch_size))
    cw = max(patch_size, min(max_px, round(W / patch_size) * patch_size))
    return x[:, :ch, :cw]


# ── Frequency-domain masking ──────────────────────────────────────────────────

def _make_radial_grid(H: int, W: int) -> Tensor:
    fy = torch.roll(torch.fft.fftfreq(H), H // 2)
    fx = torch.roll(torch.fft.fftfreq(W), W // 2)
    Fy, Fx = torch.meshgrid(fy, fx, indexing="ij")
    return (torch.sqrt(Fy**2 + Fx**2) / math.sqrt(0.5**2 + 0.5**2)).clamp(0.0, 1.0)


def _sample_band_centre(
    cfg: FreqMaskConfig,
    half_width: float,
    magnitude: Optional[Tensor],
    alp_weights: Optional[Tensor],
) -> float:
    lo = cfg.r_inner_min + half_width
    hi = cfg.r_outer_max - half_width
    if lo >= hi:
        return (lo + hi) / 2.0
    if not cfg.use_alp_bias or alp_weights is None or magnitude is None:
        return random.uniform(lo, hi)
    H, W   = magnitude.shape
    weight = magnitude.float().clamp(min=1e-6) * alp_weights.float().clamp(min=0.0)
    n_bins = 64
    radii  = _make_radial_grid(H, W)
    bins   = ((radii - lo) / (hi - lo) * n_bins).long().clamp(0, n_bins - 1)
    hist   = torch.zeros(n_bins)
    hist.scatter_add_(0, bins.flatten(), weight.flatten())
    hist   = hist.clamp(min=1e-8) / hist.sum()
    bin_idx = int(torch.multinomial(hist, 1).item())
    return float(lo + (bin_idx + 0.5) / n_bins * (hi - lo))


def _freq_mask_single_channel(
    x2d: Tensor,          # (H, W) single channel
    cfg: FreqMaskConfig,
    patch_size: int,
    alp_full: Optional[Tensor],
    mask_ratio: float,
) -> Tuple[Tensor, Tensor]:
    """
    Core freq masking on a single 2D slice.
    Returns (masked_2d, patch_energy_loss) where patch_energy_loss is (ph, pw).
    """
    H, W = x2d.shape
    ph, pw = H // patch_size, W // patch_size

    X_fft   = torch.fft.fft2(x2d)
    X_shift = torch.fft.fftshift(X_fft)
    magnitude = X_shift.abs()

    radii     = _make_radial_grid(H, W)
    band_mask = torch.zeros(H, W, dtype=torch.bool)
    for _ in range(cfg.n_bands):
        half_w = random.uniform(cfg.band_width_min, cfg.band_width_max) / 2.0
        r_c    = _sample_band_centre(cfg, half_w, magnitude, alp_full)
        ring   = (radii >= (r_c - half_w)) & (radii <= (r_c + half_w))
        ring   = ring & (radii >= cfg.r_inner_min)
        band_mask = band_mask | ring

    X_masked = X_shift.clone()
    X_masked[band_mask] = 0.0
    if cfg.perturb_phase:
        phasor   = torch.polar(torch.ones(H, W), torch.randn(H, W) * cfg.phase_perturb_sigma)
        X_masked = X_masked * phasor

    masked_2d = torch.fft.ifft2(torch.fft.ifftshift(X_masked)).real.clamp(0.0, 1.0)

    energy_removed = (magnitude**2) * band_mask.float()
    patch_energy   = (
        energy_removed
        .unfold(0, patch_size, patch_size)
        .unfold(1, patch_size, patch_size)
        .sum(dim=(-1, -2))
    )   # (ph, pw)

    return masked_2d, patch_energy


def freq_mask_image(
    x: Tensor,                         # (C, H, W)
    cfg: FreqMaskConfig,
    patch_size: int = 16,
    alp: Optional[Tensor] = None,      # (ph, pw)
    mask_ratio_override: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Frequency-domain masking for a (C, H, W) image.
    Operates on each channel independently; the spatial mask is derived from
    mean energy loss across channels.

    Returns
    -------
    masked_img   : (C, H, W)
    spatial_mask : (ph, pw) bool
    """
    C, H, W    = x.shape
    ph, pw     = H // patch_size, W // patch_size
    mask_ratio = mask_ratio_override if mask_ratio_override is not None else cfg.mask_ratio

    # Upsample ALP to image resolution once, shared across channels
    alp_full: Optional[Tensor] = None
    if alp is not None and cfg.use_alp_bias:
        alp_4d   = alp.float().unsqueeze(0).unsqueeze(0)
        alp_full = F.interpolate(
            alp_4d, size=(H, W), mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)

    masked_channels = []
    total_energy    = torch.zeros(ph, pw)

    for c in range(C):
        masked_ch, patch_energy = _freq_mask_single_channel(
            x[c], cfg, patch_size, alp_full, mask_ratio
        )
        masked_channels.append(masked_ch)
        total_energy += patch_energy

    masked_img = torch.stack(masked_channels, dim=0)   # (C, H, W)

    # Derive spatial mask from mean energy loss across channels
    total_energy = total_energy / C
    if total_energy.max() > 0:
        total_energy = total_energy / total_energy.max()
    n_mask    = max(1, int(ph * pw * mask_ratio))
    threshold = total_energy.flatten().topk(n_mask).values[-1]
    spatial_mask = total_energy >= threshold

    return masked_img, spatial_mask


def freq_mask_video(
    clip: Tensor,                      # (T, C, H, W)
    cfg: FreqMaskConfig,
    patch_size: int = 16,
    mask_ratio_override: Optional[float] = None,
    tube_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Frequency-domain tube masking for a (T, C, H, W) clip.

    tube_size controls temporal grouping for band selection:

      None or >= T  A single frequency band is chosen from a central reference
                    frame and applied to all T frames uniformly (same spatial
                    mask across the entire clip).

      < T           Frames are divided into temporal groups of tube_size.
                    Each group selects its own frequency band independently
                    from its representative frame, yielding per-group spatial
                    masks with temporal diversity between groups.  This matches
                    the spatiotemporal tubelet structure used in V-JEPA.

    Returns
    -------
    visible_clip : (T, C, H, W)  spectrally degraded clip
    tube_mask    : (T, ph, pw)   bool  True = high-energy-loss patch
    """
    T, C, H, W = clip.shape
    ph, pw     = H // patch_size, W // patch_size
    mask_ratio = mask_ratio_override if mask_ratio_override is not None else cfg.mask_ratio
    radii      = _make_radial_grid(H, W)

    effective_tube = tube_size if (tube_size is not None and tube_size < T) else T
    n_groups       = math.ceil(T / effective_tube)

    visible_frames: List[Tensor] = []
    spatial_per_frame: List[Tensor] = []

    for g in range(n_groups):
        t_start = g * effective_tube
        t_end   = min(T, t_start + effective_tube)
        n_t     = t_end - t_start

        # Representative frame for band selection (middle of this group)
        ref_t  = t_start + (n_t - 1) // 2
        X_ref  = torch.fft.fftshift(torch.fft.fft2(clip[ref_t, 0]))

        band_mask = torch.zeros(H, W, dtype=torch.bool)
        for _ in range(cfg.n_bands):
            half_w = random.uniform(cfg.band_width_min, cfg.band_width_max) / 2.0
            r_c    = _sample_band_centre(cfg, half_w, X_ref.abs(), alp_weights=None)
            ring   = (radii >= (r_c - half_w)) & (radii <= (r_c + half_w))
            ring   = ring & (radii >= cfg.r_inner_min)
            band_mask = band_mask | ring

        group_energy = torch.zeros(ph, pw)

        for t in range(t_start, t_end):
            masked_chs = []
            for c in range(C):
                X_shift = torch.fft.fftshift(torch.fft.fft2(clip[t, c])).clone()
                group_energy += (
                    X_shift.abs()**2 * band_mask.float()
                ).unfold(0, patch_size, patch_size
                ).unfold(1, patch_size, patch_size
                ).sum(dim=(-1, -2))
                X_shift[band_mask] = 0.0
                if cfg.perturb_phase:
                    phasor  = torch.polar(
                        torch.ones(H, W),
                        torch.randn(H, W) * cfg.phase_perturb_sigma,
                    )
                    X_shift = X_shift * phasor
                masked_chs.append(
                    torch.fft.ifft2(torch.fft.ifftshift(X_shift)).real.clamp(0.0, 1.0)
                )
            visible_frames.append(torch.stack(masked_chs, dim=0))   # (C, H, W)

        # Derive spatial mask from mean energy loss for this group
        group_energy = group_energy / (n_t * C)
        if group_energy.max() > 0:
            group_energy = group_energy / group_energy.max()
        n_mask    = max(1, int(ph * pw * mask_ratio))
        threshold = group_energy.flatten().topk(n_mask).values[-1]
        spatial   = group_energy >= threshold   # (ph, pw)
        spatial_per_frame.extend([spatial] * n_t)

    visible_clip = torch.stack(visible_frames, dim=0)           # (T, C, H, W)
    tube_mask    = torch.stack(spatial_per_frame, dim=0)        # (T, ph, pw)
    return visible_clip, tube_mask


def freq_mask_image_alp(
    x: Tensor,
    cfg: FreqMaskConfig,
    patch_size: int,
    saliency: Optional[Tensor],
    hardness: Optional[Tensor],
    alpha: float,
    mask_ratio_override: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    alp: Optional[Tensor] = None
    if saliency is not None or hardness is not None:
        s    = saliency if saliency is not None else torch.zeros_like(hardness)
        h    = hardness if hardness is not None else torch.zeros_like(saliency)
        raw  = alpha * s + (1.0 - alpha) * h
        flat = raw.flatten() - raw.max()
        flat = torch.exp(flat)
        alp  = (flat / (flat.sum() + 1e-8)).reshape(raw.shape)
    return freq_mask_image(x, cfg, patch_size=patch_size, alp=alp,
                           mask_ratio_override=mask_ratio_override)


# ── Spatial masking (for "spatial" and "both" strategies) ─────────────────────

def spatial_patch_mask(
    x: Tensor,              # (C, H, W)
    patch_size: int,
    mask_ratio: float,
    block_style: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Zero out randomly selected patches in pixel space.

    Parameters
    ----------
    x           : (C, H, W)
    mask_ratio  : fraction of patches to mask
    block_style : if True, masks a contiguous rectangular block (tube-style);
                  if False, masks uniformly random patches (MAE-style).

    Returns
    -------
    masked_x     : (C, H, W)  with zeroed patches
    spatial_mask : (ph, pw) bool  True = masked patch
    """
    C, H, W = x.shape
    ph, pw  = H // patch_size, W // patch_size
    n_mask  = max(1, int(ph * pw * mask_ratio))

    if block_style:
        mask_2d = _random_block_mask(ph, pw, n_mask)
    else:
        idx    = torch.randperm(ph * pw)[:n_mask]
        mask_1d = torch.zeros(ph * pw, dtype=torch.bool)
        mask_1d[idx] = True
        mask_2d = mask_1d.reshape(ph, pw)

    # Zero out masked patch windows in pixel space
    masked_x = x.clone()
    for pi in range(ph):
        for pj in range(pw):
            if mask_2d[pi, pj]:
                r0 = pi * patch_size
                c0 = pj * patch_size
                masked_x[:, r0:r0 + patch_size, c0:c0 + patch_size] = 0.0

    return masked_x, mask_2d


def spatial_tube_mask(
    clip: Tensor,                  # (T, C, H, W)
    patch_size: int,
    mask_ratio: float,
    tube_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Spatiotemporal tube masking for a (T, C, H, W) clip.

    tube_size controls the temporal granularity of masking:

      None or >= T  Classic V-JEPA tube: a single spatial block is sampled once
                    and applied identically across all T frames.  This is the
                    most aggressive temporal consistency but least diverse.

      < T           V-JEPA spatiotemporal tubelet masking: frames are divided
                    into consecutive groups of tube_size frames.  Each group
                    gets its own independently sampled spatial block, giving
                    temporal diversity across groups while maintaining spatial
                    consistency within each tube_size window.  This is the
                    recommended setting (tube_size=2 in VideoSSLTransformConfig).

    Returns
    -------
    visible_clip : (T, C, H, W)  pixels zeroed at masked patch positions
    tube_mask    : (T, ph, pw)   bool  True = masked
    """
    T, C, H, W = clip.shape
    ph, pw     = H // patch_size, W // patch_size
    n_mask     = max(1, int(ph * pw * mask_ratio))

    if tube_size is None or tube_size >= T:
        # Classic all-frames tube: one spatial block for all T frames
        mask_2d   = _random_block_mask(ph, pw, n_mask)
        tube_mask = mask_2d.unsqueeze(0).expand(T, -1, -1).contiguous()
    else:
        # Spatiotemporal tubelets: independent spatial block per temporal group
        n_groups  = math.ceil(T / tube_size)
        tube_mask = torch.zeros(T, ph, pw, dtype=torch.bool)
        for g in range(n_groups):
            t_start = g * tube_size
            t_end   = min(T, t_start + tube_size)
            tube_mask[t_start:t_end] = _random_block_mask(ph, pw, n_mask).unsqueeze(0)

    # Vectorised pixel zeroing — avoids nested Python loops over patch positions
    mask_px = (
        tube_mask
        .repeat_interleave(patch_size, dim=1)   # (T, H, pw)
        .repeat_interleave(patch_size, dim=2)   # (T, H, W)
    )
    visible = clip * (~mask_px.unsqueeze(1)).float()   # (T, C, H, W)
    return visible, tube_mask


def random_patch_mask(h: int, w: int, patch_size: int, mask_ratio: float) -> Tensor:
    ph, pw = h // patch_size, w // patch_size
    n      = ph * pw
    n_mask = int(n * mask_ratio)
    idx    = torch.randperm(n)[:n_mask]
    mask   = torch.zeros(n, dtype=torch.bool)
    mask[idx] = True
    return mask.reshape(ph, pw)


def priority_weighted_mask(
    saliency: Optional[Tensor],
    hardness: Optional[Tensor],
    alpha: float,
    mask_ratio: float,
) -> Tensor:
    if saliency is None and hardness is None:
        return random_patch_mask(14 * 16, 14 * 16, 16, mask_ratio)
    s = saliency if saliency is not None else torch.zeros_like(hardness)
    h = hardness if hardness is not None else torch.zeros_like(saliency)
    alp    = alpha * s + (1 - alpha) * h
    ph, pw = alp.shape
    flat   = alp.flatten()
    idx    = torch.multinomial(torch.softmax(flat, dim=0),
                               int(ph * pw * mask_ratio), replacement=False)
    mask   = torch.zeros(ph * pw, dtype=torch.bool)
    mask[idx] = True
    return mask.reshape(ph, pw)


def random_tube_mask(
    n_frames: int,
    ph: int,
    pw: int,
    patch_size: int,
    mask_ratio: float,
    tube_size: Optional[int] = None,
) -> Tensor:
    """Return a (n_frames, ph, pw) bool tube mask.  See spatial_tube_mask for tube_size semantics."""
    n_mask = int(ph * pw * mask_ratio)
    if tube_size is None or tube_size >= n_frames:
        mask_2d = _random_block_mask(ph, pw, n_mask)
        return mask_2d.unsqueeze(0).expand(n_frames, -1, -1).contiguous()
    n_groups  = math.ceil(n_frames / tube_size)
    tube_mask = torch.zeros(n_frames, ph, pw, dtype=torch.bool)
    for g in range(n_groups):
        t_start = g * tube_size
        t_end   = min(n_frames, t_start + tube_size)
        tube_mask[t_start:t_end] = _random_block_mask(ph, pw, n_mask).unsqueeze(0)
    return tube_mask


def _random_block_mask(ph: int, pw: int, n_mask: int) -> Tensor:
    aspect = random.uniform(0.3, 1.0 / 0.3)
    h_mask = min(max(1, int(math.sqrt(n_mask * aspect))), ph)
    w_mask = min(max(1, int(math.sqrt(n_mask / aspect))), pw)
    r0     = random.randint(0, ph - h_mask)
    c0     = random.randint(0, pw - w_mask)
    mask   = torch.zeros(ph, pw, dtype=torch.bool)
    mask[r0:r0 + h_mask, c0:c0 + w_mask] = True
    return mask


# ── Masking dispatch ──────────────────────────────────────────────────────────

def _apply_image_mask(
    crop: Tensor,             # (C, H, W)
    cfg: ImageSSLTransformConfig,
    saliency: Optional[Tensor],
    hardness: Optional[Tensor],
    alpha: float,
    mask_ratio_override: Optional[float],
) -> Tuple[Tensor, Tensor]:
    """
    Apply the configured masking strategy to a single image crop.

    Returns
    -------
    masked_crop  : (C, H, W)
    spatial_mask : (ph, pw) bool
    """
    strategy   = cfg.mask_strategy
    patch_size = cfg.patch_size
    C, H, W    = crop.shape

    if strategy == MASK_STRATEGY_FREQ:
        return freq_mask_image_alp(
            crop, cfg.freq_mask, patch_size,
            saliency, hardness, alpha,
            mask_ratio_override=mask_ratio_override,
        )

    elif strategy == MASK_STRATEGY_SPATIAL:
        ratio = mask_ratio_override if mask_ratio_override is not None \
                else cfg.spatial_mask_ratio
        return spatial_patch_mask(crop, patch_size, ratio, block_style=False)

    elif strategy == MASK_STRATEGY_BOTH:
        # Step 1: frequency masking
        freq_masked, freq_spatial = freq_mask_image_alp(
            crop, cfg.freq_mask, patch_size,
            saliency, hardness, alpha,
            mask_ratio_override=mask_ratio_override,
        )
        # Step 2: additional spatial masking on the already freq-masked crop
        ratio = mask_ratio_override if mask_ratio_override is not None \
                else cfg.spatial_mask_ratio
        spatial_masked, spatial_mask = spatial_patch_mask(
            freq_masked, patch_size, ratio, block_style=False
        )
        # Union mask: patch is masked if either strategy flagged it
        combined_mask = freq_spatial | spatial_mask
        return spatial_masked, combined_mask

    raise ValueError(f"Unknown mask_strategy: {strategy!r}")


def _apply_video_mask(
    clip: Tensor,             # (T, C, H, W)
    cfg: VideoSSLTransformConfig,
    mask_ratio: float,
    patch_size: int,
) -> Tuple[Tensor, Tensor]:
    """
    Apply the configured masking strategy to a video clip.

    Strategies
    ----------
    "freq"    Frequency-domain tubelet masking.
              A frequency band is zeroed in Fourier space per temporal group
              of tube_size frames.  The spatial tube mask is derived from
              which patches lost the most spectral energy.
              Student sees a spectrally degraded clip; teacher sees the clean
              original.  Physics-motivated for ultrasound: the model learns
              tissue structure from residual low-frequency content.

    "spatial" Spatial block tubelet masking.
              A contiguous 2D rectangular block is sampled per temporal group
              of tube_size frames and zeroed in pixel space.  Classic V-JEPA
              approach: the model must predict completely occluded regions.

    "both"    Combined freq + spatial tubelet masking (recommended for V-JEPA).
              Both strategies are applied INDEPENDENTLY to the same clip:
                • Freq masking degrades spectral content at freq_tube positions.
                • Spatial masking independently samples spatial block positions.
                • Combined visible = freq-degraded clip with spatial blocks zeroed.
                • Tube mask = union of both masks.
              Two distinct pretext signals coexist:
                1. Recover spectral content from freq-only masked patches.
                2. Predict completely occluded spatial tube patches.

    cfg.tube_size controls temporal granularity for both masking functions:
    each group of tube_size consecutive frames shares the same spatial mask,
    but adjacent groups are independently sampled (spatiotemporal tubelets).

    Returns
    -------
    visible_clip : (T, C, H, W)
    tube_mask    : (T, ph, pw) bool  True = masked (target position)
    """
    strategy  = cfg.mask_strategy
    tube_size = cfg.tube_size   # propagate temporal granularity to all masking functions

    if strategy == MASK_STRATEGY_FREQ:
        return freq_mask_video(
            clip, cfg.freq_mask, patch_size,
            mask_ratio_override=mask_ratio,
            tube_size=tube_size,
        )

    elif strategy == MASK_STRATEGY_SPATIAL:
        return spatial_tube_mask(clip, patch_size, mask_ratio, tube_size=tube_size)

    elif strategy == MASK_STRATEGY_BOTH:
        # Frequency and spatial masking are applied independently, then layered:
        #
        #   Step 1 — freq masking on the ORIGINAL clip
        #     → freq_visible:  all frames spectrally degraded at freq_tube positions
        #     → freq_tube:     (T, ph, pw) bool — high-energy-loss patches
        #
        #   Step 2 — spatial block masking on the ORIGINAL clip
        #     → spatial_tube:  (T, ph, pw) bool — contiguous spatial block per group
        #     (we only need the mask; the zeroing is applied below)
        #
        #   Combined visible:
        #     spatial_tube positions → 0  (completely invisible to student)
        #     freq_tube only positions → spectrally degraded (partially visible)
        #     neither-masked positions → original context
        #
        #   This layering gives two distinct prediction targets:
        #     • recover spectral content from freq-degraded non-zero patches
        #     • predict completely occluded spatial tube regions
        freq_visible, freq_tube = freq_mask_video(
            clip, cfg.freq_mask, patch_size,
            mask_ratio_override=mask_ratio,
            tube_size=tube_size,
        )
        # Derive spatial tube mask from the original clip geometry (not freq_visible)
        # so the two masking patterns are fully independent.
        _, spatial_tube = spatial_tube_mask(
            clip, patch_size, cfg.spatial_mask_ratio, tube_size=tube_size
        )
        # Zero spatial tube positions in the freq-degraded clip
        mask_px = (
            spatial_tube
            .repeat_interleave(patch_size, dim=1)
            .repeat_interleave(patch_size, dim=2)
        )   # (T, H, W)
        combined_visible = freq_visible * (~mask_px.unsqueeze(1)).float()
        combined_tube    = freq_tube | spatial_tube
        return combined_visible, combined_tube

    raise ValueError(f"Unknown mask_strategy: {strategy!r}")


# ── Image SSL Transform ───────────────────────────────────────────────────────

class ImageSSLTransform:
    """
    Native-resolution multi-crop transform.

    Channel handling
    ----------------
    Uses to_canonical_tensor() to produce (3, H, W) RGB crops.
    Greyscale inputs are channel-repeated to R=G=B before any processing.
    ensure_rgb() in the backbone forward() is a no-op safety check.

    Masking strategy
    ----------------
    Controlled by cfg.mask_strategy:
      "freq"    Frequency-domain masking (default)
      "spatial" Random spatial patch masking
      "both"    Frequency masking then spatial masking, union mask

    Output dict
    -----------
    views["global"]       list of n_global tensors, each (C, H_i, W_i)
                          crop[0] = masked student; crop[1] = clean teacher
    views["global_pmask"] list of (ph_i, pw_i) bool — True=real patch
    views["local"]        list of n_local tensors, each (C, h_j, w_j)
    views["local_pmask"]  list of (ph_j, pw_j) bool
    views["mask"]         (ph_0, pw_0) bool — mask on student crop[0]
    """

    def __init__(self, cfg: ImageSSLTransformConfig):
        self.cfg  = cfg
        self._blur = T.RandomApply(
            [T.GaussianBlur(kernel_size=23, sigma=cfg.blur_sigma)], p=0.5
        ) if cfg.apply_blur else None

    def __call__(
        self,
        img,
        saliency: Optional[Tensor] = None,
        hardness: Optional[Tensor] = None,
        alpha: float = 1.0,
        mask_ratio_override: Optional[float] = None,
    ) -> dict:
        # Preserve original channel count (1 for greyscale, 3 for RGB)
        x = to_canonical_tensor(img)   # (C, H, W)
        c = self.cfg

        global_crops, global_pmasks = [], []
        local_crops,  local_pmasks  = [], []

        # ── Global crop 0: masked → student ──────────────────────────────────
        raw0 = _native_crop(x, c.global_crop_scale, c.patch_size,
                            c.min_crop_px, c.max_global_crop_px)
        raw0 = self._photometric(raw0)
        # Apply masking strategy before padding so it operates on real pixels only
        m0, spatial_mask = _apply_image_mask(
            raw0, c, saliency, hardness, alpha, mask_ratio_override
        )
        padded0, pmask0 = pad_to_patch_multiple(m0, c.patch_size)
        global_crops.append(padded0)
        global_pmasks.append(pmask0)

        # ── Global crop 1: clean → teacher ───────────────────────────────────
        raw1 = _native_crop(x, c.global_crop_scale, c.patch_size,
                            c.min_crop_px, c.max_global_crop_px)
        raw1 = self._photometric(raw1)
        padded1, pmask1 = pad_to_patch_multiple(raw1, c.patch_size)
        global_crops.append(padded1)
        global_pmasks.append(pmask1)

        # ── Additional global crops ───────────────────────────────────────────
        for _ in range(c.n_global_crops - 2):
            raw = _native_crop(x, c.global_crop_scale, c.patch_size,
                               c.min_crop_px, c.max_global_crop_px)
            p, pm = pad_to_patch_multiple(self._photometric(raw), c.patch_size)
            global_crops.append(p)
            global_pmasks.append(pm)

        # ── Local crops (no masking — CLS alignment only) ─────────────────────
        for _ in range(c.n_local_crops):
            raw = _native_crop(x, c.local_crop_scale, c.patch_size,
                               c.min_crop_px, c.max_global_crop_px)
            p, pm = pad_to_patch_multiple(self._photometric(raw), c.patch_size)
            local_crops.append(p)
            local_pmasks.append(pm)

        return {
            "global":       global_crops,
            "global_pmask": global_pmasks,
            "local":        local_crops,
            "local_pmask":  local_pmasks,
            "mask":         spatial_mask,
        }

    def _photometric(self, x: Tensor) -> Tensor:
        c = self.cfg
        if c.apply_speckle and random.random() < 0.5:
            x = add_speckle_noise(x, c.speckle_sigma)
        if c.apply_solarize and random.random() < 0.2:
            x = torch.where(x < c.solarize_thresh, x, 1 - x)
        if self._blur is not None:
            x = self._blur(x)
        return x


# ── Video SSL Transform ───────────────────────────────────────────────────────

class VideoSSLTransform:
    """
    Native-resolution V-JEPA spatiotemporal tubelet masking.

    Channel handling
    ----------------
    Frames are converted with to_canonical_tensor() to (3, H, W) RGB.
    Greyscale frames are channel-repeated to R=G=B.

    Masking strategy  (cfg.mask_strategy)
    --------------------------------------
    "freq"    Frequency-domain tubelet masking.
              Per temporal group of cfg.tube_size frames, an independent
              frequency band is removed in Fourier space.  The tube_mask marks
              patches with the highest spectral energy loss.  Physics-motivated
              for ultrasound speckle — the model learns from low-frequency
              tissue structure.

    "spatial" Spatial block tubelet masking.
              Per temporal group, an independent random rectangular block is
              sampled and zeroed in pixel space.  Classic V-JEPA approach.

    "both"    Combined (default).
              Freq and spatial masking are applied INDEPENDENTLY:
                - Freq degrades spectral content everywhere in the clip.
                - Spatial sampling independently selects block regions.
                - Combined visible = freq-degraded clip with spatial blocks
                  zeroed to 0 on top.
                - tube_mask = freq_tube | spatial_tube (union).
              Two complementary pretext objectives:
                1. Predict spectral content from freq-degraded but visible patches.
                2. Predict completely occluded spatial tube regions.

    Temporal grouping
    -----------------
    cfg.tube_size defines the temporal granularity of all masking.  Every group
    of tube_size consecutive frames shares the same spatial mask; adjacent groups
    are independently sampled (true V-JEPA spatiotemporal tubelets).

    Output dict
    -----------
    out["full"]          (T, C, H, W)  clean original clip (teacher input)
    out["visible"]       (T, C, H, W)  masked clip (student input)
    out["tube_mask"]     (T, ph, pw)   bool  True = masked target position
    out["padding_mask"]  (ph, pw)      bool  True = real (non-padded) patch
    """

    def __init__(self, cfg: VideoSSLTransformConfig, patch_size: int = 16):
        self.cfg        = cfg
        self.patch_size = patch_size

    def __call__(
        self,
        frames: List,
        mask_ratio: Optional[float] = None,
        saliency: Optional[Tensor] = None,
        hardness: Optional[Tensor] = None,
    ) -> dict:
        c = self.cfg
        if mask_ratio is None:
            mask_ratio = c.tube_mask_ratio

        frames = self._temporal_sample(frames, c.n_frames)

        # Preserve channel count per frame
        clip = torch.stack([to_canonical_tensor(f) for f in frames])  # (T, C, H, W)

        clip, padding_mask = self._native_spatial_crop(clip)

        if c.apply_speckle and random.random() < 0.5:
            clip = add_speckle_noise(clip, random.uniform(0, c.speckle_sigma))

        visible_clip, tube_mask = _apply_video_mask(
            clip, c, mask_ratio, self.patch_size
        )

        return {
            "full":         clip,
            "visible":      visible_clip,
            "tube_mask":    tube_mask,
            "padding_mask": padding_mask,
        }

    def _temporal_sample(self, frames: List, n: int) -> List:
        total = len(frames)
        if total <= n:
            return frames + [frames[-1]] * (n - total)
        start   = random.randint(0, total - n * self.cfg.temporal_stride)
        indices = [min(start + i * self.cfg.temporal_stride, total - 1) for i in range(n)]
        return [frames[i] for i in indices]

    def _native_spatial_crop(self, clip: Tensor) -> Tuple[Tensor, Tensor]:
        nT, C, H, W = clip.shape
        c           = self.cfg
        ps          = self.patch_size
        area        = H * W

        for _ in range(10):
            target = random.uniform(*c.crop_scale) * area
            aspect = random.uniform(3 / 4, 4 / 3)
            ch     = int(round(math.sqrt(target * aspect)))
            cw     = int(round(math.sqrt(target / aspect)))
            ch = max(ps, min(c.max_crop_px, round(ch / ps) * ps))
            cw = max(ps, min(c.max_crop_px, round(cw / ps) * ps))
            if ch <= H and cw <= W and ch >= c.min_crop_px and cw >= c.min_crop_px:
                top  = random.randint(0, H - ch)
                left = random.randint(0, W - cw)
                cropped      = clip[:, :, top:top + ch, left:left + cw]
                padding_mask = torch.ones(ch // ps, cw // ps, dtype=torch.bool)
                return cropped, padding_mask

        ch = max(ps, min(c.max_crop_px, round(H / ps) * ps))
        cw = max(ps, min(c.max_crop_px, round(W / ps) * ps))
        return clip[:, :, :ch, :cw], torch.ones(ch // ps, cw // ps, dtype=torch.bool)
