"""
tests/dataset_adapters/training_smoke.py
=========================================
Multi-dataset, multi-phase training smoke test.

Tests all four training phases (DINOv3 image SSL, V-JEPA2 video SSL,
cross-modal alignment, downstream fine-tuning) using combined data from
BUSI, EchoNet-Dynamic, and Benin-LUS.

Usage (from project root with the .venv active):

    python -m tests.dataset_adapters.training_smoke

Environment overrides:
    US_SMOKE_DEVICE       Force device  (e.g. "cuda:0", "cpu")
    US_BUSI_ROOT          Override BUSI data root
    US_ECHONET_ROOT       Override EchoNet-Dynamic data root
    US_BENIN_ROOT         Override Benin-LUS data root
    US_SKIP_PHASE1=1      Skip Phase 1 image SSL smoke
    US_SKIP_PHASE2=1      Skip Phase 2 video SSL smoke
    US_SKIP_PHASE3=1      Skip Phase 3 alignment smoke
    US_SKIP_PHASE4=1      Skip Phase 4 downstream heads smoke
"""
from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.adapters.busi import BUSIAdapter
from data.adapters.cardiac.camus import CAMUSAdapter
from data.adapters.cardiac.echonet import EchoNetDynamicAdapter
from data.adapters.cardiac.echonet_pediatric import EchoNetPediatricAdapter
from data.adapters.cardiac.ted import TEDAdapter
from data.adapters.lung.benin_lus import BeninLUSAdapter
from data.schema.manifest import ManifestWriter, USManifestEntry, load_manifest
from data.pipeline.dataset import ImageSSLDataset, VideoSSLDataset
from data.pipeline.downstream_dataset import DownstreamDataset, PatientLevelDataset
from data.pipeline.datamodule import USFoundationDataModule
from data.pipeline.transforms import (
    ImageSSLTransformConfig,
    VideoSSLTransformConfig,
    MASK_STRATEGY_FREQ,
)
from models.branches.image_branch import ImageBranch
from models.branches.video_branch import build_video_branch
from models.branches.shared import CrossBranchDistillation, PrototypeHead
from models.registry import build_image_backbone
from models.heads.classification_head import LinearClsHead
from models.heads.segmentation_head import LinearSegHead
from models.heads.regression_head import RegressionHead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("training_smoke")

# ── Paths ─────────────────────────────────────────────────────────────────────
_STORE = Path("/capstor/store/cscs/swissai/a127/ultrasound/raw")
_DEFAULT_CAMUS_ROOT          = _STORE / "cardiac" / "CAMUS"
_DEFAULT_BUSI_ROOT           = _STORE / "breast"  / "BUSI"
_DEFAULT_ECHONET_ROOT        = _STORE / "cardiac" / "EchoNet-Dynamic"
_DEFAULT_ECHONET_PED_ROOT    = _STORE / "cardiac" / "EchoNet-Pediatric"
_DEFAULT_TED_ROOT            = _STORE / "cardiac" / "TED"
_DEFAULT_BENIN_ROOT          = _STORE / "lung"    / "Benin_Videos"

_SMOKE_OUT  = _ROOT / "dataset_exploration_outputs" / "smoke"
_SMOKE_CFG  = _ROOT / "configs" / "smoke" / "multi_dataset_smoke.yaml"
_COMBINED_MANIFEST = _SMOKE_OUT / "combined_smoke_manifest.jsonl"

N_SMOKE_ENTRIES = 32   # entries per dataset in the combined manifest
N_SMOKE_BATCHES = 2    # forward passes per phase


# ── Device ───────────────────────────────────────────────────────────────────

def _auto_device() -> str:
    """Auto-select: respect US_SMOKE_DEVICE, otherwise prefer CUDA."""
    env = os.environ.get("US_SMOKE_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        dev = "cuda:0"
        log.info("CUDA available — using %s (%s)",
                 dev, torch.cuda.get_device_name(0))
        return dev
    log.warning("CUDA not available — running on CPU (will be slow)")
    return "cpu"


# ── Manifest helpers ──────────────────────────────────────────────────────────

def _root(env_var: str, default: Path) -> Optional[Path]:
    env = os.environ.get(env_var)
    p = Path(env) if env else default
    return p if p.exists() else None


def _build_camus_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_CAMUS_ROOT", _DEFAULT_CAMUS_ROOT)
    if root is None:
        log.warning("CAMUS root not found — skipping")
        return []
    try:
        import SimpleITK  # noqa: F401
    except ImportError:
        log.warning("SimpleITK not installed — skipping CAMUS")
        return []
    entries: List[USManifestEntry] = []
    for e in CAMUSAdapter(root).iter_entries():
        # Prefer image entries for Phase 1 image SSL coverage
        if e.modality_type in ("image", "pseudo_video"):
            entries.append(e)
        if len(entries) >= n:
            break
    log.info("CAMUS: %d entries", len(entries))
    return entries


def _build_busi_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BUSI_ROOT", _DEFAULT_BUSI_ROOT)
    if root is None:
        log.warning("BUSI root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BUSIAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("BUSI: %d entries", len(entries))
    return entries


def _build_echonet_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_ECHONET_ROOT", _DEFAULT_ECHONET_ROOT)
    if root is None:
        log.warning("EchoNet root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in EchoNetDynamicAdapter(root).iter_entries():
        if e.split == "train":
            entries.append(e)
        if len(entries) >= n:
            break
    log.info("EchoNet: %d entries", len(entries))
    return entries


def _build_benin_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_BENIN_ROOT", _DEFAULT_BENIN_ROOT)
    if root is None:
        log.warning("Benin-LUS root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in BeninLUSAdapter(root).iter_entries():
        entries.append(e)
        if len(entries) >= n:
            break
    log.info("Benin-LUS: %d entries", len(entries))
    return entries


def _build_echonet_ped_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_ECHONET_PED_ROOT", _DEFAULT_ECHONET_PED_ROOT)
    if root is None:
        log.warning("EchoNet-Pediatric root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    for e in EchoNetPediatricAdapter(root).iter_entries():
        if e.split == "train":
            entries.append(e)
        if len(entries) >= n:
            break
    log.info("EchoNet-Pediatric: %d entries", len(entries))
    return entries


def _build_ted_entries(n: int = N_SMOKE_ENTRIES) -> List[USManifestEntry]:
    root = _root("US_TED_ROOT", _DEFAULT_TED_ROOT)
    if root is None:
        log.warning("TED root not found — skipping")
        return []
    entries: List[USManifestEntry] = []
    # Only take 'video' modality entries for the smoke manifest (ED/ES images
    # are a by-product of the same file; video entries are sufficient here).
    for e in TEDAdapter(root).iter_entries():
        if e.modality_type == "video":
            entries.append(e)
        if len(entries) >= n:
            break
    log.info("TED: %d entries", len(entries))
    return entries


def build_combined_manifest(force: bool = False) -> Path:
    """Build (or reuse) the combined smoke manifest."""
    _SMOKE_OUT.mkdir(parents=True, exist_ok=True)

    if _COMBINED_MANIFEST.exists() and not force:
        log.info("Reusing existing manifest: %s", _COMBINED_MANIFEST)
        return _COMBINED_MANIFEST

    all_entries: List[USManifestEntry] = (
        _build_camus_entries()
        + _build_busi_entries()
        + _build_echonet_entries()
        + _build_echonet_ped_entries()
        + _build_ted_entries()
        + _build_benin_entries()
    )

    if not all_entries:
        raise RuntimeError("No entries found — check dataset paths.")

    with ManifestWriter(_COMBINED_MANIFEST) as w:
        for e in all_entries:
            w.write(e)

    log.info("Combined manifest written: %d entries → %s",
             len(all_entries), _COMBINED_MANIFEST)
    return _COMBINED_MANIFEST


# ── Config / DataModule helpers ───────────────────────────────────────────────

def load_smoke_config() -> dict:
    with open(_SMOKE_CFG) as f:
        return yaml.safe_load(f)


def build_datamodule(cfg: dict) -> USFoundationDataModule:
    img_cfg = ImageSSLTransformConfig(
        n_global_crops=cfg["transforms"]["image"]["n_global_crops"],
        n_local_crops=cfg["transforms"]["image"]["n_local_crops"],
        max_global_crop_px=cfg["transforms"]["image"]["max_global_crop_px"],
        min_crop_px=cfg["transforms"]["image"]["min_crop_px"],
        mask_strategy=MASK_STRATEGY_FREQ,
    )
    vid_cfg = VideoSSLTransformConfig(
        n_frames=cfg["transforms"]["video"]["n_frames"],
        max_crop_px=cfg["transforms"]["video"]["max_crop_px"],
        min_crop_px=cfg["transforms"]["video"]["min_crop_px"],
    )
    dm = USFoundationDataModule(
        manifest_path=str(_COMBINED_MANIFEST),
        image_batch_size=cfg["loaders"]["image_batch_size"],
        video_batch_size=cfg["loaders"]["video_batch_size"],
        num_workers=cfg["loaders"]["num_workers"],
        pin_memory=cfg["loaders"].get("pin_memory", False),
        image_cfg=img_cfg,
        video_cfg=vid_cfg,
        total_training_steps=cfg["curriculum"]["total_training_steps"],
        image_samples_per_epoch=cfg["curriculum"]["image_samples_per_epoch"],
        video_samples_per_epoch=cfg["curriculum"]["video_samples_per_epoch"],
    )
    dm.setup()
    log.info(
        "DataModule ready — image entries: %d | video entries: %d",
        len(dm._image_entries), len(dm._video_entries),
    )
    return dm


def _to_dev(batch: dict, device: str) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    return 1.0 - (x * y).sum(dim=-1).mean()


# ── Native-resolution collate for DownstreamDataset ──────────────────────────

def _downstream_collate(samples: list) -> dict:
    """
    Collate DownstreamDataset samples at native resolution.

    Images in a batch will generally have different sizes — that is by design
    (the system is resolution-agnostic).  We pad each image to the batch-max
    (H, W) with zeros and produce a boolean padding_mask (B, ph, pw) where
    True = valid patch.  The DINOv3 backbone then ignores padding tokens via
    the attention bias we just fixed.
    """
    patch_size = 16

    # Determine batch-max spatial dims
    max_h = max(s["image"].shape[-2] for s in samples)
    max_w = max(s["image"].shape[-1] for s in samples)

    # Round up to patch-grid multiples so ph/pw are integers
    max_h = ((max_h + patch_size - 1) // patch_size) * patch_size
    max_w = ((max_w + patch_size - 1) // patch_size) * patch_size

    ph = max_h // patch_size
    pw = max_w // patch_size
    B  = len(samples)
    C  = samples[0]["image"].shape[0]

    images       = torch.zeros(B, C, max_h, max_w)
    padding_mask = torch.zeros(B, ph, pw, dtype=torch.bool)

    for i, s in enumerate(samples):
        _, h, w = s["image"].shape
        images[i, :, :h, :w] = s["image"]
        # Mark patches that are fully covered by the actual image as valid
        vh = h // patch_size
        vw = w // patch_size
        padding_mask[i, :vh, :vw] = True

    # Collate all other fields
    out: dict = {"image": images, "padding_mask": padding_mask}
    for key in samples[0]:
        if key == "image":
            continue
        vals = [s[key] for s in samples]
        v0 = vals[0]
        try:
            if isinstance(v0, torch.Tensor):
                out[key] = torch.stack(vals)
            elif isinstance(v0, (int, float)):
                out[key] = torch.tensor(vals)
            elif isinstance(v0, bool):
                out[key] = torch.tensor(vals, dtype=torch.bool)
            else:
                out[key] = vals       # lists, strings, dicts, LabelTargets etc.
        except Exception:
            out[key] = vals
    return out


# ── Phase 1: Image SSL ────────────────────────────────────────────────────────

def phase1_smoke(dm: USFoundationDataModule, device: str) -> None:
    log.info("=== Phase 1: Image SSL (DINOv3-S) ===")
    dtype = torch.float32

    student = build_image_backbone("dinov3_s", dtype=dtype)
    teacher = build_image_backbone("dinov3_s", dtype=dtype)
    branch = ImageBranch(student=student, teacher=teacher).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(branch.student.parameters(), lr=1e-4)

    loader = dm.image_loader()
    branch.train()
    n = 0
    for batch in loader:
        batch = _to_dev(batch, device)
        global_crops = batch["global_crops"].to(dtype)   # (B, 2, C, H, W)
        local_crops  = batch.get("local_crops")
        patch_mask   = batch.get("patch_mask")

        opt.zero_grad()

        # Teacher on clean crop (no padding mask needed — uniform squares)
        t_out = branch.forward_teacher(global_crops[:, 1])
        # Student on masked crop
        s_out = branch.forward_student(global_crops[:, 0])

        loss = cosine_loss(s_out["cls"], t_out["cls"])

        # Patch-level loss (if patch tokens available)
        if "patch_tokens" in s_out and "patch_tokens" in t_out:
            loss = loss + 0.5 * cosine_loss(
                s_out["patch_tokens"].mean(1),
                t_out["patch_tokens"].mean(1),
            )

        # Local crops
        if local_crops is not None:
            local_crops = local_crops.to(dtype)
            for i in range(local_crops.shape[1]):
                s_loc = branch.forward_student(local_crops[:, i])
                loss = loss + 0.3 * cosine_loss(s_loc["cls"], t_out["cls"])

        loss.backward()
        nn.utils.clip_grad_norm_(branch.student.parameters(), 1.0)
        opt.step()
        branch.update_teacher(momentum=0.9995)

        log.info("  Phase1 batch=%d  loss=%.4f  cls.shape=%s",
                 n + 1, loss.item(), tuple(s_out["cls"].shape))
        assert torch.isfinite(loss), f"Non-finite loss at batch {n+1}"
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    assert n > 0, "Phase 1: no image batches yielded — check manifest/stream split"
    log.info("Phase 1 PASS (%d batches)", n)


# ── Phase 2: Video SSL ────────────────────────────────────────────────────────

def phase2_smoke(dm: USFoundationDataModule, device: str) -> None:
    log.info("=== Phase 2: Video SSL (V-JEPA2) ===")
    if not dm._video_entries:
        log.warning("Phase 2 SKIP — no video entries in manifest")
        return

    dtype = torch.float32
    branch = build_video_branch(dtype=dtype, device=device)
    opt = torch.optim.AdamW(branch.student.parameters(), lr=1e-4)

    loader = dm.video_loader()
    branch.train()
    n = 0
    for batch in loader:
        batch = _to_dev(batch, device)
        full_clip  = batch["full_clips"].to(dtype)          # (B, T, C, H, W)
        vis_clip   = batch["visible_clips"].to(dtype)
        tube_mask  = batch.get("tube_masks")
        pad_mask   = batch.get("padding_masks")
        valid_fr   = batch.get("valid_frames")

        opt.zero_grad()
        t_out = branch.forward_teacher(full_clip, padding_mask=pad_mask,
                                       valid_frames=valid_fr)
        s_out = branch.forward_student(vis_clip, tube_mask=tube_mask,
                                       padding_mask=pad_mask,
                                       valid_frames=valid_fr)

        loss = cosine_loss(s_out["clip_cls"], t_out["clip_cls"])
        loss.backward()
        nn.utils.clip_grad_norm_(branch.student.parameters(), 1.0)
        opt.step()
        branch.update_teacher(momentum=0.9995)

        log.info("  Phase2 batch=%d  loss=%.4f  clip_cls.shape=%s",
                 n + 1, loss.item(), tuple(s_out["clip_cls"].shape))
        assert torch.isfinite(loss), f"Non-finite loss at batch {n+1}"
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    if n == 0:
        log.warning("Phase 2 SKIP — video loader yielded no batches")
        return
    log.info("Phase 2 PASS (%d batches)", n)


# ── Phase 3: Cross-modal alignment ───────────────────────────────────────────

def phase3_smoke(dm: USFoundationDataModule, device: str) -> None:
    log.info("=== Phase 3: Cross-modal Alignment ===")
    if not dm._video_entries:
        log.warning("Phase 3 SKIP — no video entries in manifest")
        return

    dtype = torch.float32

    # Image branch
    img_student = build_image_backbone("dinov3_s", dtype=dtype)
    img_teacher = build_image_backbone("dinov3_s", dtype=dtype)
    img_branch = ImageBranch(img_student, img_teacher).to(device=device, dtype=dtype)

    # Video branch
    vid_branch = build_video_branch(dtype=dtype, device=device)

    D_img = img_branch.embed_dim              # 384 for dinov3_s
    D_vid = vid_branch.student.hidden_size    # 1024 for vjepa2_l
    align_dim = 256

    cross = CrossBranchDistillation(img_dim=D_img, vid_dim=D_vid,
                                    align_dim=align_dim).to(device=device, dtype=dtype)
    # PrototypeHead works in a single shared space.
    # Video tokens (D_vid) are projected to D_img before assignment.
    proto     = PrototypeHead(embed_dim=D_img, n_prototypes=64).to(device=device, dtype=dtype)
    vid_to_img = nn.Linear(D_vid, D_img, bias=False).to(device=device, dtype=dtype)

    params = (
        list(img_branch.student.parameters())
        + list(vid_branch.student.parameters())
        + list(cross.parameters())
        + list(proto.parameters())
        + list(vid_to_img.parameters())
    )
    opt = torch.optim.AdamW(params, lr=1e-4)

    img_branch.train()
    vid_branch.train()
    cross.train()
    proto.train()
    vid_to_img.train()

    n = 0
    for dual in dm.combined_loader():
        img_batch = _to_dev(dual.image_batch, device)
        vid_batch = _to_dev(dual.video_batch, device)

        global_crops = img_batch["global_crops"].to(dtype)   # (B, 2, C, H, W)
        full_clip    = vid_batch["full_clips"].to(dtype)      # (B, T, C, H, W)
        vis_clip     = vid_batch["visible_clips"].to(dtype)
        tube_mask    = vid_batch.get("tube_masks")
        pad_mask_vid = vid_batch.get("padding_masks")

        opt.zero_grad()

        # Image arm
        t_img = img_branch.forward_teacher(global_crops[:, 1])
        s_img = img_branch.forward_student(global_crops[:, 0])
        loss_img = cosine_loss(s_img["cls"], t_img["cls"])

        # Video arm
        t_vid = vid_branch.forward_teacher(full_clip, padding_mask=pad_mask_vid)
        s_vid = vid_branch.forward_student(vis_clip, tube_mask=tube_mask,
                                           padding_mask=pad_mask_vid)
        loss_vid = cosine_loss(s_vid["clip_cls"], t_vid["clip_cls"])

        # Cross-branch distillation
        img_patches = t_img["patch_tokens"]                   # (B, N, D_img)
        vid_tubes   = s_vid.get("tube_tokens",
                       s_vid["clip_cls"].unsqueeze(1))        # (B, M, D_vid)
        loss_cross  = cross(img_patches, vid_tubes)

        # Prototype consistency: project video to img dim before assignment
        vid_tubes_proj = vid_to_img(vid_tubes)                # (B, M, D_img)
        loss_proto = proto.consistency_loss(img_patches, vid_tubes_proj)

        loss = loss_img + loss_vid + loss_cross + 0.5 * loss_proto
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        img_branch.update_teacher()
        vid_branch.update_teacher()

        log.info(
            "  Phase3 batch=%d  loss=%.4f  "
            "(img=%.3f vid=%.3f cross=%.3f proto=%.3f)",
            n + 1, loss.item(), loss_img.item(),
            loss_vid.item(), loss_cross.item(), loss_proto.item(),
        )
        assert torch.isfinite(loss), f"Non-finite loss at batch {n+1}"
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    if n == 0:
        log.warning("Phase 3 SKIP — combined loader yielded no batches")
        return
    log.info("Phase 3 PASS (%d batches)", n)


# ── Phase 4: Downstream heads ─────────────────────────────────────────────────

def _build_backbone_frozen(device: str, dtype: torch.dtype) -> nn.Module:
    bb = build_image_backbone("dinov3_s", dtype=dtype).to(device=device, dtype=dtype)
    for p in bb.parameters():
        p.requires_grad_(False)
    bb.eval()
    return bb


def _phase4_classification_smoke(
    busi_entries: List[USManifestEntry], device: str
) -> None:
    """Binary malignancy classification on BUSI."""
    entries = [e for e in busi_entries if e.task_type != "ssl_only"][:16]
    if not entries:
        log.warning("Phase4/cls SKIP — no BUSI supervised entries")
        return

    dtype = torch.float32
    bb = _build_backbone_frozen(device, dtype)
    D  = bb.hidden_size
    head = LinearClsHead(embed_dim=D, n_classes=1).to(device=device, dtype=dtype)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-3)

    ds = DownstreamDataset(entries, active_head_ids=["breast_malignancy_cls"])
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=_downstream_collate)

    head.train()
    n = 0
    for batch in loader:
        imgs     = batch["image"].to(device=device, dtype=dtype)
        pad_mask = batch.get("padding_mask")
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        if pad_mask is not None:
            pad_mask = pad_mask.to(device=device)

        opt.zero_grad()
        with torch.no_grad():
            feats = bb(imgs, padding_mask=pad_mask)
        logits = head(feats["cls"])                             # (B, 1)
        cls_label = batch.get("cls_label")
        if cls_label is None or (isinstance(cls_label, torch.Tensor) and (cls_label < 0).all()):
            loss = logits.mean() * 0.0
        else:
            lbl = (cls_label if isinstance(cls_label, torch.Tensor)
                   else torch.tensor(cls_label)).to(device=device, dtype=dtype)
            lbl = lbl.float().unsqueeze(1).clamp(0, 1)
            loss = F.binary_cross_entropy_with_logits(logits, lbl)
        loss.backward()
        opt.step()

        log.info("  Phase4/cls batch=%d  loss=%.4f  logits.shape=%s",
                 n + 1, loss.item(), tuple(logits.shape))
        assert torch.isfinite(loss)
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    log.info("Phase4/classification PASS (%d batches)", n)


def _phase4_segmentation_smoke(
    busi_entries: List[USManifestEntry], device: str
) -> None:
    """Lesion segmentation on BUSI."""
    entries = [e for e in busi_entries
               if e.task_type in ("seg", "seg_cls") and e.seg_mask_paths][:8]
    if not entries:
        log.warning("Phase4/seg SKIP — no BUSI segmentation entries")
        return

    dtype = torch.float32
    bb   = _build_backbone_frozen(device, dtype)
    D    = bb.hidden_size
    head = LinearSegHead(embed_dim=D, n_classes=1).to(device=device, dtype=dtype)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-3)

    ds = DownstreamDataset(entries, active_head_ids=["breast_lesion_seg"])
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=_downstream_collate)

    head.train()
    n = 0
    for batch in loader:
        imgs     = batch["image"].to(device=device, dtype=dtype)
        pad_mask = batch.get("padding_mask")
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        B, _, H, W = imgs.shape
        ph, pw = H // 16, W // 16
        if pad_mask is not None:
            pad_mask = pad_mask.to(device=device)

        opt.zero_grad()
        with torch.no_grad():
            feats = bb(imgs, padding_mask=pad_mask)
        patch_tokens = feats["patch_tokens"]                  # (B, N, D)
        seg_logits   = head(patch_tokens, ph=ph, pw=pw)       # (B, 1, H, W)

        seg_mask = batch.get("seg_mask")
        if seg_mask is not None and seg_mask.shape[-1] == W:
            seg_mask = seg_mask.to(device=device, dtype=dtype)
            if seg_mask.shape[1] != 1:
                seg_mask = seg_mask[:, :1]
            loss = F.binary_cross_entropy_with_logits(seg_logits, seg_mask)
        else:
            loss = seg_logits.mean() * 0.0
        loss.backward()
        opt.step()

        log.info("  Phase4/seg batch=%d  loss=%.4f  seg_logits.shape=%s",
                 n + 1, loss.item(), tuple(seg_logits.shape))
        assert torch.isfinite(loss)
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    log.info("Phase4/segmentation PASS (%d batches)", n)


def _phase4_regression_smoke(
    echonet_entries: List[USManifestEntry], device: str
) -> None:
    """Ejection-fraction regression on EchoNet."""
    if not echonet_entries:
        log.warning("Phase4/reg SKIP — no EchoNet entries")
        return

    dtype = torch.float32
    bb   = _build_backbone_frozen(device, dtype)
    D    = bb.hidden_size
    head = RegressionHead(embed_dim=D, output_min=0.0, output_max=100.0).to(
        device=device, dtype=dtype)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-3)

    ds = DownstreamDataset(echonet_entries, active_head_ids=["cardiac_ef_regression"])
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        collate_fn=_downstream_collate)

    head.train()
    n = 0
    for batch in loader:
        imgs     = batch["image"].to(device=device, dtype=dtype)
        pad_mask = batch.get("padding_mask")
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        if pad_mask is not None:
            pad_mask = pad_mask.to(device=device)

        opt.zero_grad()
        with torch.no_grad():
            feats = bb(imgs, padding_mask=pad_mask)
        pred = head(feats["cls"])                             # (B, 1)

        # EF target from source_meta or label_targets
        label_targets = batch.get("label_targets", [])
        ef_vals = []
        for lt in label_targets:
            if hasattr(lt, "head_id") and lt.head_id == "cardiac_ef_regression":
                ef_vals.append(lt.value)
        if ef_vals:
            target = torch.tensor(ef_vals, device=device, dtype=dtype).unsqueeze(1)
            loss = F.smooth_l1_loss(pred, target)
        else:
            loss = pred.mean() * 0.0
        loss.backward()
        opt.step()

        log.info("  Phase4/reg batch=%d  loss=%.4f  pred.shape=%s",
                 n + 1, loss.item(), tuple(pred.shape))
        assert torch.isfinite(loss)
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    log.info("Phase4/regression PASS (%d batches)", n)


def _patient_collate(samples: list) -> dict:
    """
    Collate PatientLevelDataset samples.

    Each sample has:
      frames:      (max_frames, C, H, W)
      frame_mask:  (max_frames,) bool
      label_targets: list[LabelTarget]

    Frames are padded to batch-max (H, W) preserving native resolution.
    """
    patch_size = 16
    max_h = max(s["frames"].shape[-2] for s in samples)
    max_w = max(s["frames"].shape[-1] for s in samples)
    max_h = ((max_h + patch_size - 1) // patch_size) * patch_size
    max_w = ((max_w + patch_size - 1) // patch_size) * patch_size

    F_  = samples[0]["frames"].shape[0]
    C   = samples[0]["frames"].shape[1]
    B   = len(samples)
    ph, pw = max_h // patch_size, max_w // patch_size

    frames_t    = torch.zeros(B, F_, C, max_h, max_w)
    frame_masks = torch.zeros(B, F_, dtype=torch.bool)
    pad_masks   = torch.zeros(B, ph, pw, dtype=torch.bool)

    for i, s in enumerate(samples):
        h, w = s["frames"].shape[-2], s["frames"].shape[-1]
        frames_t[i, :, :, :h, :w] = s["frames"]
        frame_masks[i]              = s["frame_mask"]
        vh, vw = h // patch_size, w // patch_size
        pad_masks[i, :vh, :vw]     = True

    return {
        "frames":        frames_t,
        "frame_mask":    frame_masks,
        "padding_mask":  pad_masks,
        "label_targets": [s["label_targets"] for s in samples],
    }


def _phase4_patient_cls_smoke(
    benin_entries: List[USManifestEntry], device: str
) -> None:
    """Patient-level TB classification on Benin-LUS."""
    if not benin_entries:
        log.warning("Phase4/patient_cls SKIP — no Benin entries")
        return

    dtype = torch.float32
    bb   = _build_backbone_frozen(device, dtype)
    D    = 384
    head = LinearClsHead(embed_dim=D, n_classes=1).to(device=device, dtype=dtype)
    opt  = torch.optim.AdamW(head.parameters(), lr=1e-3)

    ds = PatientLevelDataset(
        benin_entries,
        active_head_ids=["lus_patient_tb"],
        max_frames=4,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False,
                        collate_fn=_patient_collate)

    head.train()
    n = 0
    for batch in loader:
        frames    = batch["frames"].to(device=device, dtype=dtype)  # (B, F, C, H, W)
        fm        = batch["frame_mask"].to(device=device)            # (B, F)
        pad_mask  = batch["padding_mask"].to(device=device)          # (B, ph, pw)
        B, F_, C_, H, W = frames.shape

        if C_ == 1:
            frames = frames.repeat(1, 1, 3, 1, 1)

        # Flatten frames, run backbone, mean-pool valid frames for patient repr
        frames_flat = frames.view(B * F_, frames.shape[2], H, W)
        pm_flat     = pad_mask.unsqueeze(1).expand(B, F_, -1, -1
                       ).reshape(B * F_, *pad_mask.shape[1:])

        opt.zero_grad()
        with torch.no_grad():
            feats_flat = bb(frames_flat, padding_mask=pm_flat)
        cls_flat     = feats_flat["cls"].view(B, F_, D)             # (B, F, D)
        fm_f         = fm.float().unsqueeze(-1)                     # (B, F, 1)
        patient_feat = (cls_flat * fm_f).sum(1) / fm_f.sum(1).clamp(min=1)

        logits = head(patient_feat)                                 # (B, 1)

        label_targets_list = batch.get("label_targets", [])
        tb_vals = []
        for patient_targets in label_targets_list:
            for lt in (patient_targets if isinstance(patient_targets, list) else []):
                if hasattr(lt, "head_id") and lt.head_id == "lus_patient_tb":
                    tb_vals.append(float(lt.value))
                    break
        if tb_vals:
            target = torch.tensor(tb_vals, device=device, dtype=dtype).unsqueeze(1)
            target = target.clamp(0, 1)
            loss = F.binary_cross_entropy_with_logits(logits, target)
        else:
            loss = logits.mean() * 0.0
        loss.backward()
        opt.step()

        log.info("  Phase4/patient_cls batch=%d  loss=%.4f  logits.shape=%s",
                 n + 1, loss.item(), tuple(logits.shape))
        assert torch.isfinite(loss)
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    log.info("Phase4/patient_classification PASS (%d batches)", n)


def phase4_smoke(
    dm: USFoundationDataModule,
    busi_entries: List[USManifestEntry],
    echonet_entries: List[USManifestEntry],
    benin_entries: List[USManifestEntry],
    device: str,
) -> None:
    log.info("=== Phase 4: Downstream Heads ===")
    _phase4_classification_smoke(busi_entries, device)
    _phase4_segmentation_smoke(busi_entries, device)
    _phase4_regression_smoke(echonet_entries, device)
    _phase4_patient_cls_smoke(benin_entries, device)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = _auto_device()
    log.info("Device: %s", device)

    # Build manifests
    log.info("Building combined smoke manifest …")
    build_combined_manifest()

    # Cache per-dataset entries for Phase 4
    busi_entries    = _build_busi_entries()
    echonet_entries = _build_echonet_entries()
    benin_entries   = _build_benin_entries()

    # DataModule
    cfg = load_smoke_config()
    dm  = build_datamodule(cfg)

    results = {}

    def _run(name: str, fn, *args):
        skip_var = f"US_SKIP_{name.upper().replace(' ', '_')}"
        if os.environ.get(skip_var, "0") == "1":
            log.info("Skipping %s (env %s=1)", name, skip_var)
            results[name] = "SKIP"
            return
        try:
            fn(*args)
            results[name] = "PASS"
        except Exception:
            results[name] = "FAIL"
            log.error("%s FAILED:\n%s", name, traceback.format_exc())

    _run("PHASE1", phase1_smoke, dm, device)
    _run("PHASE2", phase2_smoke, dm, device)
    _run("PHASE3", phase3_smoke, dm, device)
    _run("PHASE4", phase4_smoke, dm, busi_entries, echonet_entries, benin_entries, device)

    # Summary
    print("\n" + "=" * 60)
    print("SMOKE TEST SUMMARY")
    print("=" * 60)
    for phase, status in results.items():
        icon = "✓" if status == "PASS" else ("–" if status == "SKIP" else "✗")
        print(f"  {icon}  {phase:<12}  {status}")
    print("=" * 60)

    if any(v == "FAIL" for v in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
