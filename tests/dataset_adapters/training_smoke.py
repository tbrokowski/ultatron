"""
tests/dataset_adapters/training_smoke.py
=========================================
Comprehensive multi-phase, multi-dataset training smoke test.

Phases
------
  Phase 1 — Image SSL  (DINOv3-S student/teacher)
  Phase 2 — Video SSL  (V-JEPA2-L student/teacher)
  Phase 3 — Cross-modal Alignment
  Phase 4 — All 11 finetune experiments  (Ultatron backbone, setup + 1-batch verify)
  Phase 5 — All 7 ablation backbones     (model load + dummy forward pass)

Manifest coverage
-----------------
  ALL datasets registered in ADAPTER_REGISTRY.
  Roots are read from configs/run1/data_run1.yaml (datasets: section).
  N_SMOKE_ENTRIES entries per dataset; missing / broken datasets are logged and
  skipped — they never abort the run.

Error handling
--------------
  No `assert` inside phase functions; all checks log errors and continue.
  Every Phase 4 experiment and every Phase 5 backbone is individually
  try/except-wrapped.  A final summary table shows PASS / FAIL / SKIP
  for every item.  sys.exit(1) fires only at the very end if any FAIL.

Usage (from project root, with .venv active):

    python -m tests.dataset_adapters.training_smoke

Environment overrides
---------------------
    US_SMOKE_DEVICE              Force device  (e.g. "cuda:0", "cpu")
    US_SKIP_PHASE1=1             Skip Phase 1
    US_SKIP_PHASE2=1             Skip Phase 2
    US_SKIP_PHASE3=1             Skip Phase 3
    US_SKIP_PHASE4=1             Skip Phase 4
    US_SKIP_PHASE5=1             Skip Phase 5
    US_SMOKE_FORCE_REBUILD=1     Force manifest rebuild
    US_USFM_CHECKPOINT           Override USFM checkpoint path
    US_ECHOCARE_CHECKPOINT       Override EchoCare checkpoint path
    US_OPENUS_CHECKPOINT         Override OpenUS checkpoint path
    US_OPENUS_VMAMBA_CHECKPOINT  Override OpenUS VMamba backbone checkpoint
"""
from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

# ── Project root on sys.path ──────────────────────────────────────────────────
_ROOT = Path(__file__).parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.adapters import ADAPTER_REGISTRY
from data.schema.manifest import ManifestWriter, USManifestEntry
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
from finetune.backbones.paths import ablation_weight_path
from finetune.backbones.registry import build_encoder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("training_smoke")

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_CONFIG     = _ROOT / "configs" / "run1" / "data_run1.yaml"
_FINETUNE_CFG    = _ROOT / "configs" / "finetune"
_SMOKE_CFG       = _ROOT / "configs" / "smoke" / "multi_dataset_smoke.yaml"
_SMOKE_OUT       = _ROOT / "dataset_exploration_outputs" / "smoke"
_COMBINED_MANIFEST = _SMOKE_OUT / "combined_smoke_manifest.jsonl"

N_SMOKE_ENTRIES = 8    # manifest entries per dataset
N_SMOKE_BATCHES = 2    # forward-pass batches per phase


# ── Device ────────────────────────────────────────────────────────────────────

def _auto_device() -> str:
    env = os.environ.get("US_SMOKE_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        dev = "cuda:0"
        log.info("CUDA available — using %s (%s)", dev, torch.cuda.get_device_name(0))
        return dev
    log.warning("CUDA not available — running on CPU (will be slow)")
    return "cpu"


# ── Dataset roots ─────────────────────────────────────────────────────────────

def _load_all_dataset_roots() -> Dict[str, str]:
    """
    Load the datasets: mapping from configs/run1/data_run1.yaml.

    Returns a dict  dataset_id -> root_path_string.
    """
    if not _DATA_CONFIG.exists():
        log.warning("data_run1.yaml not found at %s — roots unavailable", _DATA_CONFIG)
        return {}
    with open(_DATA_CONFIG) as f:
        raw = yaml.safe_load(f)
    roots = raw.get("datasets", {})
    log.info("Loaded %d dataset roots from %s", len(roots), _DATA_CONFIG)
    return roots


# ── Manifest building ─────────────────────────────────────────────────────────

def build_all_dataset_entries(
    dataset_roots: Dict[str, str],
    n_per_dataset: int = N_SMOKE_ENTRIES,
) -> Tuple[List[USManifestEntry], Dict[str, str]]:
    """
    Iterate ADAPTER_REGISTRY and collect up to n_per_dataset entries per
    available dataset.

    Returns
    -------
    all_entries : list of USManifestEntry
    per_dataset_status : dict  dataset_id -> "ok:N" | "skip:reason" | "fail:..."
    """
    all_entries: List[USManifestEntry] = []
    per_dataset_status: Dict[str, str] = {}

    for ds_id, adapter_cls in ADAPTER_REGISTRY.items():
        root_str = dataset_roots.get(ds_id)
        if not root_str:
            per_dataset_status[ds_id] = "skip:no_root_in_config"
            continue

        root = Path(root_str)
        if not root.exists():
            per_dataset_status[ds_id] = f"skip:root_not_found"
            log.debug("[%s] Root not found: %s", ds_id, root)
            continue

        try:
            adapter  = adapter_cls(root=str(root))
            entries: List[USManifestEntry] = []
            for e in adapter.iter_entries():
                entries.append(e)
                if len(entries) >= n_per_dataset:
                    break

            if entries:
                all_entries.extend(entries)
                per_dataset_status[ds_id] = f"ok:{len(entries)}"
                log.debug("[%s] %d entries collected", ds_id, len(entries))
            else:
                per_dataset_status[ds_id] = "skip:no_entries_yielded"
                log.warning("[%s] Adapter yielded no entries", ds_id)

        except Exception:
            per_dataset_status[ds_id] = "fail:adapter_error"
            log.error("[%s] Adapter error:\n%s", ds_id, traceback.format_exc())

    ok_count   = sum(1 for v in per_dataset_status.values() if v.startswith("ok"))
    skip_count = sum(1 for v in per_dataset_status.values() if v.startswith("skip"))
    fail_count = sum(1 for v in per_dataset_status.values() if v.startswith("fail"))
    log.info(
        "Manifest scan complete — %d datasets: %d ok, %d skip, %d fail | "
        "%d total entries",
        len(ADAPTER_REGISTRY), ok_count, skip_count, fail_count, len(all_entries),
    )
    return all_entries, per_dataset_status


def build_combined_manifest(
    dataset_roots: Dict[str, str],
    force: bool = False,
) -> Tuple[Path, Dict[str, str]]:
    """
    Build (or reuse) the combined smoke manifest.

    Always scans all adapters for per-dataset status; only rewrites the
    manifest file when missing or US_SMOKE_FORCE_REBUILD=1.

    Returns (manifest_path, per_dataset_status).
    """
    _SMOKE_OUT.mkdir(parents=True, exist_ok=True)

    all_entries, per_dataset_status = build_all_dataset_entries(dataset_roots)

    if _COMBINED_MANIFEST.exists() and not force:
        log.info(
            "Reusing existing manifest: %s  (%d entries from fresh scan)",
            _COMBINED_MANIFEST, len(all_entries),
        )
        return _COMBINED_MANIFEST, per_dataset_status

    if not all_entries:
        log.error("No entries found — verify dataset_roots in data_run1.yaml. "
                  "Continuing with an empty manifest (Phases 1-3 will be skipped).")
        # Write a placeholder so downstream code doesn't crash on missing file
        _COMBINED_MANIFEST.touch()
        return _COMBINED_MANIFEST, per_dataset_status

    with ManifestWriter(_COMBINED_MANIFEST) as w:
        for e in all_entries:
            w.write(e)

    log.info("Manifest written: %d entries → %s", len(all_entries), _COMBINED_MANIFEST)
    return _COMBINED_MANIFEST, per_dataset_status


# ── Config / DataModule ───────────────────────────────────────────────────────

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


# ── Utilities ─────────────────────────────────────────────────────────────────

def _to_dev(batch: dict, device: str) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def cosine_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = F.normalize(x.float(), dim=-1)
    y = F.normalize(y.float(), dim=-1)
    return 1.0 - (x * y).sum(dim=-1).mean()


def _downstream_collate(samples: list) -> dict:
    """Pad heterogeneous images to batch-max (H, W); produce boolean padding_mask."""
    patch_size = 16
    max_h = max(s["image"].shape[-2] for s in samples)
    max_w = max(s["image"].shape[-1] for s in samples)
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
        vh = h // patch_size
        vw = w // patch_size
        padding_mask[i, :vh, :vw] = True

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
                out[key] = vals
        except Exception:
            out[key] = vals
    return out


# ── Phase 1: Image SSL ────────────────────────────────────────────────────────

def phase1_smoke(dm: USFoundationDataModule, device: str) -> str:
    log.info("=== Phase 1: Image SSL (DINOv3-S) ===")
    dtype = torch.float32

    student = build_image_backbone("dinov3_s", dtype=dtype)
    teacher = build_image_backbone("dinov3_s", dtype=dtype)
    branch  = ImageBranch(student=student, teacher=teacher).to(device=device, dtype=dtype)
    opt     = torch.optim.AdamW(branch.student.parameters(), lr=1e-4)

    loader = dm.image_loader()
    branch.train()
    n = 0
    n_nan = 0
    for batch in loader:
        batch       = _to_dev(batch, device)
        global_crops = batch["global_crops"].to(dtype)
        local_crops  = batch.get("local_crops")

        opt.zero_grad()
        t_out = branch.forward_teacher(global_crops[:, 1])
        s_out = branch.forward_student(global_crops[:, 0])
        loss  = cosine_loss(s_out["cls"], t_out["cls"])

        if "patch_tokens" in s_out and "patch_tokens" in t_out:
            loss = loss + 0.5 * cosine_loss(
                s_out["patch_tokens"].mean(1),
                t_out["patch_tokens"].mean(1),
            )

        if local_crops is not None:
            local_crops = local_crops.to(dtype)
            for i in range(local_crops.shape[1]):
                s_loc = branch.forward_student(local_crops[:, i])
                loss  = loss + 0.3 * cosine_loss(s_loc["cls"], t_out["cls"])

        if not torch.isfinite(loss):
            log.error("Phase 1: non-finite loss at batch %d (%.6f) — skipping backward",
                      n + 1, loss.item())
            n_nan += 1
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(branch.student.parameters(), 1.0)
            opt.step()
            branch.update_teacher(momentum=0.9995)

        log.info("  Phase1 batch=%d  loss=%.4f  cls.shape=%s",
                 n + 1, loss.item(), tuple(s_out["cls"].shape))
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    if n == 0:
        log.error("Phase 1 FAIL — no image batches (empty manifest?)")
        return "FAIL"
    if n_nan == n:
        log.error("Phase 1 FAIL — all batches produced non-finite loss")
        return "FAIL"
    log.info("Phase 1 PASS (%d batches)", n)
    return "PASS"


# ── Phase 2: Video SSL ────────────────────────────────────────────────────────

def phase2_smoke(dm: USFoundationDataModule, device: str) -> str:
    log.info("=== Phase 2: Video SSL (V-JEPA2) ===")
    if not dm._video_entries:
        log.warning("Phase 2 SKIP — no video entries in manifest")
        return "SKIP"

    dtype  = torch.float32
    branch = build_video_branch(dtype=dtype, device=device)
    opt    = torch.optim.AdamW(branch.student.parameters(), lr=1e-4)

    loader = dm.video_loader()
    branch.train()
    n = 0
    n_nan = 0
    for batch in loader:
        batch     = _to_dev(batch, device)
        full_clip = batch["full_clips"].to(dtype)
        vis_clip  = batch["visible_clips"].to(dtype)
        tube_mask = batch.get("tube_masks")
        pad_mask  = batch.get("padding_masks")
        valid_fr  = batch.get("valid_frames")

        opt.zero_grad()
        t_out = branch.forward_teacher(full_clip, padding_mask=pad_mask,
                                       valid_frames=valid_fr)
        s_out = branch.forward_student(vis_clip, tube_mask=tube_mask,
                                       padding_mask=pad_mask,
                                       valid_frames=valid_fr)
        loss = cosine_loss(s_out["clip_cls"], t_out["clip_cls"])

        if not torch.isfinite(loss):
            log.error("Phase 2: non-finite loss at batch %d", n + 1)
            n_nan += 1
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(branch.student.parameters(), 1.0)
            opt.step()
            branch.update_teacher(momentum=0.9995)

        log.info("  Phase2 batch=%d  loss=%.4f  clip_cls.shape=%s",
                 n + 1, loss.item(), tuple(s_out["clip_cls"].shape))
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    if n == 0:
        log.warning("Phase 2 SKIP — video loader yielded no batches")
        return "SKIP"
    if n_nan == n:
        log.error("Phase 2 FAIL — all batches non-finite")
        return "FAIL"
    log.info("Phase 2 PASS (%d batches)", n)
    return "PASS"


# ── Phase 3: Cross-modal Alignment ───────────────────────────────────────────

def phase3_smoke(dm: USFoundationDataModule, device: str) -> str:
    log.info("=== Phase 3: Cross-modal Alignment ===")
    if not dm._video_entries:
        log.warning("Phase 3 SKIP — no video entries in manifest")
        return "SKIP"

    dtype = torch.float32

    img_student = build_image_backbone("dinov3_s", dtype=dtype)
    img_teacher = build_image_backbone("dinov3_s", dtype=dtype)
    img_branch  = ImageBranch(img_student, img_teacher).to(device=device, dtype=dtype)
    vid_branch  = build_video_branch(dtype=dtype, device=device)

    D_img = img_branch.embed_dim
    D_vid = vid_branch.student.hidden_size
    align_dim = 256

    cross      = CrossBranchDistillation(img_dim=D_img, vid_dim=D_vid,
                                         align_dim=align_dim).to(device=device, dtype=dtype)
    proto      = PrototypeHead(embed_dim=D_img, n_prototypes=64).to(device=device, dtype=dtype)
    vid_to_img = nn.Linear(D_vid, D_img, bias=False).to(device=device, dtype=dtype)

    params = (
        list(img_branch.student.parameters())
        + list(vid_branch.student.parameters())
        + list(cross.parameters())
        + list(proto.parameters())
        + list(vid_to_img.parameters())
    )
    opt = torch.optim.AdamW(params, lr=1e-4)

    img_branch.train(); vid_branch.train()
    cross.train(); proto.train(); vid_to_img.train()

    n = 0
    n_nan = 0
    for dual in dm.combined_loader():
        img_batch = _to_dev(dual.image_batch, device)
        vid_batch = _to_dev(dual.video_batch, device)

        global_crops = img_batch["global_crops"].to(dtype)
        full_clip    = vid_batch["full_clips"].to(dtype)
        vis_clip     = vid_batch["visible_clips"].to(dtype)
        tube_mask    = vid_batch.get("tube_masks")
        pad_mask_vid = vid_batch.get("padding_masks")

        opt.zero_grad()

        t_img   = img_branch.forward_teacher(global_crops[:, 1])
        s_img   = img_branch.forward_student(global_crops[:, 0])
        loss_img = cosine_loss(s_img["cls"], t_img["cls"])

        t_vid   = vid_branch.forward_teacher(full_clip, padding_mask=pad_mask_vid)
        s_vid   = vid_branch.forward_student(vis_clip, tube_mask=tube_mask,
                                             padding_mask=pad_mask_vid)
        loss_vid = cosine_loss(s_vid["clip_cls"], t_vid["clip_cls"])

        img_patches    = t_img["patch_tokens"]
        vid_tubes      = s_vid.get("tube_tokens", s_vid["clip_cls"].unsqueeze(1))
        loss_cross     = cross(img_patches, vid_tubes)
        vid_tubes_proj = vid_to_img(vid_tubes)
        loss_proto     = proto.consistency_loss(img_patches, vid_tubes_proj)

        loss = loss_img + loss_vid + loss_cross + 0.5 * loss_proto

        if not torch.isfinite(loss):
            log.error("Phase 3: non-finite loss at batch %d", n + 1)
            n_nan += 1
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            img_branch.update_teacher()
            vid_branch.update_teacher()

        log.info(
            "  Phase3 batch=%d  loss=%.4f  (img=%.3f vid=%.3f cross=%.3f proto=%.3f)",
            n + 1, loss.item(), loss_img.item(),
            loss_vid.item(), loss_cross.item(), loss_proto.item(),
        )
        n += 1
        if n >= N_SMOKE_BATCHES:
            break

    if n == 0:
        log.warning("Phase 3 SKIP — combined loader yielded no batches")
        return "SKIP"
    if n_nan == n:
        log.error("Phase 3 FAIL — all batches non-finite")
        return "FAIL"
    log.info("Phase 3 PASS (%d batches)", n)
    return "PASS"


# ── Phase 4: All Finetune Experiments ────────────────────────────────────────

# (result_key, cls_name, module, dataset_id, finetune_yaml_stem)
_STANDARD_EXPERIMENTS = [
    ("busi",       "BUSIFinetune",       "finetune.experiments.busi",              "BUSI",                    "busi"),
    ("camus",      "CAMUSFinetune",      "finetune.experiments.camus",             "CAMUS",                   "camus"),
    ("echonet",    "EchoNetFinetune",    "finetune.experiments.echonet",           "EchoNet-Dynamic",         "echonet"),
    ("tn3k",       "TN3KFinetune",       "finetune.experiments.tn3k",              "TN3K",                    "tn3k"),
    ("busbra",     "BUSBRAFinetune",     "finetune.experiments.busbra",            "BUS-BRA",                 "busbra"),
    ("cardiacudc", "CardiacUDCFinetune", "finetune.experiments.cardiacudc",        "CardiacUDC",              "cardiacudc"),
    ("echocp",     "EchoCPFinetune",     "finetune.experiments.echocp",            "EchoCP",                  "echocp"),
]

_ECHONET_EXPERIMENTS = [
    ("echonet_ped", "EchoNetPediatricFinetune", "finetune.experiments.echonet_pediatric",
     "EchoNet-Pediatric", "echonet_pediatric"),
    ("echonet_lvh", "EchoNetLVHFinetune",       "finetune.experiments.echonet_lvh",
     "EchoNet-LVH", "echonet_lvh"),
    ("mimic_lvvol", "MIMICLVVolFinetune",        "finetune.experiments.mimic_lvvol",
     "MIMIC-IV-Echo-LVVol-A4C", "mimic_lvvol"),
]

_LUS_EXPERIMENTS = [
    ("lus_patient", "LUSPatientFinetune", "finetune.experiments.lus_patient", "lus_patient"),
    ("lus_video",   "LUSVideoFinetune",   "finetune.experiments.lus_video",   "lus_video"),
]


def _load_finetune_config(yaml_stem: str):
    """Load finetune hyperparameters from configs/finetune/{yaml_stem}.yaml."""
    from finetune.base import FinetuneConfig

    cfg_path = _FINETUNE_CFG / f"{yaml_stem}.yaml"
    if cfg_path.exists():
        cfg = FinetuneConfig.from_yaml(str(cfg_path))
    else:
        log.warning("Finetune config not found: %s — using defaults", cfg_path)
        cfg = FinetuneConfig()
    # Smoke overrides — keep runs fast
    cfg.max_epochs  = 1
    cfg.batch_size  = 2
    cfg.num_workers = 0
    cfg.patience    = 1
    return cfg


def _forward_smoke_batch(exp, batch: dict, device: str, result_key: str, batch_idx: int) -> None:
    """Run one minimal forward pass for a finetune experiment batch."""
    if "image" in batch and isinstance(batch["image"], torch.Tensor):
        imgs = batch["image"].to(device=device, dtype=torch.float32)
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        with torch.no_grad():
            feats = exp.encoder.encode_image(imgs)
        log.info("[%s] batch=%d encode_image OK — cls.shape=%s",
                 result_key, batch_idx, tuple(feats["cls"].shape))
    elif "clips" in batch:
        # LUS patient MIL — encode first patient's clips
        clips = batch["clips"][0]
        if isinstance(clips, torch.Tensor):
            with torch.no_grad():
                feats = exp.encoder.encode_video(clips.to(device=device, dtype=torch.float32))
            log.info("[%s] batch=%d encode_video OK — clip_cls.shape=%s",
                     result_key, batch_idx, tuple(feats["clip_cls"].shape))
    elif "clip" in batch and isinstance(batch["clip"], torch.Tensor):
        clip = batch["clip"].to(device=device, dtype=torch.float32)
        if clip.dim() == 4:
            clip = clip.unsqueeze(0)
        with torch.no_grad():
            feats = exp.encoder.encode_video(clip)
        log.info("[%s] batch=%d encode_video OK — clip_cls.shape=%s",
                 result_key, batch_idx, tuple(feats["clip_cls"].shape))


def _smoke_experiment(
    result_key:  str,
    cls_name:    str,
    module:      str,
    data_root:   Optional[str],
    yaml_stem:   str,
    img_branch,
    vid_branch,
    device:      str,
    output_dir:  Path,
    extra_kwargs: dict = None,
) -> str:
    """
    Instantiate one finetune experiment and verify setup + dataloader.

    Returns "PASS" | "FAIL" | "SKIP".
    """
    import importlib

    if not data_root or not Path(data_root).exists():
        log.warning("[%s] data_root not found (%r) — SKIP", result_key, data_root)
        return "SKIP"

    try:
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        cfg = _load_finetune_config(yaml_stem)

        kwargs = dict(data_root=data_root, output_dir=str(output_dir / result_key), cfg=cfg)
        if extra_kwargs:
            kwargs.update(extra_kwargs)

        exp = cls(**kwargs)
        exp.setup(img_branch=img_branch, device=device, vid_branch=vid_branch)

        loader = exp.build_dataloader("train")
        n_batches = 0
        for batch in loader:
            log.info("[%s] batch=%d loaded — keys: %s",
                     result_key, n_batches + 1, list(batch.keys()))
            _forward_smoke_batch(exp, batch, device, result_key, n_batches + 1)
            n_batches += 1
            if n_batches >= N_SMOKE_BATCHES:
                break

        if n_batches == 0:
            log.error("[%s] FAIL — dataloader yielded no batches", result_key)
            return "FAIL"

        log.info("[%s] PASS (%d batches)", result_key, n_batches)
        return "PASS"

    except Exception:
        log.error("[%s] FAIL:\n%s", result_key, traceback.format_exc())
        return "FAIL"


def _smoke_lus_experiment(
    result_key:   str,
    cls_name:     str,
    module:       str,
    yaml_stem:    str,
    benin_root:   Optional[str],
    rsa_root:     Optional[str],
    img_branch,
    vid_branch,
    device:       str,
    output_dir:   Path,
) -> str:
    """Smoke one dual-root LUS experiment (Benin + RSA)."""
    import importlib

    has_benin = benin_root and Path(benin_root).exists()
    has_rsa   = rsa_root   and Path(rsa_root).exists()
    if not has_benin and not has_rsa:
        log.warning("[%s] Neither Benin nor RSA root found — SKIP", result_key)
        return "SKIP"

    benin_root = benin_root or ""
    rsa_root   = rsa_root   or ""

    try:
        mod = importlib.import_module(module)
        cls = getattr(mod, cls_name)
        cfg = _load_finetune_config(yaml_stem)
        exp = cls(
            data_root_benin=benin_root,
            data_root_rsa=rsa_root,
            output_dir=str(output_dir / result_key),
            cfg=cfg,
        )
        exp.setup(img_branch=img_branch, device=device, vid_branch=vid_branch)
        loader = exp.build_dataloader("train")
        n_batches = 0
        for batch in loader:
            log.info("[%s] batch=%d loaded — keys: %s",
                     result_key, n_batches + 1, list(batch.keys()))
            _forward_smoke_batch(exp, batch, device, result_key, n_batches + 1)
            n_batches += 1
            if n_batches >= N_SMOKE_BATCHES:
                break

        if n_batches == 0:
            log.error("[%s] FAIL — dataloader yielded no batches", result_key)
            return "FAIL"

        log.info("[%s] PASS (%d batches)", result_key, n_batches)
        return "PASS"
    except Exception:
        log.error("[%s] FAIL:\n%s", result_key, traceback.format_exc())
        return "FAIL"


def finetune_experiments_smoke(
    dataset_roots: Dict[str, str],
    device:        str,
) -> Dict[str, str]:
    """
    Run all 11 finetune experiments with a minimal Ultatron backbone.

    Returns a dict experiment_key -> "PASS" | "FAIL" | "SKIP".
    """
    log.info("=== Phase 4: Finetune Experiments Smoke ===")

    dtype = torch.float32
    log.info("Building DINOv3-S image backbone …")
    img_student  = build_image_backbone("dinov3_s", dtype=dtype)
    img_teacher  = build_image_backbone("dinov3_s", dtype=dtype)
    img_branch   = ImageBranch(img_student, img_teacher).to(device=device, dtype=dtype)
    for p in img_branch.parameters():
        p.requires_grad_(False)
    img_branch.eval()

    log.info("Building V-JEPA2-L video backbone …")
    vid_branch = build_video_branch(dtype=dtype, device=device)
    for p in vid_branch.parameters():
        p.requires_grad_(False)
    vid_branch.eval()

    out_dir = _SMOKE_OUT / "finetune"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, str] = {}

    for result_key, cls_name, module, ds_id, yaml_stem in (
        _STANDARD_EXPERIMENTS + _ECHONET_EXPERIMENTS
    ):
        root = dataset_roots.get(ds_id) or ""
        results[result_key] = _smoke_experiment(
            result_key=result_key,
            cls_name=cls_name,
            module=module,
            data_root=root,
            yaml_stem=yaml_stem,
            img_branch=img_branch,
            vid_branch=vid_branch,
            device=device,
            output_dir=out_dir,
        )

    # Dual-root LUS experiments
    benin_root = dataset_roots.get("Benin-LUS") or dataset_roots.get("BeninVideos") or ""
    rsa_root   = dataset_roots.get("RSA-LUS") or ""

    for result_key, cls_name, module, yaml_stem in _LUS_EXPERIMENTS:
        results[result_key] = _smoke_lus_experiment(
            result_key=result_key,
            cls_name=cls_name,
            module=module,
            yaml_stem=yaml_stem,
            benin_root=benin_root,
            rsa_root=rsa_root,
            img_branch=img_branch,
            vid_branch=vid_branch,
            device=device,
            output_dir=out_dir,
        )

    return results


def phase4_smoke(dataset_roots: Dict[str, str], device: str) -> Dict[str, str]:
    return finetune_experiments_smoke(dataset_roots, device)


# ── Phase 5: Ablation Backbones ───────────────────────────────────────────────

_ABLATION_BACKBONE_SPECS: List[Tuple[str, dict]] = [
    ("resnet50",   {"type": "standard",   "key": "resnet50",  "variant": "resnet50"}),
    ("vit_b_16",   {"type": "standard",   "key": "vit_b_16",  "variant": "vit_b_16"}),
    ("dinov3_b",   {"type": "dinov3",     "key": "dinov3_b",  "variant": "dinov3_b"}),
    ("biomedclip", {"type": "biomedclip", "key": "biomedclip"}),
    ("vjepa2_l",   {"type": "vjepa",      "key": "vjepa2_l",  "variant": "vjepa2_l"}),
    ("usfm", {
        "type": "usfm",
        "key": "usfm",
        "checkpoint": ablation_weight_path("USFM_latest.pth", "US_USFM_CHECKPOINT"),
        "embed_dim": 768,
    }),
    ("echocare", {
        "type": "echocare",
        "key": "echocare",
        "checkpoint": ablation_weight_path("echocare_encoder.pth", "US_ECHOCARE_CHECKPOINT"),
        "embed_dim": 2048,
        "is_video": False,
    }),
    ("openus", {
        "type": "openus",
        "key": "openus",
        "checkpoint": ablation_weight_path("openus_cpt0150.pth", "US_OPENUS_CHECKPOINT"),
        "embed_dim": 768,
        "vmamba_checkpoint": ablation_weight_path(
            "vssm_small_0229_ckpt_epoch_222.pth", "US_OPENUS_VMAMBA_CHECKPOINT"
        ),
    }),
]


def _dummy_image_forward(encoder, device: str) -> None:
    """Run a dummy 2×3×224×224 forward pass through an encoder."""
    encoder.eval()
    encoder.to(device)
    dummy = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        out = encoder.encode_image(dummy)
    cls = out["cls"]
    log.info("  dummy forward OK — cls.shape=%s  dtype=%s",
             tuple(cls.shape), cls.dtype)


def _backbone_needs_checkpoint(spec: dict) -> bool:
    return spec.get("type") in ("usfm", "echocare", "openus")


def phase5_ablation_backbones_smoke(device: str) -> Dict[str, str]:
    """
    Load and smoke-test all ablation/comparison backbones via build_encoder().

    Returns a dict backbone_key -> "PASS" | "FAIL" | "SKIP".
    """
    log.info("=== Phase 5: Ablation Backbones Smoke ===")

    results: Dict[str, str] = {}

    for key, spec in _ABLATION_BACKBONE_SPECS:
        log.info("--- %s ---", key)
        try:
            if _backbone_needs_checkpoint(spec):
                ckpt = spec.get("checkpoint", "")
                if not ckpt or not Path(ckpt).exists():
                    log.warning("[%s] Checkpoint not found at %r — SKIP", key, ckpt)
                    results[key] = "SKIP"
                    continue

            enc = build_encoder(spec, device=device)
            _dummy_image_forward(enc, device)
            log.info("[%s] PASS", key)
            results[key] = "PASS"
        except Exception:
            log.error("[%s] FAIL:\n%s", key, traceback.format_exc())
            results[key] = "FAIL"

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    device = _auto_device()
    log.info("Device: %s", device)

    force_rebuild = os.environ.get("US_SMOKE_FORCE_REBUILD", "0") == "1"

    # Load dataset roots from data_run1.yaml
    log.info("Loading dataset roots from %s …", _DATA_CONFIG)
    dataset_roots = _load_all_dataset_roots()

    # Build combined manifest
    log.info("Building combined smoke manifest …")
    _, dataset_status = build_combined_manifest(dataset_roots, force=force_rebuild)

    # DataModule
    cfg = load_smoke_config()
    dm  = build_datamodule(cfg)

    # Phase results
    phase_results:     Dict[str, str] = {}
    experiment_results: Dict[str, str] = {}
    ablation_results:  Dict[str, str] = {}

    def _run_phase(name: str, fn, *args):
        skip_var = f"US_SKIP_{name.upper()}"
        if os.environ.get(skip_var, "0") == "1":
            log.info("Skipping %s (%s=1)", name, skip_var)
            phase_results[name] = "SKIP"
            return "SKIP"
        try:
            result = fn(*args)
            phase_results[name] = result if result else "PASS"
        except Exception:
            phase_results[name] = "FAIL"
            log.error("%s FAILED:\n%s", name, traceback.format_exc())
        return phase_results[name]

    _run_phase("PHASE1", phase1_smoke, dm, device)
    _run_phase("PHASE2", phase2_smoke, dm, device)
    _run_phase("PHASE3", phase3_smoke, dm, device)

    # Phase 4 — returns per-experiment dict
    if os.environ.get("US_SKIP_PHASE4", "0") == "1":
        log.info("Skipping PHASE4 (US_SKIP_PHASE4=1)")
        phase_results["PHASE4"] = "SKIP"
    else:
        try:
            experiment_results = phase4_smoke(dataset_roots, device)
            fail_count = sum(1 for v in experiment_results.values() if v == "FAIL")
            phase_results["PHASE4"] = "FAIL" if fail_count > 0 else "PASS"
        except Exception:
            phase_results["PHASE4"] = "FAIL"
            log.error("PHASE4 outer FAIL:\n%s", traceback.format_exc())

    # Phase 5 — returns per-backbone dict
    if os.environ.get("US_SKIP_PHASE5", "0") == "1":
        log.info("Skipping PHASE5 (US_SKIP_PHASE5=1)")
        phase_results["PHASE5"] = "SKIP"
    else:
        try:
            ablation_results = phase5_ablation_backbones_smoke(device)
            fail_count = sum(1 for v in ablation_results.values() if v == "FAIL")
            phase_results["PHASE5"] = "FAIL" if fail_count > 0 else "PASS"
        except Exception:
            phase_results["PHASE5"] = "FAIL"
            log.error("PHASE5 outer FAIL:\n%s", traceback.format_exc())

    # ── Summary ───────────────────────────────────────────────────────────────
    W = 65

    def _status_label(s: str) -> str:
        return {"PASS": "PASS", "FAIL": "FAIL", "SKIP": "SKIP"}.get(s, s)

    print("\n" + "=" * W)
    print("SMOKE SUMMARY")
    print("=" * W)

    for phase in ["PHASE1", "PHASE2", "PHASE3"]:
        print(f"{phase:<22}  {_status_label(phase_results.get(phase, 'N/A'))}")
    print(f"{'PHASE4/finetune':<22}  {_status_label(phase_results.get('PHASE4', 'N/A'))}"
          f"    (see per-experiment below)")
    print(f"{'PHASE5/backbones':<22}  {_status_label(phase_results.get('PHASE5', 'N/A'))}"
          f"    (see per-backbone below)")

    # Dataset manifest coverage
    ok_list      = [(k, v) for k, v in dataset_status.items() if v.startswith("ok")]
    no_root_list = [(k, v) for k, v in dataset_status.items()
                    if v == "skip:no_root_in_config"]
    empty_list   = [(k, v) for k, v in dataset_status.items()
                    if v in ("skip:no_entries_yielded", "skip:root_not_found")]
    error_list   = [(k, v) for k, v in dataset_status.items() if v.startswith("fail")]
    total_entries = sum(
        int(v.split(":")[1])
        for _, v in ok_list
        if ":" in v and v.split(":")[1].isdigit()
    )

    print(f"\n--- Dataset manifest coverage ({len(ADAPTER_REGISTRY)} datasets) ---")
    print(f"  ok:       {len(ok_list):>3}   ({total_entries} entries total)")
    print(f"  no_root:  {len(no_root_list):>3}")
    print(f"  empty:    {len(empty_list):>3}")
    print(f"  error:    {len(error_list):>3}")
    if error_list:
        print("  Failed datasets:")
        for k, v in sorted(error_list):
            print(f"    {k:<45}  {v}")

    print(f"\n--- Finetune experiments ({len(experiment_results) or 11}) ---")
    if experiment_results:
        for exp_key, status in sorted(experiment_results.items()):
            print(f"  {exp_key:<18}  {_status_label(status)}")
    else:
        print(f"  {_status_label(phase_results.get('PHASE4', 'N/A'))}")

    print(f"\n--- Ablation backbones ({len(ablation_results) or 7}) ---")
    if ablation_results:
        for bb_key, status in sorted(ablation_results.items()):
            print(f"  {bb_key:<18}  {_status_label(status)}")
    else:
        print(f"  {_status_label(phase_results.get('PHASE5', 'N/A'))}")

    total_fail = sum(
        1 for v in {**phase_results, **experiment_results, **ablation_results}.values()
        if v == "FAIL"
    )
    print(f"\n--- Overall: {total_fail} FAIL ---")
    print("=" * W)
    if total_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
