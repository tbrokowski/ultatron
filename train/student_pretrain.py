"""
train/student_pretrain.py
=========================
Production entry point for the single-student (Hiera) pretraining pipeline
and the WP5 POCUS strong-scaling benches (E0–E6).

Moved out of ``tests.dataset_adapters.student_training_smoke`` so reviewers
read a ``train/`` script.  The tests module remains a compatibility shim.

Usage (inside the EDF container):

    python -m train.student_pretrain
    python -m train.student_pretrain --resume
    python -m train.student_pretrain --bench-stage 1 --bench-window --per-step-timing --no-ckpt

Multi-GPU:

    torchrun --nproc_per_node=4 -m train.student_pretrain
    bash scripts/submit_student_pretrain.sh
    bash scripts/pocus/submit_encoder.sh E1 --nodes 1

Benchmark flags (spec §3.1)
---------------------------
    --bench-stage k       pin curriculum stage k (1–4)
    --max-steps N
    --no-ckpt             skip checkpoint writes
    --ckpt-probe          write+read one resumable checkpoint and time it
    --per-step-timing     JSONL step breakdown (spec §5)
    --loader-only         iterate the data-loader with no model
    --bench-window        4.1 warm-up / measurement / 3 % stopping rule

Environment overrides
---------------------
    US_SMOKE_DEVICE, US_SMOKE_FORCE_REBUILD, US_STUDENT_RESUME,
    US_STUDENT_RESUME_STAGE_FRACS, US_STUDENT_RESUME_CKPT,
    US_STUDENT_HIERA_VARIANT, US_STUDENT_DINO_KEY, US_STUDENT_VJEPA_KEY,
    US_SMOKE_STEPS / US_STUDENT_STEPS, US_STUDENT_CONFIG,
    US_STUDENT_LR, US_STUDENT_WARMUP, US_STUDENT_CKPT_DIR,
    US_STUDENT_STEPS_JSONL, US_STUDENT_NUM_WORKERS
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast

_ROOT = Path(__file__).parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.adapters import ADAPTER_REGISTRY
from data.pipeline.collators import ImageSSLCollator, PairedSSLCollator, VideoSSLCollator
from data.pipeline.datamodule import USFoundationDataModule
from data.pipeline.student_datamodule import (
    StudentDataConfig,
    StudentDataModule,
    StudentMixedCollator,
    _ddp_broadcast_sample_type,
)
from data.pipeline.transforms import build_transform_configs
from data.schema.manifest import ManifestWriter, USManifestEntry, load_manifest
from models.branches.shared import PrototypeHead, ema_update
from models.losses.proto_loss import ProtoQueue
from models.heads.hierarchical_seg import build_hierarchical_seg_head
from models.student.student_config import StudentModelConfig, build_fusion_target_builder, build_student_encoder
from models.student.teacher_wrappers import FrozenDINOTeacher, FrozenVJEPATeacher
from data.infra.cscs_paths import configure_hf_environment
from finetune.backbones.paths import student_smoke_checkpoints_dir
from train.student_phase_steps import (
    _stage_bounds_from_fracs,
    student_stage1_step,
    student_stage2_step,
    student_stage3_step,
    student_stage4_divergence_step,
)
from train.alp import HardnessFeedback, configure_alp_cache
from train.bench import (
    BenchWindowController,
    CudaStepTimer,
    StepTiming,
    apply_bench_overrides,
    count_batch_items,
    gpu_count_from_env,
    peak_mem_gb,
    resolve_grad_accum,
    set_global_seed,
)
from train.ddp_hooks import try_register_allreduce_timer


class _Rank0Filter(logging.Filter):
    """Drop log records on non-zero ranks (spec §3.1: no 128× duplicated lines)."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return _is_main()


def _setup_logging() -> logging.Logger:
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logger = logging.getLogger("student_training")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    rank0 = _Rank0Filter()
    err_handler = logging.StreamHandler(sys.stderr)
    err_handler.setFormatter(fmt)
    err_handler.addFilter(rank0)
    logger.addHandler(err_handler)
    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setFormatter(fmt)
    out_handler.addFilter(rank0)
    logger.addHandler(out_handler)
    logger.propagate = False
    return logger


log = _setup_logging()


def _announce(msg: str) -> None:
    """Write a milestone line to stdout (.out) on rank 0."""
    if _is_main():
        print(msg, flush=True)

_DATA_CONFIG = _ROOT / "configs" / "run1" / "data_run1.yaml"
_SMOKE_CFG = _ROOT / "configs" / "smoke" / "student_smoke.yaml"
_PRETRAIN_CFG = _ROOT / "configs" / "student" / "student_pretrain.yaml"
_SMOKE_OUT = _ROOT / "dataset_exploration_outputs" / "smoke"
_COMBINED_MANIFEST = _SMOKE_OUT / "student_combined_manifest.jsonl"


# ── Distributed helpers ───────────────────────────────────────────────────────

def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _is_main() -> bool:
    return _rank() == 0


def _init_dist() -> None:
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return
    if dist.is_initialized():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    log.info("DDP rank=%d world_size=%d local_rank=%d", _rank(), _world_size(), local_rank)


def _barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _maybe_ddp(module: nn.Module, find_unused: bool = True) -> nn.Module:
    if not _is_ddp():
        return module
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return nn.parallel.DistributedDataParallel(
        module,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=find_unused,
    )


def _unwrap(module: nn.Module) -> nn.Module:
    return module.module if isinstance(module, nn.parallel.DistributedDataParallel) else module


def _adapt_student_state_dict(ckpt_sd: dict, model_sd: dict) -> dict:
    """
    Allow resume when only ``temporal_pos_enc`` table size changed (max_frames).

    Copies the overlapping prefix of learned frame indices; leaves any new
    slots at their current (initialized) values.
    """
    out = dict(ckpt_sd)
    key = "temporal_pos_enc.weight"
    ckpt_w = ckpt_sd.get(key)
    model_w = model_sd.get(key)
    if (
        ckpt_w is None
        or model_w is None
        or ckpt_w.shape == model_w.shape
        or ckpt_w.ndim != 2
        or ckpt_w.shape[1] != model_w.shape[1]
    ):
        return out
    n = min(ckpt_w.shape[0], model_w.shape[0])
    adapted = model_w.clone()
    adapted[:n] = ckpt_w[:n]
    out[key] = adapted
    log.warning(
        "Adapted %s for resume: ckpt %s → model %s (copied %d/%d rows)",
        key, tuple(ckpt_w.shape), tuple(model_w.shape), n, model_w.shape[0],
    )
    return out


def _adapt_seg_head_state_dict(ckpt_sd: dict, model_sd: dict) -> dict:
    """
    Migrate UPerNetDecoder state-dict across architecture changes.

    Handles two backwards-incompatible refactors:

    1. use_adapters / use_attention_gates added (adapters.*, gates.*):
       The checkpoint pre-dates these modules.  New keys are filled from
       the current model's initialized values (zero-init adapters = identity,
       so this is a safe warm-start).

    2. use_aspp refactored fusion from plain nn.Sequential to ASPPFusion:
       Old keys:  fusion.0.*  (Conv2d fpn*4→fpn),  fusion.1.*  (BN),
                  fusion.3.*  (_ConvBnGelu fpn→fpn)
       New keys:  fusion.branch1x1.*, fusion.branches_dilated.*,
                  fusion.global_branch.*, fusion.bottleneck.*
       The 3×3 refinement tail (fusion.3.* → fusion.bottleneck.3.*) has
       identical channel dimensions and is transferred; everything else in
       the ASPP uses the model's fresh random initialisation.

    Strategy:
      - Seed from model_sd (ensures every required key is present).
      - Overlay all ckpt keys that still exist with the same shape.
      - Remap the compatible fusion.3.* suffix keys.
    """
    out = dict(model_sd)  # all current model keys, fresh-initialised

    # 1. Copy every checkpoint key that still exists with a matching shape.
    copied, skipped_shape, skipped_missing = [], [], []
    for k, v in ckpt_sd.items():
        if k in model_sd:
            if v.shape == model_sd[k].shape:
                out[k] = v
                copied.append(k)
            else:
                skipped_shape.append(k)
        else:
            skipped_missing.append(k)

    # 2. Remap old plain-sequential fusion tail → ASPP bottleneck tail.
    #    fusion.3.{0,1}.* maps 1-to-1 because both are _ConvBnGelu(fpn, fpn).
    remap = {
        "fusion.3.0.weight":            "fusion.bottleneck.3.0.weight",
        "fusion.3.1.weight":            "fusion.bottleneck.3.1.weight",
        "fusion.3.1.bias":              "fusion.bottleneck.3.1.bias",
        "fusion.3.1.running_mean":      "fusion.bottleneck.3.1.running_mean",
        "fusion.3.1.running_var":       "fusion.bottleneck.3.1.running_var",
        "fusion.3.1.num_batches_tracked": "fusion.bottleneck.3.1.num_batches_tracked",
    }
    n_remapped = 0
    for old_k, new_k in remap.items():
        if old_k in ckpt_sd and new_k in model_sd:
            v = ckpt_sd[old_k]
            if v.shape == model_sd[new_k].shape:
                out[new_k] = v
                n_remapped += 1

    # 3. Summarise what happened so the user can see the migration clearly.
    fresh = [k for k in model_sd if k not in out or (k not in copied and k not in remap.values())]
    fresh_new = [k for k in model_sd if k not in ckpt_sd and k not in remap.values()]
    if skipped_missing or n_remapped or fresh_new:
        log.warning(
            "seg_head checkpoint migration: "
            "%d keys copied directly, "
            "%d fusion.3.* keys remapped → fusion.bottleneck.3.*, "
            "%d new keys fresh-initialised (adapters/gates/ASPP branches), "
            "%d old keys dropped (shape mismatch or removed: %s)",
            len(copied), n_remapped, len(fresh_new),
            len(skipped_missing) + len(skipped_shape),
            (skipped_missing + skipped_shape)[:6],
        )

    return out


def _adapt_optimizer_state_dict(
    ckpt_opt: dict,
    optimizer: torch.optim.Optimizer,
    prefix_modules: tuple[nn.Module, ...],
) -> dict:
    """
    Resume optimizer state when trainable parameter count changed.

    Student, fusion, and proto precede seg_head in the AdamW param list.  When
    only seg_head grew (e.g. adapters/gates/ASPP), copy optimizer moments for
    the shared prefix and leave new seg_head params at their fresh init.
    """
    cur = optimizer.state_dict()
    if not ckpt_opt.get("param_groups") or not cur.get("param_groups"):
        return ckpt_opt

    if len(ckpt_opt["param_groups"]) != len(cur["param_groups"]):
        raise ValueError("optimizer param group count mismatch")

    ckpt_pg = ckpt_opt["param_groups"][0]
    cur_pg = cur["param_groups"][0]
    n_ckpt = len(ckpt_pg["params"])
    n_cur = len(cur_pg["params"])
    if n_ckpt == n_cur:
        return ckpt_opt

    n_shared = sum(
        len(list(_unwrap(m).parameters()))
        for m in prefix_modules
    )
    if n_ckpt < n_shared:
        raise ValueError(
            f"checkpoint optimizer has fewer params ({n_ckpt}) than shared "
            f"prefix ({n_shared})"
        )

    new_sd = copy.deepcopy(cur)
    ckpt_state = ckpt_opt["state"]
    for i in range(n_shared):
        old_pid = ckpt_pg["params"][i]
        new_pid = cur_pg["params"][i]
        if old_pid in ckpt_state:
            new_sd["state"][new_pid] = {
                k: v.clone() if torch.is_tensor(v) else v
                for k, v in ckpt_state[old_pid].items()
            }

    for key, val in ckpt_pg.items():
        if key != "params":
            new_sd["param_groups"][0][key] = val

    log.warning(
        "Optimizer partial resume: restored AdamW state for %d shared params; "
        "%d seg_head params reset (architecture changed)",
        n_shared,
        n_cur - n_shared,
    )
    return new_sd


# ── Config / manifest ─────────────────────────────────────────────────────────

_LOADED_CONFIG_PATH: Optional[Path] = None


def _load_config() -> dict:
    global _LOADED_CONFIG_PATH
    cfg_path = os.environ.get("US_STUDENT_CONFIG")
    if cfg_path:
        path = Path(cfg_path)
    elif _PRETRAIN_CFG.exists() and os.environ.get("US_STUDENT_MODE") == "pretrain":
        path = _PRETRAIN_CFG
    else:
        path = _SMOKE_CFG
    if not path.is_absolute():
        path = _ROOT / path
    _LOADED_CONFIG_PATH = path
    with open(path) as f:
        cfg = yaml.safe_load(f)

    tcfg = cfg.setdefault("training", {})
    if steps := os.environ.get("US_STUDENT_STEPS") or os.environ.get("US_SMOKE_STEPS"):
        tcfg["total_steps"] = int(steps)
    if lr := os.environ.get("US_STUDENT_LR"):
        tcfg["lr"] = float(lr)
    if warmup := os.environ.get("US_STUDENT_WARMUP"):
        tcfg["warmup_steps"] = int(warmup)
    if nw := os.environ.get("US_STUDENT_NUM_WORKERS"):
        nw = int(nw)
        cfg.setdefault("loaders", {})["num_workers"] = nw
        cfg.setdefault("student_data", {})["num_workers"] = nw
    return cfg


def _is_pretrain_mode(cfg: dict) -> bool:
    mode = os.environ.get("US_STUDENT_MODE", "").strip().lower()
    if mode == "pretrain":
        return True
    if mode == "smoke":
        return False
    # Legacy: pretrain yaml without a smoke test manifest section.
    return "pretrain" in cfg and "smoke" not in cfg


def _load_smoke_config() -> dict:
    return _load_config()


def _resolve_hf_cache(cfg: dict) -> str:
    """Capstor scratch/store HF cache; override via pretrain/smoke hf_cache_dir or US_HF_CACHE_DIR."""
    explicit = (
        os.environ.get("US_HF_CACHE_DIR")
        or cfg.get("pretrain", {}).get("hf_cache_dir")
        or cfg.get("smoke", {}).get("hf_cache_dir")
        or cfg.get("model", {}).get("student", {}).get("hiera_hf_cache_dir")
    )
    path = configure_hf_environment(explicit)
    hf_cache = str(path)
    cfg.setdefault("model", {}).setdefault("student", {})["hiera_hf_cache_dir"] = hf_cache
    if _is_main():
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        _announce(f"HF cache → {hf_cache}")
        if token:
            log.info("HF auth: token from environment")
        elif token_file.exists():
            log.info("HF auth: token from %s", token_file)
        else:
            log.info("HF auth: none — using cached weights only")
    return hf_cache


def _resolve_ckpt_dir(cfg: dict) -> Path:
    """Capstor store dir (override via env)."""
    if env := os.environ.get("US_STUDENT_CKPT_DIR") or os.environ.get("US_STUDENT_SMOKE_CKPT_DIR"):
        return Path(env)
    for section in ("pretrain", "smoke"):
        raw = cfg.get(section, {}).get("ckpt_dir")
        if raw:
            p = Path(raw)
            return p if p.is_absolute() else _ROOT / raw
    return student_smoke_checkpoints_dir()


def _checkpoint_step(path: Path) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt["step"])


def _checkpoint_last_finite_step(ckpt: dict) -> int:
    """Last optimizer step with a finite loss (explicit field or inferred)."""
    if "last_finite_step" in ckpt:
        return int(ckpt["last_finite_step"])
    step = int(ckpt["step"])
    n_fin = int(ckpt.get("n_finite", 0))
    n_non = int(ckpt.get("n_nonfinite", 0))
    if n_non == 0 and n_fin > 0:
        return step
    return max(-1, n_fin - 1)


def _checkpoint_is_healthy(ckpt: dict) -> bool:
    """
    True when the checkpoint reflects real training progress.

    Rejects bogus latest.pt (step >> attempts) and runs where most loop
    iterations skipped the optimizer (high nonfinite rate).
    """
    step = int(ckpt["step"])
    n_fin = int(ckpt.get("n_finite", 0))
    n_non = int(ckpt.get("n_nonfinite", 0))
    attempts = n_fin + n_non
    if step > attempts + 1:
        return False
    if attempts == 0:
        return step == 0
    if n_non > max(10, step // 10):
        return False
    return n_fin >= step


def _find_resume_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Pick the best checkpoint to resume from.

    Prefers healthy checkpoints (finite updates track loop step) with the
    highest last-finite step.  Falls back to best-effort when all are degraded.

    Considers latest.pt, stage{N}_end.pt (N=1..4), and step_*.pt snapshots.
    """
    if not ckpt_dir.is_dir():
        return None

    candidates: List[Path] = []
    latest = ckpt_dir / "latest.pt"
    if latest.is_file():
        candidates.append(latest)
    for stage in (4, 3, 2, 1):
        p = ckpt_dir / f"stage{stage}_end.pt"
        if p.is_file():
            candidates.append(p)
    candidates.extend(sorted(ckpt_dir.glob("step_*.pt"), reverse=True))

    loaded: List[Tuple[Path, dict]] = []
    for path in candidates:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            int(ckpt["step"])
        except Exception:
            log.warning("Skipping unreadable checkpoint %s", path)
            continue
        loaded.append((path, ckpt))

    if not loaded:
        return None

    healthy = [(p, c) for p, c in loaded if _checkpoint_is_healthy(c)]
    pool = healthy if healthy else loaded
    if not healthy:
        log.warning(
            "No healthy checkpoint in %s — selecting by last finite step",
            ckpt_dir,
        )

    best_path, best_ckpt = max(
        pool,
        key=lambda item: (_checkpoint_last_finite_step(item[1]), int(item[1]["step"])),
    )
    step = int(best_ckpt["step"])
    n_fin = int(best_ckpt.get("n_finite", 0))
    n_non = int(best_ckpt.get("n_nonfinite", 0))
    last_fin = _checkpoint_last_finite_step(best_ckpt)
    if not _checkpoint_is_healthy(best_ckpt):
        log.warning(
            "Resuming from degraded checkpoint %s: step=%d last_finite=%d "
            "n_finite=%d n_nonfinite=%d",
            best_path.name, step, last_fin, n_fin, n_non,
        )
    elif last_fin < step:
        log.info(
            "Selected %s: step=%d last_finite=%d (n_finite=%d n_nonfinite=%d)",
            best_path.name, step, last_fin, n_fin, n_non,
        )
    return best_path


def _resolve_resume_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Explicit US_STUDENT_RESUME_CKPT overrides auto-selection."""
    env = os.environ.get("US_STUDENT_RESUME_CKPT", "").strip()
    if env:
        path = Path(env)
        if not path.is_file():
            log.error("US_STUDENT_RESUME_CKPT not found: %s", path)
            return None
        log.info("Using US_STUDENT_RESUME_CKPT=%s", path)
        return path
    return _find_resume_checkpoint(ckpt_dir)


def _load_dataset_roots() -> Dict[str, str]:
    if not _DATA_CONFIG.exists():
        log.warning("data_run1.yaml not found at %s", _DATA_CONFIG)
        return {}
    with open(_DATA_CONFIG) as f:
        raw = yaml.safe_load(f)
    roots = dict(raw.get("datasets", {}))

    exclude: set = set()
    if _SMOKE_CFG.exists():
        with open(_SMOKE_CFG) as f:
            smoke_raw = yaml.safe_load(f) or {}
        exclude = set(smoke_raw.get("smoke", {}).get("exclude_datasets", []))
    if exclude:
        roots = {k: v for k, v in roots.items() if k not in exclude}
        if _is_main():
            log.info("Excluded %d datasets from smoke manifest scan", len(exclude))
    return roots


def _student_model_config(cfg: dict) -> StudentModelConfig:
    d = dict(cfg["model"]["student"])
    for env_key, field in [
        ("US_STUDENT_HIERA_VARIANT", "hiera_variant"),
        ("US_STUDENT_DINO_KEY", "dino_teacher_key"),
        ("US_STUDENT_VJEPA_KEY", "vjepa_teacher_key"),
    ]:
        if os.environ.get(env_key):
            d[field] = os.environ[env_key]
    return StudentModelConfig.from_dict(d)


def _build_manifest_entries(
    dataset_roots: Dict[str, str],
    n_per_dataset: int,
) -> Tuple[List[USManifestEntry], Dict[str, str]]:
    all_entries: List[USManifestEntry] = []
    status: Dict[str, str] = {}

    for ds_id, root_str in sorted(dataset_roots.items()):
        adapter_cls = ADAPTER_REGISTRY.get(ds_id)
        if adapter_cls is None:
            status[ds_id] = "skip:no_adapter"
            continue
        root = Path(root_str)
        if not root.exists():
            status[ds_id] = "skip:root_not_found"
            continue
        try:
            adapter = adapter_cls(root=str(root))
            entries: List[USManifestEntry] = []
            for e in adapter.iter_entries():
                entries.append(e)
                if len(entries) >= n_per_dataset:
                    break
            if entries:
                all_entries.extend(entries)
                status[ds_id] = f"ok:{len(entries)}"
            else:
                status[ds_id] = "skip:no_entries_yielded"
        except Exception:
            status[ds_id] = "fail:adapter_error"
            log.error("[%s] Adapter error:\n%s", ds_id, traceback.format_exc())

    return all_entries, status


def _status_from_manifest(
    entries: List[USManifestEntry],
    dataset_roots: Dict[str, str],
) -> Dict[str, str]:
    """Derive coverage table from a cached manifest without re-scanning adapters."""
    from collections import Counter

    counts = Counter(e.dataset_id for e in entries)
    status: Dict[str, str] = {}
    for ds_id in sorted(dataset_roots):
        n = counts.get(ds_id, 0)
        status[ds_id] = f"ok:{n}" if n else "skip:no_entries_yielded"
    return status


def _build_combined_manifest(
    dataset_roots: Dict[str, str],
    n_per_dataset: int,
    force: bool = False,
) -> Tuple[Path, Dict[str, str]]:
    _SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    status: Dict[str, str] = {}
    allowed_ids = set(dataset_roots.keys())

    if _is_main() or not _is_ddp():
        if _COMBINED_MANIFEST.exists() and not force:
            cached = load_manifest(_COMBINED_MANIFEST)
            stale_ids = {e.dataset_id for e in cached} - allowed_ids
            if not stale_ids:
                status = _status_from_manifest(cached, dataset_roots)
                log.info("Reusing manifest %s", _COMBINED_MANIFEST)
                _barrier()
                return _COMBINED_MANIFEST, status
            log.info(
                "Manifest stale (excluded datasets present: %s) — rebuilding",
                sorted(stale_ids),
            )
            force = True

        entries, status = _build_manifest_entries(dataset_roots, n_per_dataset)

        if not entries:
            raise RuntimeError(
                "No manifest entries found. Check dataset roots in data_run1.yaml."
            )

        with ManifestWriter(_COMBINED_MANIFEST) as w:
            for e in entries:
                w.write(e)
        log.info(
            "Manifest written: %d entries from %d datasets → %s",
            len(entries),
            sum(1 for v in status.values() if v.startswith("ok")),
            _COMBINED_MANIFEST,
        )

    _barrier()

    if not _COMBINED_MANIFEST.exists():
        raise RuntimeError(f"Manifest not found after rank-0 build: {_COMBINED_MANIFEST}")

    return _COMBINED_MANIFEST, status


def _validate_dataset_coverage(status: Dict[str, str], require_all: bool) -> None:
    if not _is_main():
        return
    ok = [k for k, v in status.items() if v.startswith("ok")]
    missing_root = [k for k, v in status.items() if v == "skip:no_root_in_config"]
    not_found = [k for k, v in status.items() if v == "skip:root_not_found"]
    empty = [k for k, v in status.items() if v == "skip:no_entries_yielded"]
    failed = [k for k, v in status.items() if v.startswith("fail")]

    # Datasets we expect to have on disk (root configured in data_run1.yaml)
    expected = {
        k for k, v in status.items()
        if v not in ("skip:no_root_in_config",)
    }
    missing_expected = [k for k in expected if not status[k].startswith("ok")]

    if _is_main():
        log.info(
            "Dataset coverage: %d ok | %d no_root | %d not_found | %d empty | %d fail",
            len(ok), len(missing_root), len(not_found), len(empty), len(failed),
        )
        if failed:
            log.warning("Failed adapters: %s", failed)
        if require_all and missing_expected:
            log.warning(
                "Expected datasets without entries (%d): %s",
                len(missing_expected), missing_expected[:20],
            )

    if require_all and missing_expected:
        raise RuntimeError(
            f"Smoke requires ≥1 entry from every configured dataset. "
            f"Missing {len(missing_expected)}: {missing_expected[:10]}..."
        )
    if not ok:
        raise RuntimeError("No datasets produced manifest entries.")

    if _is_main() and missing_expected:
        _announce(
            f"[WARN] {len(missing_expected)} configured datasets yielded no entries "
            f"(continuing with {len(ok)} ok datasets)"
        )


# ── DataModule ────────────────────────────────────────────────────────────────

def _build_datamodules(cfg: dict, manifest_path: Path) -> tuple[StudentDataModule, HardnessFeedback]:
    img_cfg, vid_cfg = build_transform_configs(cfg["transforms"])
    cur = cfg.get("curriculum", {})
    ckpt_dir = _resolve_ckpt_dir(cfg)
    alp_cfg = cfg.get("alp") or {}
    alp_disk = alp_cfg.get("disk_cache_dir")
    if alp_disk is None:
        alp_disk = str(ckpt_dir / "alp_cache")
    alp_cache = configure_alp_cache(
        max_entries=int(alp_cfg.get("max_entries", 1_000_000)),
        disk_cache_dir=alp_disk,
        score_ema=float(alp_cfg.get("score_ema", 0.9)),
    )
    alp_feedback = HardnessFeedback(alp_cache)

    loaders = cfg["loaders"]
    student_sd = dict(cfg.get("student_data") or {})
    for k in ("image_batch_size", "video_batch_size", "paired_batch_size",
              "num_workers", "prefetch_factor", "persistent_workers"):
        if k in loaders:
            student_sd[k] = loaders[k]

    manifest_cfg = cfg.get("manifest", {})
    base_dm = USFoundationDataModule(
        manifest_path=str(manifest_path),
        image_batch_size=loaders["image_batch_size"],
        video_batch_size=loaders["video_batch_size"],
        num_workers=loaders["num_workers"],
        pin_memory=loaders.get("pin_memory", True),
        image_cfg=img_cfg,
        video_cfg=vid_cfg,
        total_training_steps=cur["total_training_steps"],
        image_samples_per_epoch=cur["image_samples_per_epoch"],
        video_samples_per_epoch=cur["video_samples_per_epoch"],
        curriculum_stage_fracs=cur.get("alp_stage_fracs"),
        alp_alpha_init=float(cur.get("alp_alpha_init", 0.1)),
        alp_alpha_final=float(cur.get("alp_alpha_final", 0.9)),
        alp_guidance_threshold_init=float(cur.get("alp_guidance_threshold_init", 0.1)),
        alp_guidance_threshold_final=float(cur.get("alp_guidance_threshold_final", 0.9)),
        alp_n_frames=cur.get("alp_n_frames"),
        alp_reader=alp_cache,
        hardness_temperature=float(alp_cfg.get("hardness_temperature", 1.0)),
        exclude_datasets=manifest_cfg.get("exclude_datasets"),
    )
    base_dm.setup()

    student_cfg = StudentDataConfig.from_dict(student_sd)
    collators = StudentMixedCollator(
        ImageSSLCollator(),
        VideoSSLCollator(),
        PairedSSLCollator(),
        patch_size=student_cfg.patch_size,
    )
    student_dm = StudentDataModule(base_dm, student_cfg, collators)
    if _is_main():
        snap = base_dm.curriculum_snapshot()
        log.info(
            "ALP curriculum init: stage=%s alpha=%.2f m_t=%.2f mask_ratio=%.2f pool=%d tiers=%s",
            snap.get("alp_stage"), snap.get("alp_alpha"),
            snap.get("mask_guidance_threshold"), snap.get("mask_ratio"),
            snap.get("tier_pool_size"),
            {k: snap.get(f"tier{k}_total") for k in (1, 2, 3)},
        )
    return student_dm, alp_feedback


def _batch_mask_fraction(batch: dict) -> float:
    pm = batch.get("patch_masks")
    if pm is not None and isinstance(pm, torch.Tensor):
        return float(pm.float().mean().item())
    tm = batch.get("tube_masks_s16")
    if tm is None:
        tm = batch.get("tube_masks")
    if tm is not None and isinstance(tm, torch.Tensor):
        return float(tm.float().mean().item())
    return float("nan")


def _batch_mean_tier(batch: dict) -> float:
    tiers = batch.get("tiers")
    if tiers is None or not isinstance(tiers, torch.Tensor):
        return float("nan")
    return float(tiers.float().mean().item())


def _curriculum_metrics(
    snap: dict,
    batch: dict,
    alp_feedback: HardnessFeedback,
) -> dict[str, float]:
    cache = alp_feedback.cache
    out = {
        "curriculum_alp_stage": float(snap.get("alp_stage", 0)),
        "curriculum_alpha": float(snap.get("alp_alpha", 1.0)),
        "curriculum_mask_guidance_threshold": float(snap.get("mask_guidance_threshold", 0.5)),
        "curriculum_mask_ratio": float(snap.get("mask_ratio", 0.0)),
        "curriculum_hardness_weight": float(snap.get("hardness_weight", 0.0)),
        "curriculum_tier_pool": float(snap.get("tier_pool_size", 0)),
        "curriculum_tier1_pool": float(snap.get("tier1_in_pool", 0)),
        "curriculum_tier2_pool": float(snap.get("tier2_in_pool", 0)),
        "curriculum_tier3_pool": float(snap.get("tier3_in_pool", 0)),
        "alp_cache_entries": float(len(cache)),
        "alp_cache_hit_rate": float(cache.hit_rate),
        "alp_cache_updates": float(cache.update_count),
        "batch_mask_frac": _batch_mask_fraction(batch),
        "batch_mean_tier": _batch_mean_tier(batch),
    }
    return out


def _format_curriculum_metrics(metrics: dict[str, float]) -> str:
    keys = (
        "curriculum_alp_stage", "curriculum_alpha", "curriculum_mask_guidance_threshold",
        "curriculum_mask_ratio",
        "curriculum_hardness_weight", "curriculum_tier_pool",
        "batch_mask_frac", "batch_mean_tier",
        "alp_cache_entries", "alp_cache_hit_rate", "alp_cache_updates",
    )
    parts = []
    for k in keys:
        if k in metrics and math.isfinite(metrics[k]):
            short = k.replace("curriculum_", "").replace("alp_cache_", "alp_")
            parts.append(f"{short}={metrics[k]:.4f}" if "stage" not in k else f"{short}={int(metrics[k])}")
    return "  ".join(parts)


def _to_dev(batch: dict, device: str) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _make_paired_batch(batch: dict) -> dict:
    out = dict(batch)
    out["sample_type"] = "paired"
    clips = out["full_clips"]
    T = clips.shape[1]
    out["frame"] = clips[:, T // 2]
    out["frame_pmask"] = out.get("padding_masks")
    return out


def _prepare_stage4_batch(batch: dict) -> dict:
    out = dict(batch)
    if out.get("sample_type") == "video" and "full_clips" in out:
        clips = out["full_clips"]
        T = clips.shape[1]
        out["global_crops"] = clips[:, T // 2].unsqueeze(1)
        out["global_pmasks"] = out.get("padding_masks")
        if out["global_pmasks"] is not None:
            out["global_pmasks"] = out["global_pmasks"].unsqueeze(1)
        out["sample_type"] = "image"

    if out.get("seg_masks") is None and "global_crops" in out:
        crops = out["global_crops"][:, 0]
        B, _, H, W = crops.shape
        ph, pw = max(1, H // 4), max(1, W // 4)
        out["seg_masks"] = torch.randint(0, 2, (B, 1, ph, pw)).float()
    return out


# ── Curriculum helpers ──────────────────────────────────────────────────────────

def _stage_for_step(step: int, total: int, fracs: List[float]) -> int:
    """Return curriculum stage 1–4 for global step (0-indexed)."""
    bounds = _stage_bounds_from_fracs(fracs, total)
    for stage_idx in range(1, 5):
        if step < bounds[stage_idx]:
            return stage_idx
    return 4


def _resume_stage_frac_mode() -> str:
    """config (default): yaml stage_fracs; checkpoint: keep saved schedule."""
    mode = os.environ.get("US_STUDENT_RESUME_STAGE_FRACS", "config").strip().lower()
    if mode in ("checkpoint", "legacy", "ckpt"):
        return "checkpoint"
    return "config"


def _format_stage_bounds(fracs: List[float], total: int) -> str:
    bounds = _stage_bounds_from_fracs(fracs, total)
    return " ".join(f"S{i}=[{bounds[i - 1]},{bounds[i]})" for i in range(1, 5))


def _resolve_resume_stage_fracs(
    ckpt_fracs: Optional[List[float]],
    config_fracs: List[float],
    total: int,
    completed_step: int,
) -> List[float]:
    mode = _resume_stage_frac_mode()
    if mode == "checkpoint" and ckpt_fracs is not None:
        resolved = list(ckpt_fracs)
        msg = (
            f"Resume stage_fracs=checkpoint: {resolved} | "
            f"{_format_stage_bounds(resolved, total)}"
        )
        log.info(msg)
        if _is_main():
            _announce(msg)
        return resolved

    resolved = list(config_fracs)
    if ckpt_fracs is not None and list(ckpt_fracs) != resolved:
        old_stage = _stage_for_step(completed_step, total, ckpt_fracs)
        new_stage = _stage_for_step(completed_step + 1, total, resolved)
        msg = (
            f"Resume schedule migration at step {completed_step}: "
            f"checkpoint {ckpt_fracs} → config {resolved} | "
            f"was stage {old_stage} → continuing stage {new_stage} | "
            f"old {_format_stage_bounds(ckpt_fracs, total)} | "
            f"new {_format_stage_bounds(resolved, total)}"
        )
        log.warning(msg)
        if _is_main():
            _announce(msg)
    return resolved


def _stage3_curriculum(
    step: int,
    stage3_start: int,
    stage3_end: int,
    boundaries: List[float],
) -> int:
    """Return sub-curriculum stage 1–3 within stage 3."""
    if stage3_end <= stage3_start:
        return 1
    frac = (step - stage3_start) / (stage3_end - stage3_start)
    if frac < boundaries[0]:
        return 1
    if frac < boundaries[1]:
        return 2
    return 3


def _lr_for_step(step: int, cfg: dict) -> float:
    tcfg = cfg["training"]
    warmup = tcfg.get("warmup_steps", 200)
    total = int(
        os.environ.get("US_STUDENT_STEPS")
        or os.environ.get("US_SMOKE_STEPS")
        or tcfg["total_steps"]
    )
    base_lr = float(
        os.environ.get("US_STUDENT_LR") or tcfg["lr"]
    )
    min_lr = tcfg.get("lr_min", 1e-6)
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def _update_resolution_curriculum(step: int, cfg: dict, student_dm: StudentDataModule) -> None:
    stages = cfg.get("resolution_curriculum") or []
    if not stages:
        return
    active = stages[0]
    for stage in stages:
        if step >= int(stage.get("start_step", 0)):
            active = stage
    img_px = active.get("max_global_crop_px")
    vid_px = active.get("max_crop_px")
    base = student_dm.base_dm
    if img_px is not None:
        base.image_cfg.max_global_crop_px = int(img_px)
        if base._image_dataset is not None:
            base._image_dataset.transform.cfg.max_global_crop_px = int(img_px)
    if vid_px is not None:
        base.video_cfg.max_crop_px = int(vid_px)
        if base._video_dataset is not None:
            base._video_dataset.transform.cfg.max_crop_px = int(vid_px)


_STEP_MONITOR_KEYS = ("proto_entropy", "proto_max_prob", "lam_ema_eff")


def _extract_loss_metrics(step_out: dict) -> dict[str, float]:
    """Scalar loss breakdown from a student_stage*_step return dict."""
    metrics: dict[str, float] = {}
    loss = step_out.get("loss")
    if loss is not None:
        metrics["loss_total"] = float(loss.item() if torch.is_tensor(loss) else loss)
    for key, val in step_out.items():
        if not key.startswith("loss_") and key not in _STEP_MONITOR_KEYS:
            continue
        if isinstance(val, (int, float)):
            metrics[key] = float(val)
        elif torch.is_tensor(val) and val.numel() == 1:
            metrics[key] = float(val.item())
    return metrics


def _nonfinite_loss_parts(step_out: dict) -> list[str]:
    """Names of loss_* scalars that are NaN/Inf (for diagnostic logging)."""
    bad: list[str] = []
    for key, val in step_out.items():
        if key == "loss":
            if torch.is_tensor(val) and not torch.isfinite(val):
                bad.append("loss")
            continue
        if not key.startswith("loss_"):
            continue
        if torch.is_tensor(val):
            if val.numel() == 1 and not torch.isfinite(val):
                bad.append(key)
        elif isinstance(val, float) and not math.isfinite(val):
            bad.append(key)
    return bad


def _format_loss_metrics(metrics: dict[str, float]) -> str:
    """Format loss_total plus individual loss_* components for logging."""
    if not metrics:
        return "loss=n/a"
    parts: List[str] = []
    if "loss_total" in metrics:
        parts.append(f"loss={metrics['loss_total']:.4f}")
    for key in sorted(k for k in metrics if k != "loss_total"):
        short = key[len("loss_"):] if key.startswith("loss_") else key
        parts.append(f"{short}={metrics[key]:.4f}")
    return "  ".join(parts)


def _resolve_log_dir(cfg: dict, ckpt_dir: Path) -> Path:
    """Metrics + TensorBoard root (Run:ai / viz/training.py consume metrics.jsonl)."""
    for key in ("US_STUDENT_LOG_DIR", "US_SMOKE_LOG_DIR"):
        env = os.environ.get(key)
        if env:
            return Path(env)
    run_cfg = cfg.get("pretrain") or cfg.get("smoke") or {}
    raw = run_cfg.get("log_dir") or cfg.get("training", {}).get("log_dir")
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else _ROOT / p
    return ckpt_dir / "logs"


class StudentRunMetrics:
    """
    Persist per-step training metrics for Run:ai / offline monitoring.

    Writes metrics.jsonl (same schema as train.trainer.MetricLogger) and optional
    TensorBoard scalars under log_dir/tensorboard/.
    """

    def __init__(self, log_dir: Path, rank: int = 0, use_tensorboard: bool = True):
        self.log_dir = log_dir
        self.rank = rank
        self._jsonl = None
        self._tb = None
        if rank != 0:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(log_dir / "metrics.jsonl", "a", encoding="utf-8")
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
            except ImportError:
                log.warning("tensorboard not installed — metrics.jsonl only")

    def log(
        self,
        step: int,
        stage: int,
        metrics: dict[str, float],
        *,
        sample_type: str = "",
        lr: Optional[float] = None,
    ) -> None:
        if self.rank != 0:
            return
        row: dict = {
            "step": step,
            "stage": stage,
            "sample_type": sample_type,
            **metrics,
            "ts": time.time(),
        }
        if lr is not None:
            row["lr"] = lr
        if self._jsonl:
            self._jsonl.write(json.dumps(row) + "\n")
            self._jsonl.flush()
        if self._tb:
            if lr is not None:
                self._tb.add_scalar("train/lr", lr, step)
            self._tb.add_scalar("train/stage", float(stage), step)
            for key, val in metrics.items():
                if not isinstance(val, (int, float)):
                    continue
                if key == "loss_total":
                    tag = "loss/total"
                elif key.startswith("loss_"):
                    tag = f"loss/{key[len('loss_'):]}"
                elif key.startswith("curriculum_"):
                    tag = f"curriculum/{key[len('curriculum_'):]}"
                elif key.startswith("alp_cache_"):
                    tag = f"alp/{key[len('alp_cache_'):]}"
                elif key.startswith("batch_"):
                    tag = f"batch/{key[len('batch_'):]}"
                else:
                    tag = key
                self._tb.add_scalar(tag, val, step)

    def close(self) -> None:
        if self._jsonl:
            self._jsonl.close()
            self._jsonl = None
        if self._tb:
            self._tb.close()
            self._tb = None


# ── Model bundle ──────────────────────────────────────────────────────────────

class StudentSmokeTrainer:
    def __init__(self, cfg: dict, device: str, alp_feedback: Optional[HardnessFeedback] = None):
        self.cfg = cfg
        self.device = device
        self.model_cfg = _student_model_config(cfg)
        self.dtype = self.model_cfg.torch_dtype()
        self.lam = cfg.get("loss_weights", {})
        self.tcfg = cfg["training"]
        self.total_steps = int(
            os.environ.get("US_STUDENT_STEPS")
            or os.environ.get("US_SMOKE_STEPS")
            or self.tcfg["total_steps"]
        )
        self.stage_fracs = self.tcfg["stage_fracs"]
        self.ema_momentum = self.model_cfg.ema_momentum
        self.use_amp = self.tcfg.get("use_amp", True) and device.startswith("cuda")
        # GradScaler unscale is not implemented for bfloat16 on GH200/CUDA builds.
        self.use_amp_scaler = self.use_amp and self.dtype != torch.bfloat16

        self.ckpt_dir = _resolve_ckpt_dir(cfg)
        run_cfg = cfg.get("pretrain") or cfg.get("smoke", {})
        self.save_stage_end_ckpts = run_cfg.get("save_stage_end_ckpts", True)
        self.log_dir = _resolve_log_dir(cfg, self.ckpt_dir)
        use_tb = run_cfg.get("tensorboard", True)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        self.metrics = StudentRunMetrics(self.log_dir, rank=rank, use_tensorboard=use_tb)
        if _is_main():
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
            _announce(f"Checkpoints → {self.ckpt_dir}")
            log.info("Checkpoints → %s", self.ckpt_dir)
            _announce(f"Metrics → {self.log_dir}  (metrics.jsonl + tensorboard/)")
            log.info("Metrics → %s", self.log_dir)

        log.info("Loading Hiera student (%s, align_dim=%d) ...",
                 self.model_cfg.hiera_variant, self.model_cfg.align_dim)
        student = build_student_encoder(self.model_cfg, device=device).to(dtype=self.dtype)

        log.info("Loading DINO teacher (%s) ...", self.model_cfg.dino_teacher_key)
        dino = FrozenDINOTeacher(
            backbone_key=self.model_cfg.dino_teacher_key,
            align_dim=self.model_cfg.align_dim,
            dtype=self.dtype,
            hf_cache_dir=self.model_cfg.hiera_hf_cache_dir,
        ).to(device=device, dtype=self.dtype)

        self.vjepa = None  # lazy-loaded before stage 2 (saves ~600 MB in stage 1)

        self.ema_student = copy.deepcopy(_unwrap(student))
        self.ema_student.is_ema_target = True
        for p in self.ema_student.parameters():
            p.requires_grad_(False)
        self.ema_student.eval()
        self.ema_student.to(device=device, dtype=self.dtype)

        fusion = build_fusion_target_builder(self.model_cfg, device=device)
        proto = PrototypeHead(
            embed_dim=self.model_cfg.align_dim,
            n_prototypes=self.model_cfg.n_prototypes,
        ).to(device=device)

        # Optional per-rank queue for video prototype loss.
        # Video micro-batches (B_local=1 × n_ranks) are smaller than K prototypes,
        # so plain Sinkhorn is degenerate.  A queue of frozen V-JEPA teacher logits
        # from previous steps provides the missing coverage.  Disabled by default
        # (proto_queue_size=0); set > K (n_prototypes) in config to enable.
        _n_proto = self.model_cfg.n_prototypes
        _queue_size = int(self.tcfg.get("proto_queue_size", 0))
        self.proto_queue: Optional[ProtoQueue] = (
            ProtoQueue(K=_n_proto, queue_size=_queue_size, device=device)
            if _queue_size > _n_proto else None
        )
        if self.proto_queue is not None and _is_main():
            log.info("Video proto queue enabled: size=%d, K=%d", _queue_size, _n_proto)

        seg_head = build_hierarchical_seg_head(
            _unwrap(student).embed_dims,
            n_classes=1,
            fpn_channels=128,
        ).to(device=device)

        self.student = _maybe_ddp(student)
        self.dino = dino
        self.fusion = _maybe_ddp(fusion)
        self.proto = _maybe_ddp(proto)
        self.seg_head = _maybe_ddp(seg_head)

        params = (
            list(_unwrap(self.student).parameters())
            + list(_unwrap(self.fusion).parameters())
            + list(_unwrap(self.proto).parameters())
            + list(_unwrap(self.seg_head).parameters())
        )
        self.optimizer = torch.optim.AdamW(
            params,
            lr=float(os.environ.get("US_STUDENT_LR") or self.tcfg["lr"]),
            weight_decay=self.tcfg.get("weight_decay", 0.05),
            betas=(
                self.tcfg.get("beta1", 0.9),
                self.tcfg.get("beta2", 0.95),
            ),
        )
        self.scaler = GradScaler(enabled=self.use_amp_scaler)

        self.n_finite = 0
        self.n_nonfinite = 0
        self.last_finite_step = -1
        self._accum_failed = False
        self.stage_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        self.type_counts: Dict[str, int] = {}
        self.grad_accum_steps = max(1, int(self.tcfg.get("grad_accum_steps", 1)))
        self.alp_feedback = alp_feedback
        bench = cfg.get("bench") or {}
        self.no_ckpt = bool(bench.get("no_ckpt", False))
        self.per_step_timing = bool(bench.get("per_step_timing", False))
        self.last_timing = StepTiming()
        self._fwd_timer = {"student": 0.0, "teacher": 0.0}
        self._allreduce_timer = None
        if self.per_step_timing:
            self._allreduce_timer = try_register_allreduce_timer(self.student, every=50)
            self._install_fwd_hooks()

    def _install_fwd_hooks(self) -> None:
        """Accumulate wall time of student vs frozen-teacher forwards."""
        if not hasattr(self, "_hooked_ids"):
            self._hooked_ids: set = set()

        def _add(mod: Optional[nn.Module], bucket: str) -> None:
            if mod is None:
                return
            target = _unwrap(mod)
            mid = id(target)
            if mid in self._hooked_ids:
                return
            self._hooked_ids.add(mid)

            def _pre(_m, _i):  # noqa: ANN001
                target._pocus_t0 = time.perf_counter()  # type: ignore[attr-defined]

            def _post(_m, _i, _o):  # noqa: ANN001
                t0 = getattr(target, "_pocus_t0", None)
                if t0 is not None:
                    self._fwd_timer[bucket] += time.perf_counter() - t0

            target.register_forward_pre_hook(_pre)
            target.register_forward_hook(_post)

        _add(self.student, "student")
        _add(self.dino, "teacher")
        _add(getattr(self, "vjepa", None), "teacher")

    def _reset_fwd_timer(self) -> None:
        self._fwd_timer = {"student": 0.0, "teacher": 0.0}
        if self._allreduce_timer is not None:
            self._allreduce_timer.reset_step()

    def _teacher_on_gpu(self, mod: Optional[nn.Module]) -> bool:
        if mod is None:
            return False
        try:
            return next(mod.parameters()).device.type == "cuda"
        except StopIteration:
            return False

    def _park_teacher(self, attr: str) -> None:
        """Move a frozen teacher to CPU (float32) so video/paired steps fit in GPU memory."""
        mod = getattr(self, attr, None)
        if mod is None or not self._teacher_on_gpu(mod):
            return
        log.info("Parking %s on CPU to free GPU memory", attr)
        # CPU LayerNorm rejects bf16 weights with float32 activations (see FrozenDINOTeacher).
        setattr(self, attr, mod.cpu().float())
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        _barrier()

    def _ensure_teacher_gpu(self, attr: str) -> None:
        mod = getattr(self, attr, None)
        if mod is None or self._teacher_on_gpu(mod):
            return
        log.info("Moving %s back to GPU", attr)
        setattr(self, attr, mod.to(device=self.device, dtype=self.dtype))
        _barrier()

    def _ensure_vjepa(self) -> None:
        """Load V-JEPA teacher on first use (stage 2+)."""
        if self.vjepa is None:
            log.info("Loading V-JEPA teacher (%s) ...", self.model_cfg.vjepa_teacher_key)
            self.vjepa = FrozenVJEPATeacher(
                backbone_key=self.model_cfg.vjepa_teacher_key,
                align_dim=self.model_cfg.align_dim,
                dtype=self.dtype,
                hf_cache_dir=self.model_cfg.hiera_hf_cache_dir,
            ).to(device=self.device, dtype=self.dtype)
            if self.per_step_timing:
                self._install_fwd_hooks()
            _barrier()
        else:
            self._ensure_teacher_gpu("vjepa")

    def sync_teachers_for_stage(self, stage: int) -> None:
        """Keep teachers required for this curriculum stage on GPU."""
        if stage == 1:
            self._ensure_teacher_gpu("dino")
            self._park_teacher("vjepa")
        elif stage in (2, 3):
            self._ensure_teacher_gpu("dino")
            self._ensure_vjepa()
        else:
            self._park_teacher("dino")
            self._park_teacher("vjepa")
        # Fusion is only trained in stage 3; freeze in stage 4.
        fusion = _unwrap(self.fusion)
        train_fusion = stage == 3
        for p in fusion.parameters():
            p.requires_grad_(train_fusion)

    def _run_step(
        self,
        batch: dict,
        stage: int,
        step: int,
        stage3_start: int,
        stage3_end: int,
        stage4_start: int,
        stage4_end: int,
    ) -> dict:
        student = _unwrap(self.student)
        proto = _unwrap(self.proto)
        fusion = _unwrap(self.fusion)
        seg_head = _unwrap(self.seg_head)

        if batch.get("sample_type") == "paired" and batch.get("frame") is None:
            batch = _make_paired_batch(batch)

        if stage == 2:
            self._ensure_vjepa()
        elif stage == 3:
            self._prepare_stage3_teachers(batch.get("sample_type", "image"))

        ctx = autocast(dtype=self.dtype) if self.use_amp else torch.enable_grad()
        with ctx:
            if stage == 1:
                out = student_stage1_step(
                    batch, student, self.ema_student, self.dino, proto, self.lam,
                    global_step=step, alp_feedback=self.alp_feedback,
                )
            elif stage == 2:
                out = student_stage2_step(
                    batch, student, self.ema_student, self.dino, self.vjepa, proto, self.lam,
                    global_step=step, alp_feedback=self.alp_feedback,
                    proto_queue=self.proto_queue,
                )
            elif stage == 3:
                curr = _stage3_curriculum(
                    step, stage3_start, stage3_end,
                    self.tcfg.get("stage3_curriculum_boundaries", [0.33, 0.67]),
                )
                out = student_stage3_step(
                    batch, student, self.ema_student, self.dino, self.vjepa,
                    fusion, proto, self.lam,
                    curriculum_stage=curr, global_step=step,
                    alp_feedback=self.alp_feedback,
                    proto_queue=self.proto_queue,
                )
            else:
                out = student_stage4_divergence_step(
                    batch, student, self.ema_student, proto, self.lam,
                    global_step=step,
                    stage4_start=stage4_start,
                    stage4_end=stage4_end,
                    alp_feedback=self.alp_feedback,
                    proto_queue=self.proto_queue,
                )
        return out

    def _prepare_stage3_teachers(self, sample_type: str) -> None:
        """
        Stage-3 teacher residency.

        On GH200 120 GB both DINO and V-JEPA remain GPU-resident for the entire
        stage (loaded once by sync_teachers_for_stage). No per-batch swapping.
        """

    def _set_lr(self, step: int) -> None:
        lr = _lr_for_step(step, self.cfg)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

    def _loss_is_finite(self, loss: torch.Tensor) -> bool:
        if _is_ddp():
            finite_flag = torch.tensor(
                [1 if torch.isfinite(loss) else 0],
                device=loss.device,
                dtype=torch.int32,
            )
            dist.all_reduce(finite_flag, op=dist.ReduceOp.MIN)
            return finite_flag.item() != 0
        return bool(torch.isfinite(loss))

    def train_step(
        self,
        batch: dict,
        stage: int,
        step: int,
        stage3_start: int,
        stage3_end: int,
        stage4_start: int,
        stage4_end: int,
        micro_idx: int = 0,
        grad_accum_steps: int = 1,
        cuda_timer: Optional[CudaStepTimer] = None,
    ) -> Optional[dict[str, float]]:
        if micro_idx == 0:
            self.optimizer.zero_grad(set_to_none=True)
            self._accum_failed = False
            self._reset_fwd_timer()
        elif self._accum_failed:
            return None

        if cuda_timer:
            cuda_timer.start("fwd")
        out = self._run_step(
            batch, stage, step, stage3_start, stage3_end, stage4_start, stage4_end,
        )
        t_fwd = cuda_timer.stop("fwd") if cuda_timer else 0.0
        loss = out["loss"] / grad_accum_steps

        if not self._loss_is_finite(loss):
            self._accum_failed = True
            self.n_nonfinite += 1
            self.last_timing = StepTiming(
                step=step, stage=stage, sample_type=str(batch.get("sample_type", "")),
                t_fwd_student=self._fwd_timer["student"],
                t_fwd_teachers=self._fwd_timer["teacher"],
                nonfinite=True, loss=float("nan"),
            )
            if _is_main():
                bad = _nonfinite_loss_parts(out)
                detail = f" ({', '.join(bad)})" if bad else ""
                log.warning(
                    "step=%d stage=%d micro=%d non-finite loss%s — skipping step",
                    step, stage, micro_idx, detail,
                )
            return None

        if cuda_timer:
            cuda_timer.start("bwd")
        if self.use_amp_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        t_bwd = cuda_timer.stop("bwd") if cuda_timer else 0.0

        if micro_idx + 1 < grad_accum_steps:
            return None

        if cuda_timer:
            cuda_timer.start("opt")
        params = [p for g in self.optimizer.param_groups for p in g["params"]]
        if self.use_amp_scaler:
            self.scaler.unscale_(self.optimizer)

        grad_norm = nn.utils.clip_grad_norm_(params, self.tcfg.get("grad_clip", 1.0))

        if not torch.isfinite(grad_norm):
            self._accum_failed = True
            self.n_nonfinite += 1
            self.optimizer.zero_grad(set_to_none=True)
            if _is_main():
                log.warning(
                    "step=%d stage=%d non-finite grad_norm=%s — skipping optimizer step",
                    step, stage, grad_norm,
                )
            return None

        if self.use_amp_scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        ema_update(_unwrap(self.student), self.ema_student, self.ema_momentum)
        t_opt = cuda_timer.stop("opt") if cuda_timer else 0.0
        self.n_finite += 1
        self.last_finite_step = step
        n_img, n_clip, n_fr = count_batch_items(batch)
        self.last_timing = StepTiming(
            step=step,
            stage=stage,
            sample_type=str(batch.get("sample_type", "")),
            t_fwd_student=self._fwd_timer["student"] or t_fwd,
            t_fwd_teachers=self._fwd_timer["teacher"],
            t_bwd=t_bwd,
            t_opt=t_opt,
            t_allreduce=(
                self._allreduce_timer.t_allreduce if self._allreduce_timer else 0.0
            ),
            n_images=n_img * grad_accum_steps,
            n_clips=n_clip * grad_accum_steps,
            n_frames=n_fr * grad_accum_steps,
            mem_peak_GB=peak_mem_gb(self.device),
            loss=float(out["loss"].item()) if torch.is_tensor(out["loss"]) else float(out["loss"]),
            nonfinite=False,
        )
        return _extract_loss_metrics(out)

    def _build_checkpoint(self, step: int, stage: int) -> dict:
        return {
            "step": step,
            "stage": stage,
            "total_steps": self.total_steps,
            "stage_fracs": self.stage_fracs,
            "loss_weights": dict(self.lam),
            "student": _unwrap(self.student).state_dict(),
            "ema_student": self.ema_student.state_dict(),
            "fusion": _unwrap(self.fusion).state_dict(),
            "proto": _unwrap(self.proto).state_dict(),
            "seg_head": _unwrap(self.seg_head).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.use_amp else None,
            "model_cfg": self.model_cfg.__dict__,
            "stage_counts": dict(self.stage_counts),
            "type_counts": dict(self.type_counts),
            "n_finite": self.n_finite,
            "n_nonfinite": self.n_nonfinite,
            "last_finite_step": self.last_finite_step,
        }

    def save_checkpoint(self, step: int, stage: int, path: Path, *, force: bool = False) -> None:
        if not _is_main():
            return
        if self.no_ckpt and not force:
            return
        torch.save(self._build_checkpoint(step, stage), path)
        msg = f"Checkpoint saved → {path}  (step={step} stage={stage})"
        log.info(msg)
        _announce(msg)

    def save_stage_end(self, step: int, stage: int) -> None:
        """Save stage{N}_end.pt at the boundary between curriculum stages."""
        if not self.save_stage_end_ckpts or stage not in (1, 2, 3, 4):
            return
        self.save_checkpoint(step, stage, self.ckpt_dir / f"stage{stage}_end.pt")

    def load_checkpoint(self, path: Path) -> int:
        """
        Restore model, optimizer, and counters from a checkpoint.

        Returns the completed step stored in the checkpoint (0-indexed).
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        step = int(ckpt["step"])
        ckpt_fracs = ckpt.get("stage_fracs")
        self.stage_fracs = _resolve_resume_stage_fracs(
            ckpt_fracs, self.stage_fracs, self.total_steps, step,
        )
        stage = int(ckpt.get("stage", _stage_for_step(step, self.total_steps, self.stage_fracs)))

        ckpt_total = ckpt.get("total_steps")
        if ckpt_total is not None and int(ckpt_total) != self.total_steps:
            log.warning(
                "Checkpoint total_steps=%s differs from run config %s",
                ckpt_total, self.total_steps,
            )

        if ckpt.get("loss_weights"):
            self.lam = {**ckpt["loss_weights"], **self.lam}

        _unwrap(self.student).load_state_dict(
            _adapt_student_state_dict(ckpt["student"], _unwrap(self.student).state_dict()),
            strict=True,
        )
        self.ema_student.load_state_dict(
            _adapt_student_state_dict(ckpt["ema_student"], self.ema_student.state_dict()),
            strict=True,
        )
        _unwrap(self.fusion).load_state_dict(ckpt["fusion"], strict=True)
        _unwrap(self.proto).load_state_dict(ckpt["proto"], strict=True)
        _unwrap(self.seg_head).load_state_dict(
            _adapt_seg_head_state_dict(ckpt["seg_head"], _unwrap(self.seg_head).state_dict()),
            strict=True,
        )

        if "optimizer" in ckpt:
            try:
                self.optimizer.load_state_dict(
                    _adapt_optimizer_state_dict(
                        ckpt["optimizer"],
                        self.optimizer,
                        (self.student, self.fusion, self.proto),
                    )
                )
            except Exception as exc:
                log.warning(
                    "Skipping optimizer state restore (%s) — using fresh optimizer",
                    exc,
                )
        if self.use_amp_scaler and ckpt.get("scaler") is not None:
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception as exc:
                log.warning(
                    "Skipping scaler state restore (%s) — using fresh scaler",
                    exc,
                )

        raw_counts = ckpt.get("stage_counts") or {}
        self.stage_counts = {int(k): int(v) for k, v in raw_counts.items()}
        for s in (1, 2, 3, 4):
            self.stage_counts.setdefault(s, 0)
        self.type_counts = dict(ckpt.get("type_counts") or {})
        self.n_finite = int(ckpt.get("n_finite", sum(self.stage_counts.values())))
        self.n_nonfinite = int(ckpt.get("n_nonfinite", 0))
        self.last_finite_step = int(
            ckpt.get("last_finite_step", _checkpoint_last_finite_step(ckpt))
        )

        resume_stage = _stage_for_step(step + 1, self.total_steps, self.stage_fracs)
        self.sync_teachers_for_stage(resume_stage)

        msg = (
            f"Resumed from {path.name}  (step={step} ckpt_stage={stage} "
            f"resume_stage={resume_stage} fracs={self.stage_fracs} "
            f"last_finite={self.last_finite_step} "
            f"n_finite={self.n_finite} n_nonfinite={self.n_nonfinite})"
        )
        log.info(msg)
        if _is_main():
            _announce(msg)
        return step


# ── Training loop ─────────────────────────────────────────────────────────────

def run_training(
    cfg: dict,
    student_dm: StudentDataModule,
    trainer: StudentSmokeTrainer,
    alp_feedback: HardnessFeedback,
    start_step: int = 0,
) -> Dict:
    device = trainer.device
    total = trainer.total_steps
    fracs = trainer.stage_fracs
    log_every = trainer.tcfg.get("log_every", 50)
    ckpt_every = trainer.tcfg.get("ckpt_every", 1000)

    stage_bounds = [0]
    for f in fracs:
        stage_bounds.append(stage_bounds[-1] + int(total * f))
    stage_bounds[-1] = total
    stage3_start, stage3_end = stage_bounds[2], stage_bounds[3]
    stage4_start, stage4_end = stage_bounds[3], stage_bounds[4]

    if start_step >= total:
        if _is_main():
            _announce(f"Training already complete ({start_step}/{total} steps) — nothing to do")
        return {
            "total_steps": total,
            "finite_steps": trainer.n_finite,
            "nonfinite_steps": trainer.n_nonfinite,
            "stage_counts": trainer.stage_counts,
            "type_counts": trainer.type_counts,
            "elapsed_s": 0.0,
            "steps_per_s": 0.0,
            "ckpt_dir": str(trainer.ckpt_dir),
            "resumed": True,
            "start_step": start_step,
        }

    n_gpus = _world_size() if _is_ddp() else gpu_count_from_env()
    bench = cfg.get("bench") or {}
    per_step = bool(bench.get("per_step_timing", False) or trainer.per_step_timing)
    window_ctl = BenchWindowController(enabled=bool(bench.get("bench_window", False)))
    steps_jsonl_path = bench.get("steps_jsonl") or os.environ.get("US_STUDENT_STEPS_JSONL")
    steps_fp = None
    if _is_main() and (per_step or window_ctl.enabled) and steps_jsonl_path:
        Path(steps_jsonl_path).parent.mkdir(parents=True, exist_ok=True)
        steps_fp = open(steps_jsonl_path, "a", encoding="utf-8")
    cuda_timer = CudaStepTimer(enabled=per_step, device=device)
    active_stage: Optional[int] = None

    pretrain = _is_pretrain_mode(cfg)
    run_kind = "pretrain" if pretrain else "smoke"
    if _is_main():
        resume_note = f" from step {start_step}" if start_step > 0 else ""
        _announce(
            f"Starting {total}-step student {run_kind}{resume_note} | stage bounds={stage_bounds} | "
            f"DDP={_is_ddp()} | n_gpus={n_gpus} | GBS_img={trainer.tcfg.get('global_batch_image')} "
            f"GBS_vid={trainer.tcfg.get('global_batch_video')} | ckpt_dir={trainer.ckpt_dir}"
        )
        log.info(
            "Starting %d-step %s run%s | stages: %s | DDP=%s | n_gpus=%d | ckpt_dir=%s",
            total, run_kind, resume_note, stage_bounds, _is_ddp(), n_gpus, trainer.ckpt_dir,
        )

    t0 = time.time()
    metrics_every = int(trainer.tcfg.get("metrics_every", 1))
    batch: dict = {}

    try:
        for opt_step in range(start_step, total):
            stage = _stage_for_step(opt_step, total, fracs)
            if stage != active_stage:
                trainer.sync_teachers_for_stage(stage)
                active_stage = stage
            student_dm.set_stage(stage)
            _update_resolution_curriculum(opt_step, cfg, student_dm)
            cur_snap = student_dm.update_curriculum(opt_step)

            if _is_ddp():
                student_dm.set_epoch(opt_step)

            # Pin sample type for the whole optimizer step, then type-specific accum.
            stype = student_dm.current_mix.sample_type() if (_is_main() or not _is_ddp()) else "image"
            stype = _ddp_broadcast_sample_type(stype)
            try:
                grad_accum = resolve_grad_accum(cfg, n_gpus, stype)
            except ValueError as exc:
                log.warning("accum fallback (%s) — using grad_accum_steps", exc)
                grad_accum = max(1, int(trainer.tcfg.get("grad_accum_steps", 1)))

            step_metrics: Optional[dict[str, float]] = None
            trainer._accum_failed = False
            t_data = 0.0
            t_step0 = time.perf_counter()
            for micro_idx in range(grad_accum):
                if trainer._accum_failed:
                    break
                t_d0 = time.perf_counter()
                batch = student_dm.next_batch(forced_type=stype)
                t_data += time.perf_counter() - t_d0
                stype = str(batch.get("sample_type", stype))
                batch = _to_dev(batch, device)
                trainer._set_lr(opt_step)
                metrics = trainer.train_step(
                    batch, stage, opt_step, stage3_start, stage3_end,
                    stage4_start, stage4_end,
                    micro_idx=micro_idx, grad_accum_steps=grad_accum,
                    cuda_timer=cuda_timer if per_step else None,
                )
                if metrics is not None:
                    step_metrics = metrics
            t_step = time.perf_counter() - t_step0
            trainer.last_timing.t_step = t_step
            trainer.last_timing.t_data_wait = t_data
            trainer.last_timing.step = opt_step
            trainer.last_timing.stage = stage
            trainer.last_timing.sample_type = str(batch.get("sample_type", stype))

            if trainer._accum_failed:
                trainer.optimizer.zero_grad(set_to_none=True)
                if _is_main() and steps_fp:
                    row = trainer.last_timing.to_dict()
                    row["nonfinite"] = True
                    steps_fp.write(json.dumps(row) + "\n")
                    steps_fp.flush()
                continue

            if step_metrics is not None:
                trainer.stage_counts[stage] += 1
                st = batch.get("sample_type", "?")
                trainer.type_counts[st] = trainer.type_counts.get(st, 0) + 1
                cur_metrics = _curriculum_metrics(cur_snap, batch, alp_feedback)
                step_metrics.update(cur_metrics)
                if _is_main() and (opt_step % metrics_every == 0 or opt_step == total - 1):
                    trainer.metrics.log(
                        opt_step,
                        stage,
                        step_metrics,
                        sample_type=st,
                        lr=trainer.optimizer.param_groups[0]["lr"],
                    )
                if _is_main() and opt_step == 0 and device.startswith("cuda"):
                    mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                    log.info("max_memory_allocated after step 0: %.2f GB", mem_gb)

            if _is_main() and steps_fp:
                row = trainer.last_timing.to_dict()
                if step_metrics:
                    row["loss"] = step_metrics.get("loss_total", row.get("loss"))
                steps_fp.write(json.dumps(row) + "\n")
                steps_fp.flush()

            stop = window_ctl.observe(t_step)
            if _is_ddp():
                flag = torch.tensor(
                    [1 if stop else 0],
                    device=device if str(device).startswith("cuda") else "cpu",
                    dtype=torch.int32,
                )
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                stop = bool(flag.item())
            if stop:
                if _is_main() and window_ctl.result is not None:
                    wr = window_ctl.result
                    _announce(
                        f"Bench window stop: stable={wr.stable} n={wr.n_steps} "
                        f"t̄={wr.t_mean:.4f}s half_diff={wr.half_rel_diff:.3%} ({wr.reason})"
                    )
                    log.info("Bench window: %s", wr)
                break

            if _is_main() and (opt_step % log_every == 0 or opt_step == total - 1):
                elapsed = time.time() - t0
                sps = (opt_step + 1 - start_step) / max(elapsed, 1e-6)
                loss_str = _format_loss_metrics(step_metrics) if step_metrics else "loss=n/a"
                cur_str = _format_curriculum_metrics(step_metrics) if step_metrics else ""
                line = (
                    f"step={opt_step:4d}/{total}  stage={stage}  "
                    f"type={batch.get('sample_type', '?')}  "
                    f"finite={trainer.n_finite}  nonfinite={trainer.n_nonfinite}  {sps:.2f} step/s"
                    f"  t_step={t_step:.3f}s t_data={t_data:.3f}s accum={grad_accum}"
                )
                _announce(line)
                log.info(line)
                _announce(f"  {loss_str}")
                log.info("  %s", loss_str)
                if cur_str:
                    _announce(f"  curriculum: {cur_str}")
                    log.info("  curriculum: %s", cur_str)

            # Stage-boundary checkpoints (stage1_end.pt, stage2_end.pt, stage3_end.pt)
            if opt_step + 1 < total:
                next_stage = _stage_for_step(opt_step + 1, total, fracs)
                if next_stage != stage:
                    _barrier()
                    trainer.save_stage_end(opt_step, stage)

            if _is_main() and ckpt_every > 0 and opt_step > 0 and opt_step % ckpt_every == 0:
                trainer.save_checkpoint(opt_step, stage, trainer.ckpt_dir / f"step_{opt_step:05d}.pt")
    finally:
        if steps_fp:
            steps_fp.close()
        if _is_main():
            trainer.metrics.close()

    if trainer.last_finite_step >= 0:
        latest_step = trainer.last_finite_step
    elif trainer.n_finite > 0:
        latest_step = start_step + trainer.n_finite - 1
    else:
        latest_step = total - 1
    latest_stage = _stage_for_step(latest_step, total, fracs)
    if _is_main():
        trainer.save_checkpoint(latest_step, latest_stage, trainer.ckpt_dir / "latest.pt")

    _barrier()
    elapsed = time.time() - t0
    return {
        "total_steps": total,
        "finite_steps": trainer.n_finite,
        "nonfinite_steps": trainer.n_nonfinite,
        "stage_counts": trainer.stage_counts,
        "type_counts": trainer.type_counts,
        "elapsed_s": elapsed,
        "steps_per_s": (total - start_step) / max(elapsed, 1e-6),
        "ckpt_dir": str(trainer.ckpt_dir),
        "log_dir": str(trainer.log_dir),
        "start_step": start_step,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Student pipeline smoke / pretrain / WP5 bench")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in ckpt_dir (highest step wins)",
    )
    parser.add_argument("--bench-stage", type=int, choices=(1, 2, 3, 4), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-ckpt", action="store_true")
    parser.add_argument("--ckpt-probe", action="store_true")
    parser.add_argument("--per-step-timing", action="store_true")
    parser.add_argument("--loader-only", action="store_true")
    parser.add_argument("--bench-window", action="store_true")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--video-manifest", type=str, default=None)
    parser.add_argument("--image-mbs", type=int, default=None)
    parser.add_argument("--video-mbs", type=int, default=None)
    parser.add_argument("--gbs-img", type=int, default=None)
    parser.add_argument("--gbs-vid", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--ckpt-probe-scratch", type=str, default=None)
    parser.add_argument("--ckpt-probe-store", type=str, default=None)
    parser.add_argument("--forced-type", choices=("image", "video"), default=None)
    return parser.parse_args()


def _should_resume(args: argparse.Namespace) -> bool:
    return args.resume or os.environ.get("US_STUDENT_RESUME", "0") == "1"


def _run_loader_only(
    cfg: dict,
    student_dm: StudentDataModule,
    device: str,
    args: argparse.Namespace,
) -> dict:
    """E5: iterate the data-loader with no model (images/s, clips/s)."""
    n = int(args.max_steps or cfg.get("training", {}).get("total_steps", 50))
    forced = args.forced_type
    stage = int(args.bench_stage or 1)
    student_dm.set_stage(stage)
    if _is_ddp():
        student_dm.set_epoch(0)
    times: list[float] = []
    n_img = n_clip = n_fr = 0
    t0 = time.perf_counter()
    for i in range(n):
        t1 = time.perf_counter()
        batch = student_dm.next_batch(forced_type=forced)
        dt = time.perf_counter() - t1
        times.append(dt)
        a, b, c = count_batch_items(batch)
        n_img += a
        n_clip += b
        n_fr += c
        if _is_main() and i % 10 == 0:
            log.info("loader-only step=%d t=%.3fs type=%s", i, dt, batch.get("sample_type"))
    elapsed = time.perf_counter() - t0
    mean_t = sum(times) / len(times) if times else float("nan")
    out = {
        "mode": "loader_only",
        "steps": n,
        "elapsed_s": elapsed,
        "t_mean": mean_t,
        "images": n_img,
        "clips": n_clip,
        "frames": n_fr,
        "images_per_s": n_img / elapsed if elapsed else float("nan"),
        "clips_per_s": n_clip / elapsed if elapsed else float("nan"),
        "forced_type": forced,
        "stage": stage,
    }
    jsonl = os.environ.get("US_STUDENT_STEPS_JSONL")
    if _is_main() and jsonl:
        Path(jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl, "a", encoding="utf-8") as f:
            for i, t in enumerate(times):
                f.write(json.dumps({
                    "step": i, "stage": stage, "type": forced or "mixed",
                    "t_step": t, "t_data_wait": t, "loader_only": True,
                }) + "\n")
    return out


def _run_ckpt_probe(trainer: StudentSmokeTrainer, args: argparse.Namespace) -> dict:
    """E4: time one resumable checkpoint write and read on scratch and store."""
    user = os.environ.get("USER", "unknown")
    scratch = Path(
        args.ckpt_probe_scratch
        or f"/capstor/scratch/cscs/{user}/pocus-bench/ckpt_probe"
    )
    store = Path(
        args.ckpt_probe_store
        or "/capstor/store/cscs/swissai/a127/pocus-bench/ckpt_probe"
    )
    results = []
    for dest, label in ((scratch, "scratch"), (store, "store")):
        try:
            dest.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            results.append({"dest": label, "path": str(dest), "error": str(exc)})
            continue
        path = dest / "ckpt_probe.pt"
        t0 = time.perf_counter()
        trainer.save_checkpoint(0, 1, path, force=True)
        if _is_ddp():
            _barrier()
        t_write = time.perf_counter() - t0
        size = path.stat().st_size if path.exists() else 0
        t1 = time.perf_counter()
        if path.exists():
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            _ = ckpt.get("step")
        t_read = time.perf_counter() - t1
        results.append({
            "dest": label,
            "path": str(path),
            "bytes": size,
            "t_write_s": t_write,
            "t_read_s": t_read,
            "write_GBps": (size / t_write / 1e9) if t_write else float("nan"),
        })
        jsonl = os.environ.get("US_STUDENT_STEPS_JSONL")
        if _is_main() and jsonl:
            Path(jsonl).parent.mkdir(parents=True, exist_ok=True)
            with open(jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": 0, "type": "ckpt_probe", **results[-1]}) + "\n")
    return {"mode": "ckpt_probe", "results": results}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    try:
        mp.set_sharing_strategy("file_system")
    except RuntimeError:
        pass
    _init_dist()
    device = _auto_device()
    cfg = _load_config()
    apply_bench_overrides(cfg, args)
    cfg.setdefault("bench", {})
    cfg["bench"].update({
        "per_step_timing": bool(args.per_step_timing or args.bench_window or args.loader_only),
        "bench_window": bool(args.bench_window),
        "no_ckpt": bool(args.no_ckpt) and not bool(args.ckpt_probe),
        "steps_jsonl": os.environ.get("US_STUDENT_STEPS_JSONL"),
        "forced_type": args.forced_type,
    })
    if args.manifest:
        cfg.setdefault("manifest", {})["path"] = args.manifest
    seed = int(args.seed if args.seed is not None else cfg.get("training", {}).get("seed", 1234))
    set_global_seed(seed)
    _resolve_hf_cache(cfg)
    pretrain = _is_pretrain_mode(cfg)
    if _is_main():
        log.info("Loaded config: %s", _LOADED_CONFIG_PATH)
        log.info("Mode: %s", "pretrain" if pretrain else "smoke")
    smoke_cfg = cfg.get("smoke", {})
    stats: Optional[Dict] = None
    dataset_status: Dict[str, str] = {}
    start_step = 0

    try:
        if pretrain:
            manifest_path = Path(cfg["manifest"]["path"])
            if not manifest_path.is_absolute():
                manifest_path = _ROOT / manifest_path
            if not manifest_path.exists():
                raise FileNotFoundError(f"Pretrain manifest not found: {manifest_path}")
            if _is_main():
                _announce(f"Pretrain mode — manifest: {manifest_path}")
        else:
            dataset_roots = _load_dataset_roots()
            n_per_ds = smoke_cfg.get("entries_per_dataset", 8)
            force_rebuild = os.environ.get("US_SMOKE_FORCE_REBUILD", "0") == "1"
            manifest_path, dataset_status = _build_combined_manifest(
                dataset_roots, n_per_ds, force=force_rebuild,
            )
            _validate_dataset_coverage(
                dataset_status,
                require_all=smoke_cfg.get("require_all_datasets", False),
            )
            if _is_main():
                _print_dataset_table(dataset_status)

        student_dm, alp_feedback = _build_datamodules(cfg, manifest_path)

        if args.loader_only:
            stats = _run_loader_only(cfg, student_dm, device, args)
            if _is_main():
                print(json.dumps(stats, indent=2, default=str))
            return

        trainer = StudentSmokeTrainer(cfg, device, alp_feedback=alp_feedback)

        if args.ckpt_probe:
            stats = _run_ckpt_probe(trainer, args)
            if _is_main():
                print(json.dumps(stats, indent=2, default=str))
            return

        if _should_resume(args):
            ckpt_path = _resolve_resume_checkpoint(trainer.ckpt_dir)
            if ckpt_path is None:
                if _is_main():
                    _announce(f"No checkpoint found in {trainer.ckpt_dir} — starting from step 0")
                log.warning("Resume requested but no checkpoint in %s", trainer.ckpt_dir)
            else:
                completed_step = trainer.load_checkpoint(ckpt_path)
                start_step = completed_step + 1
                resume_stage = _stage_for_step(start_step, trainer.total_steps, trainer.stage_fracs)
                student_dm.set_stage(resume_stage)
                if _is_ddp():
                    student_dm.set_epoch(start_step)
                _barrier()

        if _is_main():
            msg = (
                f"DataModule ready — image={len(student_dm.base_dm._image_entries)} "
                f"video={len(student_dm.base_dm._video_entries)} entries | "
                f"steps={trainer.total_steps} lr={trainer.tcfg.get('lr')}"
            )
            if start_step > 0:
                msg += f" | resume_step={start_step}"
            _announce(msg)
            log.info(msg)

        runtime_cfg = {**cfg.get("smoke", {}), **cfg.get("pretrain", {})}
        if loader_warmup := os.environ.get("US_STUDENT_LOADER_WARMUP"):
            warmup_batches = int(loader_warmup)
        else:
            warmup_batches = int(runtime_cfg.get("loader_warmup_batches", 2))
        init_stage = _stage_for_step(start_step, trainer.total_steps, trainer.stage_fracs)
        if start_step > 0 and init_stage >= 3 and "US_STUDENT_LOADER_WARMUP" not in os.environ:
            warmup_batches = 0
            if _is_main():
                log.info(
                    "Skipping dataloader warmup (resume into stage %d at step %d)",
                    init_stage, start_step,
                )
        trainer.sync_teachers_for_stage(init_stage)
        student_dm.set_stage(init_stage)
        init_snap = student_dm.update_curriculum(start_step)
        if _is_main():
            log.info(
                "ALP curriculum at step %d: %s",
                start_step, _format_curriculum_metrics(_curriculum_metrics(init_snap, {}, alp_feedback)),
            )
        if warmup_batches > 0:
            if _is_main():
                if start_step > 0:
                    log.info(
                        "Warming up dataloaders (%d batches/stream, resume step %d)...",
                        warmup_batches, start_step,
                    )
                else:
                    log.info("Warming up dataloaders (%d batches/stream)...", warmup_batches)
            if start_step == 0:
                student_dm.set_stage(1)
            student_dm.warmup_loaders(warmup_batches, epoch=start_step)

        stats = run_training(cfg, student_dm, trainer, alp_feedback, start_step=start_step)

        if _is_main():
            if dataset_status:
                _print_final_summary(stats, dataset_status)
            else:
                _print_pretrain_summary(stats)

        if not pretrain:
            if stats["nonfinite_steps"] > stats["total_steps"] * 0.05:
                raise RuntimeError(f"Too many non-finite steps: {stats['nonfinite_steps']}")
            if stats["finite_steps"] < stats["total_steps"] * 0.9:
                raise RuntimeError(
                    f"Too few successful steps: {stats['finite_steps']} / {stats['total_steps']}"
                )
    except Exception:
        log.error("Student training failed:\n%s", traceback.format_exc())
        if _is_main():
            print("STUDENT TRAINING FAILED — see .err for traceback", flush=True)
        sys.exit(1)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


def _auto_device() -> str:
    env = os.environ.get("US_SMOKE_DEVICE")
    if env:
        return env
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        return f"cuda:{local_rank}"
    return "cpu"


def _print_dataset_table(status: Dict[str, str]) -> None:
    W = 72
    print("\n" + "=" * W)
    print("DATASET MANIFEST COVERAGE")
    print("=" * W)
    for ds_id in sorted(status):
        print(f"  {ds_id:<50}  {status[ds_id]}")
    ok = sum(1 for v in status.values() if v.startswith("ok"))
    print(f"\n  Total with data: {ok} / {len(status)} configured datasets")
    print("=" * W + "\n")


def _print_pretrain_summary(stats: Dict) -> None:
    W = 72
    print("\n" + "=" * W)
    print("STUDENT PRETRAIN RUN COMPLETE")
    print("=" * W)
    print(f"  Steps           : {stats['total_steps']}")
    print(f"  Finite steps    : {stats['finite_steps']}")
    print(f"  Non-finite steps: {stats['nonfinite_steps']}")
    print(f"  Elapsed         : {stats['elapsed_s']:.1f}s  ({stats['steps_per_s']:.2f} step/s)")
    print(f"  Stage counts    : {json.dumps(stats['stage_counts'])}")
    print(f"  Checkpoints     : {stats.get('ckpt_dir', '')}")
    print(f"  Metrics         : {stats.get('log_dir', '')}")
    print("=" * W)


def _print_final_summary(stats: Dict, dataset_status: Dict[str, str]) -> None:
    W = 72
    print("\n" + "=" * W)
    print("STUDENT SMOKE RUN COMPLETE")
    print("=" * W)
    print(f"  Steps           : {stats['total_steps']}")
    print(f"  Finite steps    : {stats['finite_steps']}")
    print(f"  Non-finite steps: {stats['nonfinite_steps']}")
    print(f"  Elapsed         : {stats['elapsed_s']:.1f}s  ({stats['steps_per_s']:.2f} step/s)")
    print(f"  Stage counts    : {json.dumps(stats['stage_counts'])}")
    print(f"  Sample types    : {json.dumps(stats['type_counts'])}")
    ok = sum(1 for v in dataset_status.values() if v.startswith("ok"))
    print(f"  Datasets w/data : {ok}")
    print(f"  Checkpoints     : {stats.get('ckpt_dir', '')}")
    print(f"  Metrics         : {stats.get('log_dir', '')}")
    print("    stage1_end.pt  stage2_end.pt  stage3_end.pt  stage4_end.pt  latest.pt")
    print("=" * W)


if __name__ == "__main__":
    main()
