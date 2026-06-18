"""
finetune/backbones/student_encoder.py  ·  Hiera student checkpoint encoder
==========================================================================

Loads StudentPretrainPilot / student-pretrain checkpoints (student, ema_student keys)
and exposes the BackboneEncoder interface used by FinetuneExperiment.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch import Tensor

from finetune.backbones.base import BackboneEncoder

log = logging.getLogger(__name__)


def _default_student_cfg_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs" / "student" / "student_pretrain_pilot.yaml"
    )


def _load_student_cfg_dict(
    ckpt: dict,
    student_cfg_override: Optional[dict] = None,
) -> dict:
    if student_cfg_override:
        return dict(student_cfg_override)
    if isinstance(ckpt.get("model_cfg"), dict):
        return dict(ckpt["model_cfg"])
    import yaml
    cfg_path = _default_student_cfg_path()
    if cfg_path.exists():
        with open(cfg_path) as f:
            full = yaml.safe_load(f)
        return dict(full.get("model", {}).get("student", {}))
    return {}


class StudentEncoder(BackboneEncoder):
    """
    Hiera student backbone loaded from a student-pipeline checkpoint.

    Parameters
    ----------
    checkpoint : str
        Path to .pt file (stage1_end.pt, latest.pt, etc.)
    label : str
        Display name for comparison reports.
    use_ema : bool
        Load ema_student weights when True, else student.
    device : str
    student_cfg : dict or None
        Optional StudentModelConfig fields; defaults to ckpt model_cfg or
        configs/student/student_pretrain_pilot.yaml.
    """

    def __init__(
        self,
        checkpoint:    str,
        label:         str = "student",
        use_ema:       bool = True,
        device:        str = "cpu",
        student_cfg:   Optional[dict] = None,
    ):
        from models.student.student_config import StudentModelConfig, build_student_encoder

        ckpt_path = Path(checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Student checkpoint not found: {checkpoint}")

        log.info("[%s] Loading student checkpoint: %s", label, checkpoint)
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

        cfg_dict = _load_student_cfg_dict(ckpt, student_cfg)
        model_cfg = StudentModelConfig.from_dict(cfg_dict)

        try:
            from data.infra.cscs_paths import CSCSConfig
            cscs = CSCSConfig.from_env()
            _scratch_hf = cscs.scratch_path("hf_cache")
            _store_hf = cscs.store_path("hf_cache")
            hf_cache = str(_scratch_hf if _scratch_hf.exists() else _store_hf)
        except Exception:
            import os
            hf_cache = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
        if not model_cfg.hiera_hf_cache_dir:
            model_cfg.hiera_hf_cache_dir = hf_cache

        self._label = label
        self.backbone = build_student_encoder(model_cfg, device=device)

        state_key = "ema_student" if use_ema else "student"
        state = ckpt.get(state_key)
        if state is None:
            alt = "student" if use_ema else "ema_student"
            log.warning("[%s] %s missing — falling back to %s", label, state_key, alt)
            state = ckpt.get(alt)
        if state is None:
            raise KeyError(
                f"Checkpoint {checkpoint!r} has no 'student' or 'ema_student' key."
            )

        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if missing:
            log.warning("[%s] Missing keys: %d", label, len(missing))
        if unexpected:
            log.warning("[%s] Unexpected keys: %d", label, len(unexpected))

        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.backbone.eval()

        step = ckpt.get("step", ckpt.get("global_step", "?"))
        stage = ckpt.get("stage", ckpt.get("current_phase", "?"))
        log.info("[%s] Loaded step=%s stage=%s (use_ema=%s)", label, step, stage, use_ema)

    @property
    def name(self) -> str:
        return self._label

    @property
    def embed_dim(self) -> int:
        return self.backbone.hidden_size

    @property
    def embed_dims(self) -> list[int]:
        """Per-stage Hiera widths [D1, D2, D3, D4] for UPerNetDecoder."""
        return list(self.backbone.embed_dims)

    @property
    def video_embed_dim(self) -> int:
        return self.backbone.hidden_size

    @property
    def is_video_native(self) -> bool:
        return True

    @torch.no_grad()
    def encode_image(
        self,
        images: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> dict:
        """Encode (B, C, H, W) images as single-frame clips."""
        if images.dim() == 3:
            images = images.unsqueeze(0)
        x = images.unsqueeze(1)  # (B, 1, C, H, W)
        out = self.backbone(x, padding_mask=padding_mask)
        return {
            "cls":          out["global"],
            "patch_tokens": out["F4"][:, 0],   # (B, N4, D4) legacy / cls path
            "F1":           out["F1"],         # (B, 1, N1, D1) for UPerNetDecoder
            "F2":           out["F2"],
            "F3":           out["F3"],
            "F4":           out["F4"],
        }

    @torch.no_grad()
    def encode_video(self, clips: Tensor) -> dict:
        """Encode (B, T, C, H, W) video clips."""
        if clips.dim() == 4:
            clips = clips.unsqueeze(0)
        out = self.backbone(clips)
        f4 = out["F4"]  # (B, T, N4, D4)
        B, T, N, D = f4.shape
        tube_tokens  = f4.reshape(B, T * N, D)
        frame_tokens = f4.mean(dim=2)                       # (B, T, D4) per-frame global
        return {
            "clip_cls":     out["global"],
            "frame_tokens": frame_tokens,
            "tube_tokens":  tube_tokens,
        }

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        return iter([])

    def to(self, *args, **kwargs) -> "StudentEncoder":
        self.backbone.to(*args, **kwargs)
        return self

    def eval(self) -> "StudentEncoder":
        self.backbone.eval()
        return self

    def train(self, mode: bool = True) -> "StudentEncoder":
        self.backbone.eval()
        return self
