"""
finetune/backbones/registry.py  ·  BackboneEncoder factory
===========================================================

build_encoder(spec, device, train_cfg) → BackboneEncoder

``spec`` is one entry from the ``backbones:`` list in comparison.yaml:

    - key: ours_phase3
      type: ours
      checkpoint: /path/to/phase3_end.pt

    - key: vjepa2_l
      type: vjepa
      # checkpoint: optional path to standalone V-JEPA2 weights

    - key: resnet50
      type: standard

    - key: biomedclip
      type: biomedclip

    - key: usfm
      type: usfm
      checkpoint: /path/to/USFM_latest.pth
      embed_dim: 768       # optional; default 768

    - key: echocare
      type: echocare
      checkpoint: /path/to/echocare.pth
      embed_dim: 768

    - key: openus
      type: openus
      checkpoint: /path/to/openus.pth
      embed_dim: 768

    - key: dinov3_b
      type: dinov3
      variant: dinov3_b
"""
from __future__ import annotations

import logging
from typing import Optional

from finetune.backbones.base import BackboneEncoder
from finetune.backbones.paths import ablation_weight_path

log = logging.getLogger(__name__)


def build_encoder(
    spec:      dict,
    device:    str = "cuda",
    train_cfg: Optional[dict] = None,
) -> BackboneEncoder:
    """
    Instantiate a BackboneEncoder from a backbone spec dict.

    Parameters
    ----------
    spec : dict
        Must have at minimum ``key`` and ``type`` fields.
    device : str
    train_cfg : dict or None
        The ``model:`` section of the training YAML.  Required for
        type='ours' and optionally used for type='vjepa'.

    Returns
    -------
    BackboneEncoder (ready for inference on *device*)
    """
    btype = spec.get("type", spec.get("key", ""))
    key   = spec.get("key", btype)
    log.info(f"Building encoder: key={key!r}  type={btype!r}")

    encoder: BackboneEncoder

    # ── Hiera student (StudentSmoke / student-pretrain checkpoints) ───────────
    if btype == "student":
        from finetune.backbones.student_encoder import StudentEncoder
        encoder = StudentEncoder(
            checkpoint  = spec["checkpoint"],
            label       = key,
            use_ema     = spec.get("use_ema", True),
            device      = device,
            student_cfg = spec.get("student_cfg"),
        )

    # ── Ours (Ultatron checkpoints) ───────────────────────────────────────────
    elif btype == "ours":
        from finetune.backbones.ultatron_encoder import UltatronEncoder
        if train_cfg is None:
            raise ValueError("train_cfg required for type='ours' encoders.")
        encoder = UltatronEncoder(
            train_cfg  = train_cfg,
            checkpoint = spec["checkpoint"],
            label      = key,
            device     = device,
        )

    # ── Standalone V-JEPA2 ────────────────────────────────────────────────────
    elif btype in ("vjepa", "vjepa2", "vjepa2_l", "vjepa2_h", "vjepa2_g"):
        from finetune.backbones.vjepa_encoder import VJEPAEncoder
        variant = spec.get("variant", key if key.startswith("vjepa2") else "vjepa2_l")
        encoder = VJEPAEncoder(
            variant      = variant,
            checkpoint   = spec.get("checkpoint"),
            device       = device,
            hf_cache_dir = spec.get("hf_cache_dir"),
        )

    # ── Standard torchvision models ───────────────────────────────────────────
    elif btype == "standard":
        _variant = spec.get("variant", key)
        _pretrained = spec.get("pretrained", True)
        if "resnet" in _variant.lower():
            from finetune.backbones.standard_encoders import ResNet50Encoder
            encoder = ResNet50Encoder(pretrained=_pretrained)
        else:
            from finetune.backbones.standard_encoders import ViTEncoder
            encoder = ViTEncoder(pretrained=_pretrained)

    # ── BioMed-CLIP ───────────────────────────────────────────────────────────
    elif btype == "biomedclip" or key == "biomedclip":
        from finetune.backbones.biomedclip_encoder import BioMedCLIPEncoder
        encoder = BioMedCLIPEncoder(
            hf_model_id = spec.get("hf_model_id",
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
            cache_dir   = spec.get("hf_cache_dir"),
        )

    # ── USFM ─────────────────────────────────────────────────────────────────
    elif btype == "usfm" or key == "usfm":
        from finetune.backbones.usfm_encoder import USFMEncoder
        encoder = USFMEncoder(
            checkpoint = spec.get(
                "checkpoint",
                ablation_weight_path("USFM_latest.pth", "US_USFM_CHECKPOINT"),
            ),
            embed_dim  = spec.get("embed_dim", 768),
        )

    # ── EchoCare ──────────────────────────────────────────────────────────────
    elif btype == "echocare" or key == "echocare":
        from finetune.backbones.echocare_encoder import EchoCareEncoder
        encoder = EchoCareEncoder(
            checkpoint = spec.get(
                "checkpoint",
                ablation_weight_path("echocare_encoder.pth", "US_ECHOCARE_CHECKPOINT"),
            ),
            embed_dim  = spec.get("embed_dim", 2048),
            is_video   = spec.get("is_video", False),
        )

    # ── OpenUS ────────────────────────────────────────────────────────────────
    elif btype == "openus" or key == "openus":
        from finetune.backbones.openus_encoder import OpenUSEncoder
        encoder = OpenUSEncoder(
            checkpoint = spec.get(
                "checkpoint",
                ablation_weight_path("openus_cpt0150.pth", "US_OPENUS_CHECKPOINT"),
            ),
            vmamba_checkpoint = spec.get(
                "vmamba_checkpoint",
                ablation_weight_path(
                    "vssm_small_0229_ckpt_epoch_222.pth",
                    "US_OPENUS_VMAMBA_CHECKPOINT",
                ),
            ),
            embed_dim         = spec.get("embed_dim", 768),
        )

    # ── DINOv3 ────────────────────────────────────────────────────────────────
    elif btype == "dinov3" or str(key).startswith("dinov3"):
        from finetune.backbones.dinov3_encoder import DINOv3Encoder
        variant = spec.get("variant", key if str(key).startswith("dinov3") else "dinov3_b")
        encoder = DINOv3Encoder(
            variant      = variant,
            hf_cache_dir = spec.get("hf_cache_dir"),
        )

    else:
        raise ValueError(
            f"Unknown backbone type {btype!r} (key={key!r}). "
            f"Valid types: student, ours, vjepa, standard, dinov3, biomedclip, usfm, echocare, openus."
        )

    return encoder.to(device).eval()
