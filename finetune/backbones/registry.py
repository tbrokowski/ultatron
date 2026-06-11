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
"""
from __future__ import annotations

import logging
from typing import Optional

from finetune.backbones.base import BackboneEncoder

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
    BackboneEncoder (ready to call .eval() and .to(device))
    """
    btype = spec.get("type", spec.get("key", ""))
    key   = spec.get("key", btype)
    log.info(f"Building encoder: key={key!r}  type={btype!r}")

    # ── Ours (Ultatron checkpoints) ───────────────────────────────────────────
    if btype == "ours":
        from finetune.backbones.ultatron_encoder import UltatronEncoder
        if train_cfg is None:
            raise ValueError("train_cfg required for type='ours' encoders.")
        return UltatronEncoder(
            train_cfg  = train_cfg,
            checkpoint = spec["checkpoint"],
            label      = key,
            device     = device,
        )

    # ── Standalone V-JEPA2 ────────────────────────────────────────────────────
    if btype in ("vjepa", "vjepa2", "vjepa2_l", "vjepa2_h", "vjepa2_g"):
        from finetune.backbones.vjepa_encoder import VJEPAEncoder
        variant = spec.get("variant", key if key.startswith("vjepa2") else "vjepa2_l")
        return VJEPAEncoder(
            variant      = variant,
            checkpoint   = spec.get("checkpoint"),
            device       = device,
            hf_cache_dir = spec.get("hf_cache_dir"),
        )

    # ── Standard torchvision models ───────────────────────────────────────────
    if btype == "standard" or key == "resnet50":
        from finetune.backbones.standard_encoders import ResNet50Encoder
        return ResNet50Encoder(pretrained=spec.get("pretrained", True))

    if btype == "standard" or key in ("vit_b_16", "vit"):
        from finetune.backbones.standard_encoders import ViTEncoder
        return ViTEncoder(pretrained=spec.get("pretrained", True))

    # Explicit key-based dispatch for standard type with multiple models
    if btype == "standard":
        _variant = spec.get("variant", key)
        if "resnet" in _variant.lower():
            from finetune.backbones.standard_encoders import ResNet50Encoder
            return ResNet50Encoder(pretrained=spec.get("pretrained", True))
        else:
            from finetune.backbones.standard_encoders import ViTEncoder
            return ViTEncoder(pretrained=spec.get("pretrained", True))

    # ── BioMed-CLIP ───────────────────────────────────────────────────────────
    if btype == "biomedclip" or key == "biomedclip":
        from finetune.backbones.biomedclip_encoder import BioMedCLIPEncoder
        return BioMedCLIPEncoder(
            hf_model_id = spec.get("hf_model_id",
                "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"),
            cache_dir   = spec.get("hf_cache_dir"),
        )

    # ── USFM ─────────────────────────────────────────────────────────────────
    if btype == "usfm" or key == "usfm":
        from finetune.backbones.usfm_encoder import USFMEncoder
        return USFMEncoder(
            checkpoint = spec["checkpoint"],
            embed_dim  = spec.get("embed_dim", 768),
        )

    # ── EchoCare ──────────────────────────────────────────────────────────────
    if btype == "echocare" or key == "echocare":
        from finetune.backbones.echocare_encoder import EchoCareEncoder
        return EchoCareEncoder(
            checkpoint = spec["checkpoint"],
            embed_dim  = spec.get("embed_dim", 768),
            is_video   = spec.get("is_video", True),
        )

    # ── OpenUS ────────────────────────────────────────────────────────────────
    if btype == "openus" or key == "openus":
        from finetune.backbones.openus_encoder import OpenUSEncoder
        return OpenUSEncoder(
            checkpoint = spec["checkpoint"],
            embed_dim  = spec.get("embed_dim", 768),
        )

    raise ValueError(
        f"Unknown backbone type {btype!r} (key={key!r}). "
        f"Valid types: ours, vjepa, standard, biomedclip, usfm, echocare, openus."
    )
