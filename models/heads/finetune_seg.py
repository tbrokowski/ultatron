"""
models/heads/finetune_seg.py  ·  Finetune segmentation head routing
====================================================================

Comparison / finetune protocol (frozen backbone, trainable head only):

  Student / Hiera (4-scale F1–F4):
      head_type=upernet  → UPerNetDecoder (SegAdapters + ASPP + attention gates)
      head_type=dpt      → EnhancedDPTSegHead on F4 tokens (single-scale ablation)

  All other frozen backbones (ViT, ResNet, CLIP, US FMs, …):
      head_type=dpt      → EnhancedDPTSegHead on patch_tokens
      head_type=upernet  → skipped (not hierarchical)

  ``linear`` remains available as a linear-probe ablation baseline.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

from .hierarchical_seg import UPerNetDecoder, build_hierarchical_seg_head
from .segmentation_head import (
    DPTSegHead,
    EnhancedDPTSegHead,
    LinearSegHead,
    build_seg_head,
)


_HIERARCHICAL_HEAD_TYPES = frozenset({"upernet", "hierarchical", "multiscale"})


def encoder_has_hierarchical_features(encoder: Any) -> bool:
    embed_dims = getattr(encoder, "embed_dims", None)
    return isinstance(embed_dims, (list, tuple)) and len(embed_dims) == 4


def hierarchical_feature_dict(feats: dict) -> dict:
    """UPerNet input from ``encode_image`` (requires F1–F4)."""
    missing = [k for k in ("F1", "F2", "F3", "F4") if k not in feats]
    if missing:
        raise KeyError(
            f"Hierarchical seg head requires F1–F4 in encoder output; missing {missing}"
        )
    return {k: feats[k] for k in ("F1", "F2", "F3", "F4")}


def is_hierarchical_seg_head(head: nn.Module) -> bool:
    return isinstance(head, UPerNetDecoder)


def is_enhanced_dpt_head(head: nn.Module) -> bool:
    return isinstance(head, EnhancedDPTSegHead)


def head_type_requires_hierarchy(head_type: str) -> bool:
    return head_type.lower() in _HIERARCHICAL_HEAD_TYPES


def filter_head_types_for_encoder(head_types: list[str], encoder: Any) -> list[str]:
    """Drop hierarchical head types when the encoder lacks F1–F4 multi-scale features."""
    filtered: list[str] = []
    for head_type in head_types:
        if (
            head_type_requires_hierarchy(head_type)
            and not encoder_has_hierarchical_features(encoder)
        ):
            log.warning(
                "Skipping %s — %s requires a hierarchical encoder (%s)",
                head_type,
                head_type,
                type(encoder).__name__,
            )
            continue
        filtered.append(head_type)
    return filtered


def seg_encoder_dim(encoder: Any, embed_dim: Optional[int] = None) -> int:
    """Token width for dense heads (patch_embed_dim when it differs from CLS dim)."""
    if embed_dim is not None:
        return embed_dim
    return getattr(encoder, "patch_embed_dim", getattr(encoder, "embed_dim", 768))


def _build_enhanced_dpt(
    embed_dim: int,
    n_classes: int,
    patch_size: int,
    output_size: Optional[int],
    neck_channels: int,
    use_adapters: bool,
    use_aspp: bool,
    use_refine_up: bool,
) -> EnhancedDPTSegHead:
    return EnhancedDPTSegHead(
        embed_dim=embed_dim,
        n_classes=n_classes,
        patch_size=patch_size,
        neck_channels=neck_channels,
        output_size=output_size,
        use_adapters=use_adapters,
        use_aspp=use_aspp,
        use_refine_up=use_refine_up,
    )


def build_finetune_seg_head(
    encoder: Any,
    n_classes: int,
    head_type: str = "dpt",
    patch_size: int = 16,
    output_size: Optional[int] = None,
    fpn_channels: int = 256,
    neck_channels: int = 256,
    embed_dim: Optional[int] = None,
    use_refine_up: bool = False,
    use_adapters: bool = True,
    use_aspp: bool = True,
    use_attention_gates: bool = True,
) -> nn.Module:
    """
    Build a segmentation head matched to the encoder family.

    head_type
    ---------
    linear       : LinearSegHead (ablation baseline)
    dpt          : EnhancedDPTSegHead on patch_tokens (F4 for student)
    upernet / hierarchical / multiscale : UPerNetDecoder (student / Hiera only)
    """
    ht = head_type.lower()
    embed_dim = seg_encoder_dim(encoder, embed_dim)

    if ht == "linear":
        return LinearSegHead(embed_dim, n_classes, patch_size)

    if ht in _HIERARCHICAL_HEAD_TYPES:
        if not encoder_has_hierarchical_features(encoder):
            log.warning(
                "head_type=%r requires a hierarchical encoder (%s); using dpt instead",
                head_type,
                type(encoder).__name__,
            )
            ht = "dpt"
        else:
            return build_hierarchical_seg_head(
                list(encoder.embed_dims),
                n_classes,
                fpn_channels,
                use_adapters=use_adapters,
                use_attention_gates=use_attention_gates,
                use_aspp=use_aspp,
                use_refine_up=use_refine_up,
            )

    if ht == "dpt":
        return _build_enhanced_dpt(
            embed_dim,
            n_classes,
            patch_size,
            output_size,
            neck_channels,
            use_adapters,
            use_aspp,
            use_refine_up,
        )

    if ht == "dpt_legacy":
        return DPTSegHead(
            embed_dim,
            n_classes,
            patch_size,
            neck_channels=neck_channels,
            output_size=output_size,
        )

    raise ValueError(
        f"Unknown segmentation head_type={head_type!r}. "
        "Choose 'linear', 'dpt', 'dpt_legacy', or 'upernet'."
    )


def forward_seg_head(
    head: nn.Module,
    feats: dict,
    padding_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run seg head on encoder features (UPerNet or patch-token head)."""
    if is_hierarchical_seg_head(head):
        return head(hierarchical_feature_dict(feats), padding_mask=padding_mask)
    return head(feats["patch_tokens"], padding_mask=padding_mask)
