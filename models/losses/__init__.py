"""
oura/models/losses/__init__.py
All SSL and supervised loss functions.  Pure functions — no nn.Module state.
Learnable parameters live in models/branches/shared.py (CrossBranchDistillation,
PrototypeHead) and models/heads/.  These modules only compute loss values.
"""
from .image_losses import (
    dino_cls_loss,
    dino_cls_loss_multicrop,
    ibot_patch_loss,
    koleo_loss,
)
from .video_losses import (
    jepa_tube_loss,
    clip_cls_loss,
    video_ssl_loss,
)
from .cross_branch import (
    cross_branch_loss,
    cross_branch_loss_from_tokens,
)
from .proto_loss import (
    proto_assign,
    proto_consistency_loss,
    proto_loss_from_tokens,
)

__all__ = [
    # Image
    "dino_cls_loss",
    "dino_cls_loss_multicrop",
    "ibot_patch_loss",
    "koleo_loss",
    # Video
    "jepa_tube_loss",
    "clip_cls_loss",
    "video_ssl_loss",
    # Cross-branch
    "cross_branch_loss",
    "cross_branch_loss_from_tokens",
    # Prototype
    "proto_assign",
    "proto_consistency_loss",
    "proto_loss_from_tokens",
]
