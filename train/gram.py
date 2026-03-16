"""
oura/train/gram.py  ·  Gram anchoring (DINOv3)
===============================================

Extracted from training_integration.py into its own module so it can be
imported by phase_steps.py without pulling in the entire training scaffold.

GramTeacher
-----------
Maintains a *hard snapshot* of the student backbone for Gram anchoring.
Unlike the EMA teacher (updated every step), the Gram teacher is a
deepcopy frozen at a specific training step and replaced every
gram_refresh_interval steps.

gram_loss
---------
L_gram = (1/B) Σ_b ||X_S_b @ X_S_b.T  −  X_G_b @ X_G_b.T||_F²

where X_S and X_G are L2-normalised patch token matrices.
Padding tokens are zeroed before computing Gram matrices.

Purpose
-------
Prevents patch-token locality degradation during long training.
As the student learns, its representations drift — the Gram matrix
of patch similarities should remain consistent with an earlier
snapshot.  This acts as a soft anchor, not a hard constraint.

Activated at gram_start_step (default 100k = 33% through training).
Refreshed every gram_refresh_interval steps (default 50k).
"""
from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class GramTeacher:
    """
    Frozen hard-snapshot of the student backbone for Gram anchoring.

    Not an nn.Module — it holds a deepcopy of the backbone and is not
    part of the trainable graph.

    Usage
    -----
        gram = GramTeacher(img_branch.student, start_step=100_000)

        # Inside the step function (after student forward):
        if gram.is_active(global_step):
            gram.maybe_refresh(img_branch.student, global_step)
            X_S = F.normalize(student_patches, dim=-1)  # (B, N, D)
            X_G = gram.forward(pixels, padding_mask)     # (B, N, D)
            loss_gram = gram_loss(X_S, X_G, padding_mask)
    """

    def __init__(
        self,
        student_backbone,
        gram_start_step: int = 100_000,
        gram_refresh_interval: int = 50_000,
    ):
        self.gram_start_step      = gram_start_step
        self.gram_refresh_interval= gram_refresh_interval
        self._snapshot: Optional[nn.Module] = None
        self._last_refresh: int = -1

    def is_active(self, global_step: int) -> bool:
        return global_step >= self.gram_start_step

    def maybe_refresh(self, student_backbone, global_step: int):
        """Hard-copy student weights into the Gram snapshot."""
        if not self.is_active(global_step):
            return
        steps_since = global_step - self._last_refresh
        if self._snapshot is None or steps_since >= self.gram_refresh_interval:
            self._snapshot = copy.deepcopy(student_backbone)
            for p in self._snapshot.parameters():
                p.requires_grad_(False)
            self._snapshot.eval()
            self._last_refresh = global_step
            log.info(f"GramTeacher refreshed at step {global_step}")

    @torch.no_grad()
    def forward(
        self,
        pixel_values: torch.Tensor,              # (B, 3, H, W) RGB
        padding_mask: Optional[torch.Tensor] = None,  # (B, ph, pw)
    ) -> torch.Tensor:
        """
        Run the frozen snapshot and return L2-normalised patch tokens.

        Returns
        -------
        X_G : (B, N, D)  L2-normalised, padding positions zeroed.
        """
        if self._snapshot is None:
            raise RuntimeError(
                "GramTeacher.forward() called before snapshot exists. "
                "Call maybe_refresh() first."
            )
        out    = self._snapshot(pixel_values, padding_mask=padding_mask)
        tokens = out["patch_tokens"]                 # (B, N, D)
        tokens = F.normalize(tokens, dim=-1)

        if padding_mask is not None:
            flat = padding_mask.flatten(1).unsqueeze(-1).float()  # (B, N, 1)
            tokens = tokens * flat

        return tokens


def gram_loss(
    X_S: torch.Tensor,              # (B, N, D)  student patch tokens, L2-normed
    X_G: torch.Tensor,              # (B, N, D)  Gram teacher tokens, L2-normed
    padding_mask: Optional[torch.Tensor] = None,  # (B, ph, pw)
) -> torch.Tensor:
    """
    L_gram = (1/B) Σ_b ||X_S_b @ X_S_b.T  −  X_G_b @ X_G_b.T||_F²

    Padding tokens are zeroed before Gram matrix computation.
    """
    if padding_mask is not None:
        flat = padding_mask.flatten(1).unsqueeze(-1).float()  # (B, N, 1)
        X_S  = X_S * flat
        X_G  = X_G * flat

    G_S  = torch.bmm(X_S, X_S.transpose(1, 2))   # (B, N, N)
    G_G  = torch.bmm(X_G, X_G.transpose(1, 2))

    diff = G_S - G_G
    return (diff * diff).sum(dim=(1, 2)).mean()
