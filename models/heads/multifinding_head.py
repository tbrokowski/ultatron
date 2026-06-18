"""
models/heads/multifinding_head.py  ·  Per-finding binary classification heads
===============================================================================

MultiFindingBinaryHead applies independent binary classifiers (one per finding)
to a shared clip-level embedding.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.heads.classification_head import LinearClsHead, MLPClsHead

DEFAULT_LUS_FINDINGS: Tuple[str, ...] = (
    "a_line",
    "b_line",
    "confluent_b_line",
    "pleural_effusion",
    "large_consolidation",
    "small_consolidation",
    "pneumothorax",
)


class MultiFindingBinaryHead(nn.Module):
    """
    Shared embedding → N independent binary heads (1 logit each).

    Output shape: (B, N_findings)
    """

    def __init__(
        self,
        embed_dim: int,
        finding_names: Sequence[str] = DEFAULT_LUS_FINDINGS,
        head_type: str = "mlp",
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.finding_names = tuple(finding_names)
        hidden = hidden_dim or embed_dim // 2

        heads = {}
        for name in self.finding_names:
            if head_type == "linear":
                heads[name] = LinearClsHead(embed_dim=embed_dim, n_classes=1)
            elif head_type == "mlp":
                heads[name] = MLPClsHead(
                    embed_dim=embed_dim,
                    n_classes=1,
                    hidden_dim=hidden,
                    dropout=dropout,
                )
            else:
                raise ValueError(
                    f"Unknown head_type: {head_type!r}. Choose 'linear' or 'mlp'."
                )
        self.heads = nn.ModuleDict(heads)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, D) → (B, N_findings)"""
        logits = [self.heads[name](x) for name in self.finding_names]
        return torch.cat(logits, dim=-1)

    def __repr__(self) -> str:
        head_cls = type(next(iter(self.heads.values()))).__name__
        return f"MultiFindingBinaryHead({head_cls}, n={len(self.finding_names)})"


def build_multifinding_head(
    embed_dim: int,
    finding_names: Sequence[str] = DEFAULT_LUS_FINDINGS,
    head_type: str = "mlp",
    **kwargs,
) -> MultiFindingBinaryHead:
    return MultiFindingBinaryHead(
        embed_dim=embed_dim,
        finding_names=finding_names,
        head_type=head_type,
        **kwargs,
    )
