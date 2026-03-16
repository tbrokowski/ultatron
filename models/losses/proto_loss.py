"""
models/losses/proto_loss.py  ·  Prototype consistency loss
===============================================================

Formalisation of the prototype consistency objective from
models/branches/shared.py (PrototypeHead.consistency_loss).

This module separates the loss *computation* from the learnable
*prototype parameters*.  The PrototypeHead nn.Module lives in
models/branches/shared.py and holds the K prototype vectors.
This file holds the pure loss functions called by phase_steps.py.

proto_assign
------------
Given patch tokens and a prototype matrix, compute the soft assignment
distribution (probability of each prototype) for each sample.

proto_consistency_loss
----------------------
Symmetric cross-entropy between image and video prototype distributions.
L_proto = -0.5 * [H(p_img, p_vid) + H(p_vid, p_img)]
where H(p, q) = -Σ p_i log(q_i)

This encourages image and video representations to map to the same
prototype distribution for corresponding (matched) samples.

Loss weight
-----------
λ7 in the full loss:
    L_total = L_img + L_vid + λ6·L_cross + λ7·L_proto + λ_gram·L_gram
λ7 is 0.0 in Stage 1, ramped to 0.5 in Stage 2–3.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def proto_assign(
    tokens: Tensor,       # (B, N, D)  patch tokens or tube tokens
    prototypes: Tensor,   # (K, D)     L2-normalised prototype vectors
    temperature: float = 0.1,
) -> Tensor:
    """
    Soft prototype assignment via cosine similarity.

    Parameters
    ----------
    tokens     : (B, N, D)  tokens to assign (mean-pooled internally)
    prototypes : (K, D)     prototype matrix — L2-normalised before dot product
    temperature: float      softmax sharpness; lower = harder assignment

    Returns
    -------
    p : (B, K) float32  soft assignment probabilities (sum to 1 over K)
    """
    feat  = F.normalize(tokens.float().mean(1), dim=-1)   # (B, D)
    proto = F.normalize(prototypes.float(), dim=-1)        # (K, D)
    logits = feat @ proto.T                                # (B, K)
    return F.softmax(logits / temperature, dim=-1)


def proto_consistency_loss(
    p_img: Tensor,   # (B_img, K)  image prototype distribution
    p_vid: Tensor,   # (B_vid, K)  video prototype distribution
    eps: float = 1e-8,
) -> Tensor:
    """
    Symmetric cross-entropy between image and video prototype distributions.

    L = -0.5 * [Σ p_img * log(p_vid) + Σ p_vid * log(p_img)]

    Operates on min(B_img, B_vid) samples.  When batch sizes differ
    (image and video batch sizes differ in config), aligns on the smaller.

    Returns a scalar loss tensor.
    """
    B = min(p_img.shape[0], p_vid.shape[0])
    if B == 0:
        return p_img.new_tensor(0.0)

    p_i = p_img[:B]
    p_v = p_vid[:B]

    loss_iv = -(p_i * (p_v + eps).log()).sum(-1).mean()
    loss_vi = -(p_v * (p_i + eps).log()).sum(-1).mean()
    return (loss_iv + loss_vi) / 2.0


def proto_loss_from_tokens(
    img_tokens: Tensor,    # (B_img, N, D)
    vid_tokens: Tensor,    # (B_vid, M, D)
    prototypes: Tensor,    # (K, D)
    temperature: float = 0.1,
) -> Tensor:
    """
    Convenience wrapper: assign tokens then compute consistency loss.

    Called by phase_steps.py when raw backbone tokens are available.
    The prototypes tensor is proto_head.prototypes from PrototypeHead.
    """
    p_img = proto_assign(img_tokens, prototypes, temperature)
    p_vid = proto_assign(vid_tokens, prototypes, temperature)
    return proto_consistency_loss(p_img, p_vid)
