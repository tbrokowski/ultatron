"""
models/losses/proto_loss.py  ·  Prototype consistency loss
===============================================================

SwAV-style asymmetric prototype loss (Caron et al. 2020).

Design
------
The original symmetric cross-entropy had two problems:
  1. No collapse prevention — both sides could map to identical prototypes.
  2. Operated in misaligned backbone spaces (D=1024 native, not align_dim).

New design:
  1. Image teacher assigns prototypes via Sinkhorn-Knopp normalisation
     (balanced assignment — prototypes are used roughly equally).
  2. Video student predicts the assignment via softmax (sharp but unconstrained).
  3. Loss is one-directional: -Σ sinkhorn_target * log(student_pred).

This is semantically cleaner: the stable image teacher sets the target
label; the video student learns to predict which semantic prototype
describes the content, grounded in the image's spatial semantics.

Both modalities are projected to align_dim via cross_distill.proj_img/vid
BEFORE being passed here, so all proto logic operates in the shared space.

Distributed training
--------------------
Sinkhorn-Knopp requires a global view of the batch to produce a balanced
assignment (each of the K prototypes used equally across ALL samples, not
just those on one GPU).  swav_proto_loss_from_tokens therefore all-gathers
the image logits from all ranks before running Sinkhorn, then slices out
this rank's local target rows.  The video student logits are NOT gathered —
gradients flow through the local vid_logits only, which is correct under DDP.

Functions
---------
proto_assign            Soft prototype assignment (B, K) from (B, N, D) tokens.
sinkhorn                Iterative Sinkhorn-Knopp equalisation.
swav_proto_loss         Asymmetric SwAV: teacher assigns → student predicts.
proto_consistency_loss  (legacy) Symmetric cross-entropy, kept for backward compat.
proto_loss_from_tokens  (legacy) Symmetric wrapper called from old code paths.
swav_proto_loss_from_tokens  Convenience wrapper for phase3_step (distributed-aware).
"""
from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_gather_nograd(t: Tensor) -> Tensor:
    """Gather tensor from all ranks; no gradient through the result."""
    if not _is_dist() or dist.get_world_size() == 1:
        return t.detach()
    gathered = [torch.zeros_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t.contiguous())
    return torch.cat(gathered, dim=0)


def proto_assign(
    tokens: Tensor,       # (B, N, D)  patch tokens or tube tokens
    prototypes: Tensor,   # (K, D)     L2-normalised prototype vectors
    temperature: float = 0.1,
) -> Tensor:
    """
    Soft prototype assignment via cosine similarity.

    Parameters
    ----------
    tokens     : (B, N, D) or (B, D)  tokens to assign (mean-pooled if 3-D)
    prototypes : (K, D)     prototype matrix — L2-normalised before dot product
    temperature: float      softmax sharpness; lower = harder assignment

    Returns
    -------
    logits : (B, K) float32  raw (un-softmaxed) assignment scores
    """
    if tokens.dim() == 3:
        feat = F.normalize(tokens.float().mean(1), dim=-1)   # (B, D)
    else:
        feat = F.normalize(tokens.float(), dim=-1)           # (B, D)
    proto  = F.normalize(prototypes.float(), dim=-1)          # (K, D)
    logits = feat @ proto.T                                   # (B, K)
    return logits / temperature


def sinkhorn(
    logits: Tensor,    # (B, K)  raw assignment scores
    n_iters: int = 3,
    eps: float = 0.05,
) -> Tensor:
    """
    Sinkhorn-Knopp normalisation for balanced prototype assignment.

    Converts raw logit scores to a doubly-stochastic assignment matrix
    where each prototype is used roughly equally across the batch and
    each sample sums to 1 over prototypes.

    Following SwAV (Caron et al. 2020) — applied to the teacher side
    to produce a balanced, stable target distribution.

    Parameters
    ----------
    logits : (B, K)  raw assignment scores (before softmax)
    n_iters: int     number of Sinkhorn iterations (3 is sufficient)
    eps    : float   sharpness of the exponential (lower = sharper)

    Returns
    -------
    Q : (B, K) float32  doubly-normalised assignment (rows and cols sum to 1/B, 1/K)
    """
    Q = torch.exp(logits.float() / eps)   # (B, K)
    Q = Q / Q.sum()                       # global normalisation

    B, K = Q.shape
    for _ in range(n_iters):
        Q = Q / (Q.sum(dim=0, keepdim=True) * K)    # normalise columns (K prototypes equally)
        Q = Q / (Q.sum(dim=1, keepdim=True) * B)    # normalise rows (B samples)
    return Q


def swav_proto_loss(
    img_logits: Tensor,    # (B, K)  image teacher raw assignment scores
    vid_logits: Tensor,    # (B, K)  video student raw assignment scores
    temperature: float = 0.1,
    n_sinkhorn_iters: int = 3,
    eps: float = 1e-8,
) -> Tensor:
    """
    Asymmetric SwAV prototype loss.

    target     = sinkhorn(img_logits)         — balanced, stable image assignment
    prediction = softmax(vid_logits / temp)   — video student prediction

    L = -Σ_b Σ_k  target[b,k] * log(prediction[b,k])

    The image teacher side uses Sinkhorn (stop-gradient enforced by caller via
    torch.no_grad on teacher forward pass). The video student side uses a standard
    softmax so gradients flow back through the video branch.

    Operates on min(B_img, B_vid) samples.
    """
    B = min(img_logits.shape[0], vid_logits.shape[0])
    if B == 0:
        return img_logits.new_tensor(0.0)

    img_l = img_logits[:B]
    vid_l = vid_logits[:B]

    with torch.no_grad():
        target = sinkhorn(img_l, n_sinkhorn_iters)    # (B, K) balanced distribution

    prediction = F.softmax(vid_l, dim=-1)              # (B, K) student prediction

    loss = -(target * (prediction + eps).log()).sum(dim=-1).mean()
    return loss


def swav_proto_loss_from_tokens(
    img_tokens: Tensor,    # (B_img, N, D) or (B_img, D) — projected to align_dim
    vid_tokens: Tensor,    # (B_vid, M, D) or (B_vid, D) — projected to align_dim
    prototypes: Tensor,    # (K, D)
    temperature: float = 0.1,
) -> Tensor:
    """
    Distributed-aware SwAV wrapper for phase3_step.
    Tokens are already projected to align_dim by cross_distill.proj_img/vid.

    Sinkhorn strategy (distributed):
        - All-gather image logits across ranks before Sinkhorn so the balanced
          assignment uses the full global batch (required for K prototypes to be
          used equally when B_local << K).
        - Slice this rank's local rows from the global Sinkhorn target.
        - Video logits are NOT gathered — gradients flow through local vid only.

    Single-GPU / non-distributed: identical to the original per-rank behaviour.
    """
    img_logits_local = proto_assign(img_tokens, prototypes, temperature)   # (B_local, K)
    vid_logits_local = proto_assign(vid_tokens, prototypes, temperature)   # (B_local, K)

    if _is_dist() and dist.get_world_size() > 1:
        # Gather image logits from all ranks for a globally balanced Sinkhorn
        img_logits_global = _all_gather_nograd(img_logits_local)   # (B_global, K)

        with torch.no_grad():
            target_global = sinkhorn(img_logits_global)            # (B_global, K)

        # Extract this rank's slice of the global target
        rank     = dist.get_rank()
        B_local  = img_logits_local.shape[0]
        start    = rank * B_local
        target   = target_global[start : start + B_local]         # (B_local, K)
    else:
        with torch.no_grad():
            target = sinkhorn(img_logits_local)                    # (B_local, K)

    B = min(target.shape[0], vid_logits_local.shape[0])
    if B == 0:
        return img_tokens.new_tensor(0.0)

    prediction = F.softmax(vid_logits_local[:B], dim=-1)           # (B_local, K)
    eps        = 1e-8
    loss = -(target[:B] * (prediction + eps).log()).sum(dim=-1).mean()
    return loss


# ── Legacy symmetric functions (kept for backward compatibility) ───────────────

def proto_consistency_loss(
    p_img: Tensor,   # (B_img, K)  image prototype distribution
    p_vid: Tensor,   # (B_vid, K)  video prototype distribution
    eps: float = 1e-8,
) -> Tensor:
    """
    Symmetric cross-entropy between image and video prototype distributions.
    L = -0.5 * [Σ p_img * log(p_vid) + Σ p_vid * log(p_img)]
    Kept for backward compatibility; prefer swav_proto_loss for new code.
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
    Legacy symmetric wrapper.  Kept for backward compatibility.
    New code should call swav_proto_loss_from_tokens instead.
    """
    p_img = F.softmax(proto_assign(img_tokens, prototypes, temperature), dim=-1)
    p_vid = F.softmax(proto_assign(vid_tokens, prototypes, temperature), dim=-1)
    return proto_consistency_loss(p_img, p_vid)
