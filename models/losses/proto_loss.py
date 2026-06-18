"""
models/losses/proto_loss.py  ·  Prototype consistency loss
===============================================================

SwAV-style asymmetric prototype loss (Caron et al. 2020).

Mathematical framework
----------------------
Given features f (L2-normalised, D-dim) and prototypes C (K × D, L2-normalised):

  raw cosine similarities:  s = f · C^T   ∈ [-1, 1]^{B × K}

Teacher target (balanced assignment):
  Q = Sinkhorn-Knopp( exp(s / ε_sinkhorn) )       ε = 0.05  (sharp)

Student prediction:
  p = softmax( s / τ_prediction )                  τ = 0.1   (softer)

Loss:
  L = -Σ_b Σ_k  Q[b,k] · log p[b,k]   (cross-entropy, averaged over batch)

Key invariant: ε < τ  (teacher assigns sharper than student predicts).
The two temperatures are intentionally INDEPENDENT — sinkhorn() receives
logits already divided by ε_sinkhorn; it does NOT apply any further scaling.

Distributed training
--------------------
Sinkhorn-Knopp requires a global view of the batch to produce a balanced
assignment (each of the K prototypes used roughly equally across ALL samples).
swav_proto_loss_from_tokens therefore all-gathers the teacher logits from all
ranks before running Sinkhorn, then slices out this rank's local target rows.
Video student logits are NOT gathered — gradients flow through local vid only.

Requirement: B_global ≥ K.  When this is not satisfied (e.g. video micro-batch
B=1 × 32 ranks = 32 < K=256) Sinkhorn cannot balance and the function returns
0.0 rather than producing a degenerate loss.
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
    tokens: Tensor,       # (B, N, D) or (B, D)
    prototypes: Tensor,   # (K, D)
    temperature: float = 0.1,
) -> Tensor:
    """
    Soft prototype assignment via cosine similarity, scaled by temperature.

    Returns (B, K) raw logits = cos_sim / temperature.
    Call this for the STUDENT (softmax) path only.
    The teacher (Sinkhorn) path computes raw cosine sims independently at its
    own temperature (sinkhorn_eps) inside swav_proto_loss_from_tokens.
    """
    if tokens.dim() == 3:
        feat = F.normalize(tokens.float().mean(1), dim=-1)   # (B, D)
    else:
        feat = F.normalize(tokens.float(), dim=-1)           # (B, D)
    proto = F.normalize(prototypes.float(), dim=-1)          # (K, D)
    return (feat @ proto.T) / temperature                    # (B, K)


def sinkhorn(
    logits: Tensor,    # (B, K)  — already scaled by 1/sinkhorn_eps
    n_iters: int = 3,
) -> Tensor:
    """
    Sinkhorn-Knopp normalisation for balanced prototype assignment.

    Input: logits that have ALREADY been divided by sinkhorn_eps (e.g. 0.05).
    This function applies NO additional temperature scaling.

    Returns Q: (B, K) approximately doubly-stochastic assignment where
      Σ_k Q[b,k] = 1/B  for all b   (rows: sample weight uniform)
      Σ_b Q[b,k] = 1/K  for all k   (cols: prototype utilisation balanced)

    The max-shift before exp is an invariant transformation that prevents
    float overflow without changing the Sinkhorn output.
    """
    x = logits.float()
    x = x - x.max()            # shift for numerical stability (exp(x - max) ∈ (0, 1])
    Q = torch.exp(x)           # (B, K)
    Q = Q / Q.sum()            # global normalisation: Σ_{b,k} Q[b,k] = 1

    B, K = Q.shape
    for _ in range(n_iters):
        # Column normalisation: each prototype used with equal total weight 1/K
        Q = Q / (Q.sum(dim=0, keepdim=True) * K)
        # Row normalisation: each sample contributes equal total weight 1/B
        Q = Q / (Q.sum(dim=1, keepdim=True) * B)
    return Q


def swav_proto_loss_from_tokens(
    img_tokens: Tensor,            # (B, N, D) or (B, D) — teacher tokens, already stopgrad
    vid_tokens: Tensor,            # (B, M, D) or (B, D) — student tokens
    prototypes: Tensor,            # (K, D)
    temperature: float = 0.1,     # τ: student softmax temperature
    sinkhorn_eps: float = 0.05,   # ε: teacher Sinkhorn temperature (< τ → sharper target)
    n_sinkhorn_iters: int = 3,
) -> Tensor:
    """
    Distributed-aware asymmetric SwAV prototype loss.

    Teacher target  = Sinkhorn( exp(cos_sim / sinkhorn_eps) )   [stop-gradient]
    Student predict = softmax(  cos_sim / temperature       )   [gradient flows]
    L = -Σ Q * log(p) averaged over the batch.

    Temperature invariant:  sinkhorn_eps < temperature
      (teacher targets are sharper than student predictions, following SwAV §3.2)

    Returns 0.0 when B_global < K (insufficient batch for balanced assignment).
    """
    # ── Compute L2-normalised features ──────────────────────────────────────
    def _pool_and_norm(t: Tensor) -> Tensor:
        v = t.float()
        if v.dim() == 3:
            v = v.mean(1)
        return F.normalize(v, dim=-1)

    feat_img = _pool_and_norm(img_tokens)   # (B_local, D)
    feat_vid = _pool_and_norm(vid_tokens)   # (B_local, D)
    proto    = F.normalize(prototypes.float(), dim=-1)   # (K, D)

    cos_img = feat_img @ proto.T   # (B_local, K)  raw cosine sims ∈ [-1, 1]
    cos_vid = feat_vid @ proto.T   # (B_local, K)

    # Teacher logits use sinkhorn_eps (sharper); student uses temperature (softer)
    img_logits_local = cos_img / sinkhorn_eps   # (B_local, K)  for Sinkhorn
    vid_logits_local = cos_vid / temperature    # (B_local, K)  for softmax

    # ── Compute Sinkhorn target globally across all ranks ────────────────────
    if _is_dist() and dist.get_world_size() > 1:
        img_logits_global = _all_gather_nograd(img_logits_local)   # (B_global, K)
        B_global, K = img_logits_global.shape
        if B_global < K:
            # Can't balance K prototypes with fewer than K samples — skip loss
            return img_tokens.new_tensor(0.0)
        with torch.no_grad():
            target_global = sinkhorn(img_logits_global, n_sinkhorn_iters)   # (B_global, K)
        B_local = img_logits_local.shape[0]
        rank    = dist.get_rank()
        start   = rank * B_local
        target  = target_global[start : start + B_local]   # (B_local, K)
    else:
        B_local, K = img_logits_local.shape
        if B_local < K:
            return img_tokens.new_tensor(0.0)
        with torch.no_grad():
            target = sinkhorn(img_logits_local, n_sinkhorn_iters)   # (B_local, K)

    B = min(target.shape[0], vid_logits_local.shape[0])
    if B == 0:
        return img_tokens.new_tensor(0.0)

    # Cross-entropy: -Σ_k Q[b,k] * log p[b,k], averaged over b.
    # Use log_softmax for numerical stability (avoids log(softmax + eps) approximation).
    log_p = F.log_softmax(vid_logits_local[:B], dim=-1)   # (B, K)
    loss  = -(target[:B] * log_p).sum(dim=-1).mean()
    return loss


class ProtoQueue:
    """
    FIFO teacher-logit queue for small-batch Sinkhorn.

    Motivation
    ----------
    Sinkhorn-Knopp requires B_global ≥ K to produce a balanced assignment.
    For video micro-batches (e.g. B_local=1 × 32 ranks = 32 < K=256) the
    global batch is too small.  Since V-JEPA is a FROZEN teacher its logit
    vectors for any clip are deterministic — there is no EMA drift — making
    a FIFO queue of past teacher logits much more stable than the MoCo queue.

    Usage
    -----
    queue = ProtoQueue(K=256, queue_size=512, device=device)

    # Inside the training step, AFTER computing teacher logits:
    loss = swav_proto_loss_with_queue(
        teacher_logits_local,   # (B_local, K)  already divided by sinkhorn_eps
        student_logits_local,   # (B_local, K)  already divided by temperature
        queue,
    )
    queue.enqueue(teacher_logits_local.detach())

    Notes
    -----
    - Queue is per-rank.  Before Sinkhorn we all-gather both the current
      logits AND the queue across ranks to restore global coverage.
    - `queue_size` should be ≥ K and a multiple of `B_local` for tidy math,
      but any value ≥ K works.
    - When the queue is not yet full (early training), the filled portion is
      used.  We only run Sinkhorn once there are ≥ K entries total.
    """

    def __init__(self, K: int, queue_size: int = 512, device=None) -> None:
        self.K          = K
        self.queue_size = queue_size
        dev = device if device is not None else torch.device("cpu")
        # Buffer stores up to queue_size logit rows; pointer wraps around.
        self._buf = torch.zeros(queue_size, K, device=dev)
        self._ptr = 0
        self._filled = 0   # how many valid rows are in the buffer

    @property
    def device(self):
        return self._buf.device

    def enqueue(self, logits: Tensor) -> None:
        """Add (B, K) rows to the queue, evicting the oldest when full."""
        logits = logits.detach().float().to(self._buf.device)
        B = logits.shape[0]
        if B >= self.queue_size:
            # Rare: batch larger than queue — just keep the latest entries
            self._buf[:] = logits[-self.queue_size:]
            self._ptr = 0
            self._filled = self.queue_size
            return
        end = self._ptr + B
        if end <= self.queue_size:
            self._buf[self._ptr:end] = logits
        else:
            split = self.queue_size - self._ptr
            self._buf[self._ptr:] = logits[:split]
            self._buf[:B - split] = logits[split:]
        self._ptr = end % self.queue_size
        self._filled = min(self._filled + B, self.queue_size)

    def get(self) -> Tensor:
        """Return the filled portion of the queue as a (N_filled, K) tensor."""
        if self._filled < self.queue_size:
            # Buffer not yet full — return only the valid rows
            if self._ptr <= self._filled:
                return self._buf[:self._filled]
            # Wrapped: valid rows are [ptr-filled:ptr] modulo queue_size
            # (shouldn't happen before full, but be safe)
        return self._buf.clone()

    def __len__(self) -> int:
        return self._filled


def swav_proto_loss_with_queue(
    teacher_logits_local: Tensor,   # (B_local, K)  already divided by sinkhorn_eps
    student_logits_local: Tensor,   # (B_local, K)  already divided by temperature
    queue: "ProtoQueue",
    n_sinkhorn_iters: int = 3,
) -> Tensor:
    """
    SwAV loss with a teacher-logit queue for small-batch video.

    Prepends queue entries to the current teacher logits, runs Sinkhorn on
    the augmented batch, then slices only the CURRENT step's rows as targets
    for the student.  Queue entries have no corresponding student predictions
    and are discarded after Sinkhorn.

    All-gathers across ranks (current logits only, not the full queue) so
    the global view is queue_per_rank × n_ranks + B_global_current.

    Call ``queue.enqueue(teacher_logits_local.detach())`` AFTER this function.
    """
    K = teacher_logits_local.shape[1]

    # Gather current teacher logits across ranks (no grad)
    if _is_dist() and dist.get_world_size() > 1:
        cur_global = _all_gather_nograd(teacher_logits_local)    # (B_global_cur, K)
    else:
        cur_global = teacher_logits_local.detach()

    # Prepend queue entries (local per-rank, already stable frozen-teacher logits)
    q_entries = queue.get()   # (N_q, K) — may be empty early in training
    if len(q_entries) > 0:
        augmented = torch.cat([q_entries.to(cur_global.device), cur_global], dim=0)
    else:
        augmented = cur_global

    B_aug = augmented.shape[0]
    if B_aug < K:
        # Queue not yet warm enough; skip this step
        return teacher_logits_local.new_tensor(0.0)

    with torch.no_grad():
        target_aug = sinkhorn(augmented, n_sinkhorn_iters)   # (B_aug, K)

    # Target rows for the CURRENT batch are the LAST B_global_cur entries
    B_global_cur = cur_global.shape[0]
    target_global_cur = target_aug[-B_global_cur:]           # (B_global_cur, K)

    # Slice this rank's local rows
    if _is_dist() and dist.get_world_size() > 1:
        B_local = teacher_logits_local.shape[0]
        rank    = dist.get_rank()
        start   = rank * B_local
        target  = target_global_cur[start : start + B_local]   # (B_local, K)
    else:
        target = target_global_cur

    B = min(target.shape[0], student_logits_local.shape[0])
    if B == 0:
        return teacher_logits_local.new_tensor(0.0)

    log_p = F.log_softmax(student_logits_local[:B], dim=-1)
    return -(target[:B] * log_p).sum(dim=-1).mean()


def swav_proto_loss(
    img_logits: Tensor,    # (B, K)  image teacher raw cosine-sim scores (NOT pre-scaled)
    vid_logits: Tensor,    # (B, K)  video student raw cosine-sim scores (NOT pre-scaled)
    temperature: float = 0.1,
    sinkhorn_eps: float = 0.05,
    n_sinkhorn_iters: int = 3,
) -> Tensor:
    """
    Standalone asymmetric SwAV loss from pre-computed raw cosine similarities.

    img_logits and vid_logits must be RAW cosine similarities (not yet divided
    by any temperature). This function applies sinkhorn_eps and temperature
    internally.

    Prefer swav_proto_loss_from_tokens for the main training path.
    """
    B = min(img_logits.shape[0], vid_logits.shape[0])
    if B == 0 or B < img_logits.shape[1]:
        return img_logits.new_tensor(0.0)

    img_l = (img_logits[:B] / sinkhorn_eps)
    vid_l = (vid_logits[:B] / temperature)

    with torch.no_grad():
        target = sinkhorn(img_l, n_sinkhorn_iters)

    log_p = F.log_softmax(vid_l, dim=-1)
    return -(target * log_p).sum(dim=-1).mean()


# ── Legacy symmetric functions (kept for backward compatibility) ──────────────

def proto_consistency_loss(
    p_img: Tensor,   # (B, K)  image prototype distribution (softmax output)
    p_vid: Tensor,   # (B, K)  video prototype distribution (softmax output)
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
    img_tokens: Tensor,    # (B, N, D)
    vid_tokens: Tensor,    # (B, M, D)
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
