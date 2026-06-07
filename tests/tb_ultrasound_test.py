"""
tests/tb_ultrasound_test.py  ·  MONAI SwinTransformer → TB prediction (video-based)
=====================================================================================

Integrates the MONAI SwinUNETR encoder backbone into the existing video-based
LUS patient TB prediction pipeline (GatedAttentionMIL head).

Architecture
------------
Each patient is a *bag* of video clips (matching LUSPatientBagDataset):

    Patient bag: N clips × (T frames × 3 × H × W)
         ↓  SwinFrameEncoder  (per-frame 2D Swin, then mean-pool over T)
    N clip embeddings: (N, D=2048)
         ↓  GatedAttentionMIL  (from finetune/experiments/lus_patient.py)
    Scalar TB logit  →  sigmoid  →  TB probability

Swin backbone (MONAI SwinTransformer, spatial_dims=2, use_v2=True):
    Input: (B, 3, H, W)   — one frame per batch entry
    Output: 5 feature maps:
        x_outs[0]: (B, 128,  H/2,  W/2)   feature_size × 1
        x_outs[1]: (B, 256,  H/4,  W/4)   feature_size × 2
        x_outs[2]: (B, 512,  H/8,  W/8)   feature_size × 4
        x_outs[3]: (B, 1024, H/16, W/16)  feature_size × 8
        x_outs[4]: (B, 2048, H/32, W/32)  feature_size × 16
    We take x_outs[-1] and global-average-pool → (B, 2048) clip embedding.

Weight loading (echocare_encoder.pt)
-------------------------------------
Load via load_echocare_weights(encoder, path).  The checkpoint is expected
to contain the SwinTransformer state_dict directly (or under a "state_dict"
key).  mask_token is stripped if present (it is SSL-only).  Will be wired in
once the checkpoint file is available.

Run (no real data needed):
    python tests/tb_ultrasound_test.py

Run as pytest:
    pytest tests/tb_ultrasound_test.py -v
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

FEATURE_SIZE    = 128          # Swin embed_dim; final stage = 16 × 128 = 2048
IN_CHANNELS     = 3
CLIP_EMBED_DIM  = FEATURE_SIZE * 16   # 2048  — after global-average-pool of stage-4
MIL_HIDDEN_DIM  = 256
N_FRAMES        = 8
CLIP_SIZE       = 256
CLIP_ENCODE_BS  = 4            # frames processed in one Swin forward pass


# ── Swin frame encoder ─────────────────────────────────────────────────────────

class SwinFrameEncoder(nn.Module):
    """
    Wraps the MONAI SwinTransformer (2-D, SwinV2) to produce per-clip embeddings
    suitable for the GatedAttentionMIL TB head.

    Forward path for one clip (T frames):
        frames  (T, 3, H, W)
            → Swin stage-4 feature map  (T, 2048, H/32, W/32)
            → global average pool       (T, 2048)
            → mean over T frames        (2048,)          ← clip embedding

    For a full patient bag with N clips:
        bag  (N, T, 3, H, W)  → encoded  (N, 2048)
    """

    def __init__(
        self,
        feature_size:   int  = FEATURE_SIZE,
        in_channels:    int  = IN_CHANNELS,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        from monai.networks.nets.swin_unetr import SwinTransformer

        self.encoder = SwinTransformer(
            in_chans       = in_channels,
            embed_dim      = feature_size,
            window_size    = [8, 8],
            patch_size     = [2, 2],
            depths         = [2, 2, 18, 2],
            num_heads      = [4, 8, 16, 32],
            mlp_ratio      = 4.0,
            qkv_bias       = True,
            use_checkpoint = use_checkpoint,
            spatial_dims   = 2,
            use_v2         = True,
        )
        self.embed_dim  = feature_size * 16   # stage-4 channels
        self.hidden_size = self.embed_dim     # alias for compatibility

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of frames through the Swin backbone.

        Args:
            frames: (B, 3, H, W)
        Returns:
            (B, embed_dim)  — global-average-pooled stage-4 features
        """
        x_outs = self.encoder(frames)    # list of 5 feature maps
        feat   = x_outs[-1]             # (B, 2048, h, w)
        return feat.mean(dim=[-2, -1])  # (B, 2048)  global average pool

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Encode a single clip by processing frames in mini-batches.

        Args:
            clip: (T, 3, H, W)  — frames for one clip
        Returns:
            (embed_dim,)  — mean-pooled clip embedding
        """
        T = clip.shape[0]
        frame_embs = []
        for start in range(0, T, CLIP_ENCODE_BS):
            frames = clip[start : start + CLIP_ENCODE_BS]   # (bs, 3, H, W)
            frame_embs.append(self.encode_frames(frames))
        return torch.cat(frame_embs, dim=0).mean(dim=0)     # (embed_dim,)

    def encode_bag(
        self,
        clips: torch.Tensor,
        encode_bs: int = 1,
    ) -> torch.Tensor:
        """
        Encode all clips for one patient (the MIL bag).

        Args:
            clips:     (N, T, 3, H, W)
            encode_bs: how many clips to encode at once (memory budget)
        Returns:
            (N, embed_dim)
        """
        N = clips.shape[0]
        bag_embs = []
        for i in range(0, N, encode_bs):
            sub = clips[i : i + encode_bs]   # (bs, T, 3, H, W)
            bs  = sub.shape[0]
            # Process each clip independently
            clip_embs = torch.stack(
                [self.forward(sub[j]) for j in range(bs)], dim=0
            )                                # (bs, embed_dim)
            bag_embs.append(clip_embs)
        return torch.cat(bag_embs, dim=0)    # (N, embed_dim)


# ── Weight loading ─────────────────────────────────────────────────────────────

ECHOCARE_CKPT = Path(__file__).resolve().parents[1] / "echocare_encoder.pth"


def load_echocare_weights(
    encoder:    SwinFrameEncoder,
    ckpt_path:  str | Path = ECHOCARE_CKPT,
    strict:     bool = True,
) -> None:
    """
    Load pre-trained echocare weights into the Swin backbone.

    Checkpoint format (echocare_encoder.pth):
      - Top-level object: flat OrderedDict (358 keys, no nesting)
      - Key namespace matches MONAI SwinTransformer exactly
      - Contains 'mask_token' (SSL pre-training artefact) which is stripped

    Args:
        encoder:   SwinFrameEncoder instance
        ckpt_path: path to echocare_encoder.pth
                   (defaults to <repo_root>/echocare_encoder.pth)
        strict:    passed to load_state_dict
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"echocare_encoder.pth not found at {ckpt_path}.\n"
            "Expected at the repo root: Ultatron/echocare_encoder.pth"
        )

    log.info(f"Loading echocare encoder weights from {ckpt_path} ...")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Strip SSL-only key absent from inference-time SwinTransformer
    state_dict.pop("mask_token", None)

    encoder.encoder.load_state_dict(state_dict, strict=strict)
    log.info(
        f"echocare encoder loaded: {len(state_dict)} tensors, strict={strict}"
    )


# ── TB predictor (backbone + MIL head) ────────────────────────────────────────

class TBSwinPredictor(nn.Module):
    """
    Full TB predictor: SwinFrameEncoder backbone + GatedAttentionMIL head.

    The backbone processes video clips frame-by-frame (2-D Swin).
    The MIL head aggregates clip embeddings into a patient-level TB logit.

    Usage:
        model = TBSwinPredictor()
        # load echocare pre-trained weights (auto-loaded if checkpoint exists):
        # load_echocare_weights(model.backbone)   # uses repo-root default path

        # Patient bag: N clips, each T frames at CLIP_SIZE × CLIP_SIZE
        clips = torch.rand(5, 8, 3, 256, 256)   # (N=5, T=8, 3, 256, 256)
        logit = model(clips)                     # scalar TB logit
        prob  = torch.sigmoid(logit)             # TB probability ∈ [0, 1]
    """

    def __init__(
        self,
        feature_size:  int  = FEATURE_SIZE,
        in_channels:   int  = IN_CHANNELS,
        mil_hidden_dim: int = MIL_HIDDEN_DIM,
        use_checkpoint: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.backbone = SwinFrameEncoder(
            feature_size   = feature_size,
            in_channels    = in_channels,
            use_checkpoint = use_checkpoint,
        )

        # Import the existing GatedAttentionMIL head — no duplication
        from finetune.experiments.lus_patient import GatedAttentionMIL

        self.head = GatedAttentionMIL(
            embed_dim  = self.backbone.embed_dim,
            hidden_dim = mil_hidden_dim,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def encode_bag(self, clips: torch.Tensor) -> torch.Tensor:
        """Encode patient bag with frozen backbone. clips: (N, T, 3, H, W)."""
        self.backbone.eval()
        return self.backbone.encode_bag(clips)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Patient-level TB forward pass.

        Args:
            clips: (N, T, 3, H, W)  — all clips for one patient
        Returns:
            scalar logit (pre-sigmoid)
        """
        H = self.encode_bag(clips)   # (N, embed_dim)
        return self.head(H)          # scalar

    def predict_proba(self, clips: torch.Tensor) -> float:
        """Convenience wrapper → TB probability ∈ [0, 1]."""
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(clips)).item()


# ── Tests / smoke checks ───────────────────────────────────────────────────────

def test_swin_backbone_output_shapes():
    """Verify the raw MONAI SwinTransformer produces the expected 5-level pyramid."""
    from monai.networks.nets.swin_unetr import SwinTransformer

    encoder = SwinTransformer(
        in_chans       = IN_CHANNELS,
        embed_dim      = FEATURE_SIZE,
        window_size    = [8, 8],
        patch_size     = [2, 2],
        depths         = [2, 2, 18, 2],
        num_heads      = [4, 8, 16, 32],
        mlp_ratio      = 4.0,
        qkv_bias       = True,
        use_checkpoint = False,   # off for fast unit test
        spatial_dims   = 2,
        use_v2         = True,
    )
    encoder.eval()

    x      = torch.rand(1, IN_CHANNELS, CLIP_SIZE, CLIP_SIZE)
    x_outs = encoder(x)

    expected_shapes = [
        (1, 128,  128, 128),
        (1, 256,   64,  64),
        (1, 512,   32,  32),
        (1, 1024,  16,  16),
        (1, 2048,   8,   8),
    ]
    actual_shapes = [tuple(o.shape) for o in x_outs]

    print("\nSwin feature pyramid:")
    for i, (exp, got) in enumerate(zip(expected_shapes, actual_shapes)):
        status = "✓" if exp == got else "✗"
        print(f"  stage {i}: expected {exp}  got {got}  {status}")
        assert exp == got, f"stage {i} shape mismatch: expected {exp}, got {got}"

    print("test_swin_backbone_output_shapes PASSED\n")


def test_swin_frame_encoder():
    """SwinFrameEncoder: single clip → (embed_dim,) clip embedding."""
    enc = SwinFrameEncoder(feature_size=FEATURE_SIZE, use_checkpoint=False)
    enc.eval()

    clip  = torch.rand(N_FRAMES, IN_CHANNELS, CLIP_SIZE, CLIP_SIZE)
    emb   = enc(clip)

    assert emb.shape == (CLIP_EMBED_DIM,), \
        f"Expected ({CLIP_EMBED_DIM},), got {emb.shape}"
    assert torch.isfinite(emb).all(), "Clip embedding contains NaN/Inf"
    print(f"test_swin_frame_encoder PASSED — embedding shape: {emb.shape}")


def test_swin_bag_encoding():
    """SwinFrameEncoder.encode_bag: (N, T, 3, H, W) → (N, embed_dim)."""
    enc   = SwinFrameEncoder(feature_size=FEATURE_SIZE, use_checkpoint=False)
    N_clips = 3
    clips = torch.rand(N_clips, N_FRAMES, IN_CHANNELS, CLIP_SIZE, CLIP_SIZE)

    enc.eval()
    with torch.no_grad():
        bag_embs = enc.encode_bag(clips)

    assert bag_embs.shape == (N_clips, CLIP_EMBED_DIM), \
        f"Expected ({N_clips}, {CLIP_EMBED_DIM}), got {bag_embs.shape}"
    print(f"test_swin_bag_encoding PASSED — bag shape: {bag_embs.shape}")


def test_tb_swin_predictor():
    """
    Full end-to-end TB prediction:
      patient bag (N clips) → scalar TB logit → sigmoid TB probability.
    """
    model = TBSwinPredictor(
        feature_size    = FEATURE_SIZE,
        mil_hidden_dim  = MIL_HIDDEN_DIM,
        use_checkpoint  = False,
        freeze_backbone = True,
    )
    model.eval()

    N_clips = 4
    clips   = torch.rand(N_clips, N_FRAMES, IN_CHANNELS, CLIP_SIZE, CLIP_SIZE)

    with torch.no_grad():
        logit = model(clips)
        prob  = torch.sigmoid(logit)

    assert logit.ndim == 0, f"Expected scalar logit, got shape {logit.shape}"
    assert 0.0 <= prob.item() <= 1.0, f"Probability out of range: {prob.item()}"

    print(f"\ntest_tb_swin_predictor PASSED")
    print(f"  Input:  {N_clips} clips × {N_FRAMES} frames × "
          f"{IN_CHANNELS}×{CLIP_SIZE}×{CLIP_SIZE}")
    print(f"  Logit:  {logit.item():.4f}")
    print(f"  TB prob: {prob.item():.4f}")


def test_load_echocare_weights():
    """Load echocare_encoder.pth into SwinFrameEncoder — strict=True must pass."""
    if not ECHOCARE_CKPT.exists():
        import pytest
        pytest.skip(f"echocare_encoder.pth not found at {ECHOCARE_CKPT}")

    enc = SwinFrameEncoder(feature_size=FEATURE_SIZE, use_checkpoint=False)
    load_echocare_weights(enc, ECHOCARE_CKPT, strict=True)

    # Spot-check: patch_embed weight should not be random after loading
    w = enc.encoder.patch_embed.proj.weight
    assert w.abs().sum().item() > 0, "patch_embed weights are all zero after load"
    print(f"\ntest_load_echocare_weights PASSED — {len(list(enc.parameters()))} param tensors loaded")


def test_head_attention_weights():
    """GatedAttentionMIL.attention_weights returns per-clip scores summing to 1."""
    from finetune.experiments.lus_patient import GatedAttentionMIL

    head = GatedAttentionMIL(embed_dim=CLIP_EMBED_DIM, hidden_dim=MIL_HIDDEN_DIM)
    head.eval()

    N   = 6
    H   = torch.rand(N, CLIP_EMBED_DIM)
    attn = head.attention_weights(H)

    assert attn.shape == (N,), f"Expected ({N},), got {attn.shape}"
    assert abs(attn.sum().item() - 1.0) < 1e-5, \
        f"Attention weights do not sum to 1: {attn.sum().item():.6f}"
    print(f"\ntest_head_attention_weights PASSED")
    print(f"  Attention weights: {attn.detach().cpu().numpy().round(4)}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TB Ultrasound Swin test")
    p.add_argument("--feature_size",          default=FEATURE_SIZE, type=int)
    p.add_argument("--in_channels",           default=IN_CHANNELS,  type=int)
    p.add_argument("--pretrained_checkpoint", default=str(ECHOCARE_CKPT),
                   help="Path to echocare_encoder.pth "
                        f"(default: {ECHOCARE_CKPT})")
    p.add_argument("--use_checkpoint",        default=False, type=bool,
                   help="Gradient checkpointing (saves VRAM, slower for smoke tests)")
    p.add_argument("--n_clips",               default=4,    type=int)
    p.add_argument("--n_frames",              default=N_FRAMES, type=int)
    p.add_argument("--clip_size",             default=CLIP_SIZE, type=int)
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    print("=" * 60)
    print("TB-Ultrasound Swin Backbone Test")
    print("=" * 60)

    # ── 1. Backbone shape smoke test ──────────────────────────────────────────
    print("\n[1/4] Swin feature pyramid shapes ...")
    test_swin_backbone_output_shapes()

    # ── 2. Per-clip encoding ──────────────────────────────────────────────────
    print("[2/4] SwinFrameEncoder single clip ...")
    test_swin_frame_encoder()

    # ── 3. Patient bag encoding ───────────────────────────────────────────────
    print("[3/4] Patient bag encoding ...")
    test_swin_bag_encoding()

    # ── 4. Full TB predictor ──────────────────────────────────────────────────
    print("[4/4] Full TBSwinPredictor forward pass ...")

    model = TBSwinPredictor(
        feature_size    = args.feature_size,
        mil_hidden_dim  = MIL_HIDDEN_DIM,
        use_checkpoint  = args.use_checkpoint,
        freeze_backbone = True,
    )

    # Load echocare weights (auto-loads from repo root if file exists)
    ckpt = Path(args.pretrained_checkpoint)
    if ckpt.exists():
        load_echocare_weights(model.backbone, ckpt)
        print(f"  echocare weights loaded from {ckpt}")
    else:
        print(f"  WARNING: {ckpt} not found — running with random weights.")

    model.eval()
    clips = torch.rand(
        args.n_clips, args.n_frames,
        args.in_channels, args.clip_size, args.clip_size,
    )
    with torch.no_grad():
        logit = model(clips)
        prob  = torch.sigmoid(logit)

    print(f"\n  backbone embed_dim : {model.backbone.embed_dim}")
    print(f"  MIL hidden_dim     : {MIL_HIDDEN_DIM}")
    print(f"  Input bag          : {args.n_clips} clips × {args.n_frames} frames "
          f"× {args.in_channels}×{args.clip_size}×{args.clip_size}")
    print(f"  TB logit           : {logit.item():.4f}")
    print(f"  TB probability     : {prob.item():.4f}")
    print("\nAll checks passed.")

    # ── 5. Attention weight interpretability ─────────────────────────────────
    test_head_attention_weights()


if __name__ == "__main__":
    main()
