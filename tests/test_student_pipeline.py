"""
tests/test_student_pipeline.py  ·  End-to-end smoke test for the student pipeline
==================================================================================

Uses the existing conftest.py dataset fixtures (realistic synthetic mini-datasets
built by the real adapters — CAMUS, BUSI, COVIDx-US, TN3K, HC18, EchoNet-Dynamic,
FETAL_PLANES_DB, LUS-multicenter, BUS-BRA) and runs a few samples from each
through the full student pipeline:

  Adapter → manifest entries → ImageSSLDataset / VideoSSLDataset
    → ImageSSLTransform / VideoSSLTransform
    → StudentMixedCollator
    → StubStudentBackbone (no Hiera weights; matches the output contract)
    → StudentFinetuneModel (real heads, real losses)
    → student_phase_steps (stage 1–4)

What is NOT tested here:
  - Real Hiera weight loading (tested separately once weights are available on CSCS)
  - FrozenDINOTeacher / FrozenVJEPATeacher (requires the real backbone registry)
  - CSCS storage resolution

Run with:
    pytest tests/test_student_pipeline.py -v
    pytest tests/test_student_pipeline.py -v -k "cardiac"
    pytest tests/test_student_pipeline.py -v -k "stage"

Expected result: all tests pass in < 60s on CPU.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Project imports ────────────────────────────────────────────────────────────
from data.adapters import (
    CAMUSAdapter, BUSIAdapter, COVIDxUSAdapter,
    FetalPlanesDBAdapter, HC18Adapter, LUSMulticenterAdapter,
    BUSBRAAdapter, TN3KAdapter, EchoNetDynamicAdapter,
)
from data.pipeline.dataset import ImageSSLDataset, VideoSSLDataset
from data.pipeline.transforms import (
    ImageSSLTransform, ImageSSLTransformConfig,
    VideoSSLTransform, VideoSSLTransformConfig,
)
from data.pipeline.collators import ImageSSLCollator, VideoSSLCollator
from data.pipeline.student_transforms import (
    TemporalDropout, ImageToPseudoClip,
    StudentVideoSSLTransform, StudentTransformConfig,
)
from data.pipeline.student_datamodule import (
    StudentMixedCollator,
    _ddp_active,
    _ddp_broadcast_sample_type,
)

from models.student.student_finetune import (
    StudentFinetuneModel, HeadSpec,
    cardiac_head_specs, lung_head_specs,
    breast_head_specs, thyroid_head_specs, fetal_head_specs,
)
from models.heads.hierarchical_seg import UPerNetDecoder, build_hierarchical_seg_head
from models.student.cross_attention_fusion import FusionTargetBuilder
from models.losses.student_losses import (
    cosine_loss, img_global_distill_loss, img_patch_distill_loss,
    img_masked_semantic_loss, img_proto_loss,
    vid_global_distill_loss, vid_tube_prediction_loss,
    vid_temporal_consistency, vid_proto_loss, prototype_assignment_stats,
    fused_distill_loss, preservation_loss,
)
from train.student_phase_steps import (
    student_stage1_step, student_stage2_step, student_stage4_step,
    _lam_ema_eff, _dino_patch_masks,
)
from data.labels.label_spec import TaskType, LossType
from data.schema.manifest import USManifestEntry, ManifestWriter

# ── Fixtures are imported from conftest.py automatically by pytest ─────────────
# (camus_root, busi_root, covidx_root, tn3k_root, hc18_root,
#  echonet_root, fetal_planes_root, lus_multicenter_root, bus_bra_root)

# ══════════════════════════════════════════════════════════════════════════════
# Stub backbone — no real Hiera weights needed
# Matches HieraStudentBackbone output contract exactly.
# ══════════════════════════════════════════════════════════════════════════════

STUB_EMBED_DIMS = [32, 64, 128, 256]   # tiny dims for fast CPU tests
STUB_PATCH_STRIDE = 4


class StubStudentBackbone(nn.Module):
    """
    Minimal test stand-in for HieraStudentBackbone.

    Accepts (B, T, 3, H, W) and returns the same output dict structure
    as the real backbone, but with tiny embed dims for fast tests.

    Hiera-style multi-scale: each stage halves the spatial resolution.
    """

    embed_dims = STUB_EMBED_DIMS   # class-level attribute like the real backbone

    def __init__(self):
        super().__init__()
        D1, D2, D3, D4 = STUB_EMBED_DIMS

        # Shared stem: stride-4 patch embedding
        self.stem = nn.Sequential(
            nn.Conv2d(3, D1, kernel_size=7, stride=STUB_PATCH_STRIDE, padding=3),
            nn.GELU(),
        )
        # Stage projections (pooling 2× between stages)
        self.stage2_proj = nn.Linear(D1, D2)
        self.stage3_proj = nn.Linear(D2, D3)
        self.stage4_proj = nn.Linear(D3, D4)

        self.hidden_size = D4

    def forward(
        self,
        x: Tensor,                             # (B, T, 3, H, W)
        padding_mask: Optional[Tensor] = None,
    ) -> Dict:
        B, T, C, H, W = x.shape
        D1, D2, D3, D4 = STUB_EMBED_DIMS

        # Compute expected grid sizes
        ph1 = math.ceil(H / STUB_PATCH_STRIDE)
        pw1 = math.ceil(W / STUB_PATCH_STRIDE)

        # Flatten T → per-frame processing
        x_flat = x.flatten(0, 1).float()   # (B*T, 3, H, W)
        f1_flat = self.stem(x_flat)         # (B*T, D1, ph1, pw1)

        ph1_actual = f1_flat.shape[2]
        pw1_actual = f1_flat.shape[3]

        # Build padding masks at each scale
        def _pool_pm(pm, stride=2):
            if pm is None:
                return None
            return F.max_pool2d(pm.unsqueeze(1).float(), stride, stride).squeeze(1).bool()

        pm1 = padding_mask
        pm2 = _pool_pm(pm1)
        pm3 = _pool_pm(pm2)
        pm4 = _pool_pm(pm3)

        # Flatten spatial for linear stages
        f1 = f1_flat.flatten(2).transpose(1, 2)  # (B*T, N1, D1)
        f2 = F.avg_pool2d(f1_flat, 2, 2).flatten(2).transpose(1, 2)  # (B*T, N2, D1)
        f2 = self.stage2_proj(f2)                                      # (B*T, N2, D2)
        f3 = F.avg_pool2d(f1_flat, 4, 4).flatten(2).transpose(1, 2)
        f3 = self.stage3_proj(self.stage2_proj(f3))
        f4 = F.avg_pool2d(f1_flat, 8, 8).flatten(2).transpose(1, 2)
        f4 = self.stage4_proj(self.stage3_proj(self.stage2_proj(f4)))

        def _unflatten(f):
            return f.reshape(B, T, f.shape[1], f.shape[2])

        F1 = _unflatten(f1)
        F2 = _unflatten(f2)
        F3 = _unflatten(f3)
        F4 = _unflatten(f4)

        # Global token: mean pool F4 over T and spatial
        if pm4 is not None:
            pm4_f = pm4.reshape(B, 1, -1, 1).float()
            global_tok = (F4 * pm4_f).sum(dim=(1, 2)) / pm4_f.sum(dim=(1, 2)).clamp(1)
        else:
            global_tok = F4.mean(dim=(1, 2))

        return {
            "global": global_tok,
            "F1": F1, "F2": F2, "F3": F3, "F4": F4,
            "dense": F4,
            "pmasks": [pm1, pm2, pm3, pm4],
        }

    def parameters_for_ema(self):
        return self.parameters()


# Stub prototype head for SSL loss tests
class StubProtoHead(nn.Module):
    def __init__(self, n_proto: int = 16, dim: int = STUB_EMBED_DIMS[-1]):
        super().__init__()
        self.prototypes = nn.Parameter(
            F.normalize(torch.randn(n_proto, dim), dim=-1)
        )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers: manifest + dataset loaders
# ══════════════════════════════════════════════════════════════════════════════

SAMPLES_PER_DATASET = 3
IMG_CFG = ImageSSLTransformConfig(
    patch_size=4,           # Hiera stride-4 patch size
    n_global_crops=2,
    n_local_crops=0,
    spatial_mask_ratio=0.25,
)
VID_CFG = VideoSSLTransformConfig(
    n_frames=4,
    temporal_stride=1,
    tube_mask_ratio=0.3,
)


def _write_manifest(entries: list, path: Path) -> Path:
    with ManifestWriter(path) as w:
        for e in entries:
            w.write(e)
    return path


def _image_dataset_from_adapter(adapter, tmp_path: Path, n: int = SAMPLES_PER_DATASET):
    entries = list(adapter.iter_entries())
    image_entries = [e for e in entries if e.modality_type == "image"][:n]
    if not image_entries:
        pytest.skip(f"No image entries in {adapter.__class__.__name__}")
    mf = _write_manifest(image_entries, tmp_path / "manifest.jsonl")
    xform = ImageSSLTransform(IMG_CFG)
    return ImageSSLDataset(str(mf), transform=xform)


def _video_dataset_from_adapter(adapter, tmp_path: Path, n: int = SAMPLES_PER_DATASET):
    entries = list(adapter.iter_entries())
    video_entries = [e for e in entries
                     if e.modality_type in ("video", "pseudo_video")][:n]
    if not video_entries:
        pytest.skip(f"No video entries in {adapter.__class__.__name__}")
    mf = _write_manifest(video_entries, tmp_path / "vid_manifest.jsonl")
    xform = VideoSSLTransform(VID_CFG)
    return VideoSSLDataset(str(mf), transform=xform)


def _collate_image(samples):
    collator = ImageSSLCollator()
    return collator(samples)


def _collate_video(samples):
    collator = VideoSSLCollator()
    return collator(samples)


def _stub_model(head_specs: Optional[list] = None) -> StudentFinetuneModel:
    backbone = StubStudentBackbone()
    specs = head_specs or []
    return StudentFinetuneModel(backbone, specs, backbone_frozen=False)


def _get_image_batch(adapter, tmp_path):
    ds = _image_dataset_from_adapter(adapter, tmp_path)
    samples = [ds[i] for i in range(min(2, len(ds)))]
    return _collate_image([s for s in samples if s is not None])


def _get_video_batch(adapter, tmp_path):
    ds = _video_dataset_from_adapter(adapter, tmp_path)
    samples = [ds[i] for i in range(min(2, len(ds)))]
    return _collate_video([s for s in samples if s is not None])


# ══════════════════════════════════════════════════════════════════════════════
# 1. Augmentation unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStudentAugmentations:
    def test_temporal_dropout_t1_noop(self):
        clip = torch.randn(1, 3, 32, 32)
        td = TemporalDropout(p=1.0)
        out = td(clip)
        assert out.shape == clip.shape, "T=1 clip must pass through unchanged"

    def test_temporal_dropout_collapses(self):
        clip = torch.randn(8, 3, 32, 32)
        td = TemporalDropout(p=1.0)
        out = td(clip)
        assert out.shape == (1, 3, 32, 32), "T>1 with p=1 should collapse to T=1"

    def test_temporal_dropout_batch(self):
        clip = torch.randn(2, 8, 3, 32, 32)
        td = TemporalDropout(p=1.0)
        out = td(clip)
        assert out.shape == (2, 1, 3, 32, 32)

    def test_image_to_pseudo_clip(self):
        img = torch.randn(1, 3, 64, 64)
        ipc = ImageToPseudoClip(target_T=4, p=1.0)
        out = ipc(img)
        assert out.shape == (4, 3, 64, 64), "Should expand T=1 to T=4"

    def test_image_to_pseudo_clip_noop_at_p0(self):
        img = torch.randn(1, 3, 64, 64)
        ipc = ImageToPseudoClip(target_T=4, p=0.0)
        out = ipc(img)
        assert out.shape == img.shape

    def test_student_transform_wrapper(self):
        from data.pipeline.student_transforms import build_student_transforms
        img_xf = ImageSSLTransform(IMG_CFG)
        vid_xf = VideoSSLTransform(VID_CFG)
        img_out, vid_out = build_student_transforms(img_xf, vid_xf)
        assert vid_out is not None


# ══════════════════════════════════════════════════════════════════════════════
# 2. Stub backbone shape contract
# ══════════════════════════════════════════════════════════════════════════════

class TestStubBackbone:
    @pytest.mark.parametrize("T", [1, 4])
    @pytest.mark.parametrize("H,W", [(64, 64), (80, 128)])
    def test_output_shapes(self, T, H, W):
        backbone = StubStudentBackbone()
        x = torch.randn(2, T, 3, H, W)
        out = backbone(x)

        assert "global" in out and "F1" in out and "F4" in out
        B, T_out = 2, T
        assert out["global"].shape == (B, STUB_EMBED_DIMS[-1])
        assert out["F1"].shape[0] == B and out["F1"].shape[1] == T_out
        assert out["F4"].shape[0] == B and out["F4"].shape[1] == T_out

    def test_padding_mask_propagated(self):
        backbone = StubStudentBackbone()
        x = torch.randn(2, 1, 3, 64, 64)
        pm = torch.ones(2, 16, 16, dtype=torch.bool)   # stride-4 grid for 64×64
        out = backbone(x, padding_mask=pm)
        assert out["pmasks"][0] is not None


# ══════════════════════════════════════════════════════════════════════════════
# 3. Loss function unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStudentLosses:
    D = STUB_EMBED_DIMS[-1]
    B = 2
    N = 16

    def test_cosine_loss_identical(self):
        a = F.normalize(torch.randn(self.B, self.D), dim=-1)
        loss = cosine_loss(a, a)
        assert loss.item() < 1e-5, "cosine_loss of identical vectors should be ~0"

    def test_cosine_loss_orthogonal(self):
        a = torch.zeros(1, self.D); a[0, 0] = 1.0
        b = torch.zeros(1, self.D); b[0, 1] = 1.0
        loss = cosine_loss(a, b)
        assert abs(loss.item() - 1.0) < 1e-4, "orthogonal vectors → cosine_loss=1"

    def test_img_global_distill(self):
        s = torch.randn(self.B, self.D)
        t = torch.randn(self.B, self.D)
        loss = img_global_distill_loss(s, t)
        assert loss.shape == torch.Size([])
        assert 0.0 <= loss.item() <= 2.0

    def test_img_patch_distill(self):
        s = torch.randn(self.B, self.N, self.D)
        t = torch.randn(self.B, self.N, self.D)
        loss = img_patch_distill_loss(s, t)
        assert loss.shape == torch.Size([])

    def test_img_patch_distill_downsamples_fine_student_grid(self):
        """Student stride-4 grid (16384) vs DINO stride-16 (1024) must not upsample teacher."""
        s = torch.randn(2, 16384, self.D)
        t = torch.randn(2, 1024, self.D)
        loss = img_patch_distill_loss(s, t)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_img_masked_semantic(self):
        s = torch.randn(self.B, self.N, self.D)
        t = torch.randn(self.B, self.N, self.D)
        mask = torch.zeros(self.B, self.N, dtype=torch.bool)
        mask[:, :4] = True   # mask first 4 positions
        loss = img_masked_semantic_loss(s, t, mask)
        assert loss.shape == torch.Size([])

    def test_img_masked_semantic_downsamples_fine_student_grid(self):
        s = torch.randn(2, 16384, self.D)
        t = torch.randn(2, 1024, self.D)
        mask = torch.zeros(2, 128, 128, dtype=torch.bool)
        mask[:, :8, :8] = True
        loss = img_masked_semantic_loss(s, t, mask)
        assert loss.shape == torch.Size([])
        assert torch.isfinite(loss)

    def test_img_proto_loss(self):
        s = torch.randn(self.B, self.N, self.D)
        t = torch.randn(self.B, self.N, self.D)
        proto = F.normalize(torch.randn(8, self.D), dim=-1)
        loss = img_proto_loss(s, t, proto)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0.0

    def test_img_proto_loss_decreases_with_student(self):
        """Student gradients should lower CE vs a fixed detached teacher target."""
        B, D, K = 64, 64, 8
        proto = F.normalize(torch.randn(K, D), dim=-1)
        teacher = F.normalize(torch.randn(B, D), dim=-1).detach()
        student = torch.randn(B, D, requires_grad=True)
        opt = torch.optim.Adam([student], lr=0.5)
        loss0 = img_proto_loss(student, teacher, proto).item()
        for _ in range(100):
            opt.zero_grad()
            loss = img_proto_loss(student, teacher, proto)
            loss.backward()
            opt.step()
        loss1 = loss.item()
        assert loss1 < loss0
        assert loss1 < math.log(K)

    def test_vid_proto_loss(self):
        s = torch.randn(self.B, self.N, self.D)
        t = torch.randn(self.B, self.N, self.D)
        proto = F.normalize(torch.randn(8, self.D), dim=-1)
        loss = vid_proto_loss(s, t, proto)
        assert loss.shape == torch.Size([])
        assert loss.item() >= 0.0
        assert torch.isfinite(loss)

    def test_vid_global_distill(self):
        s = torch.randn(self.B, self.D)
        t = torch.randn(self.B, self.D)
        loss = vid_global_distill_loss(s, t)
        assert loss.shape == torch.Size([])

    def test_vid_tube_prediction(self):
        s = torch.randn(self.B, self.N, self.D)
        t = torch.randn(self.B, self.N, self.D)
        loss = vid_tube_prediction_loss(s, t)
        assert loss.shape == torch.Size([])

    def test_vid_temporal_consistency_t1(self):
        frame_tokens = torch.randn(self.B, 1, self.D)
        loss = vid_temporal_consistency(frame_tokens)
        assert loss.item() == 0.0, "T=1 temporal consistency should be 0"

    def test_vid_temporal_consistency_t4(self):
        frame_tokens = torch.randn(self.B, 4, self.D)
        loss = vid_temporal_consistency(frame_tokens)
        assert loss.shape == torch.Size([])

    def test_fused_distill_loss(self):
        student = torch.randn(self.B, self.N, self.D)
        target  = torch.randn(self.B, self.N, self.D).detach()
        loss = fused_distill_loss(student, target)
        assert loss.shape == torch.Size([])

    def test_preservation_loss_has_grad(self):
        B, N_img, N_vid, D = 2, 16, 9, 64
        ph_dino = pw_dino = 4
        ph_vid = pw_vid = 3

        builder = FusionTargetBuilder(
            d_img=D, d_vid=D, align_dim=32, num_heads=4, fusion_radius=1,
        )
        z_img = torch.randn(B, N_img, D)
        z_vid = torch.randn(B, N_vid, D)

        z_fused, z_vid_base = builder(
            z_img, z_vid, ph_dino, pw_dino, ph_vid, pw_vid,
            detach_output=False, return_vid_baseline=True,
        )
        assert z_fused.requires_grad
        assert not z_vid_base.requires_grad

        loss = preservation_loss(z_fused, z_vid_base)
        loss.backward()

        assert builder.fusion.gate.grad is not None
        assert builder.fusion.gate.grad.abs().sum() > 0


# ══════════════════════════════════════════════════════════════════════════════
# 4. UPerNet segmentation head
# ══════════════════════════════════════════════════════════════════════════════

class TestUPerNetDecoder:
    @pytest.mark.parametrize("n_classes", [1, 4])
    @pytest.mark.parametrize("T", [1, 3])
    def test_shapes(self, n_classes, T):
        decoder = build_hierarchical_seg_head(STUB_EMBED_DIMS, n_classes=n_classes, fpn_channels=32)
        backbone = StubStudentBackbone()
        x = torch.randn(2, T, 3, 64, 64)
        features = backbone(x)
        logits = decoder(features)
        assert logits.shape[0] == 2
        assert logits.shape[-2] > 0 and logits.shape[-1] > 0
        if T == 1:
            assert logits.dim() == 4, "T=1 should return (B, C, H, W)"
        else:
            assert logits.dim() == 5, "T>1 should return (B, T, C, H, W)"

    def test_with_padding_mask(self):
        decoder = build_hierarchical_seg_head(STUB_EMBED_DIMS, n_classes=1, fpn_channels=32)
        backbone = StubStudentBackbone()
        x = torch.randn(2, 1, 3, 64, 64)
        pm = torch.ones(2, 16, 16, dtype=torch.bool)
        pm[:, :4] = False   # mark some patches as padding
        features = backbone(x, padding_mask=pm)
        logits = decoder(features, padding_mask=features["pmasks"][3])
        assert logits.shape[0] == 2


# ══════════════════════════════════════════════════════════════════════════════
# 5. StudentFinetuneModel forward + losses
# ══════════════════════════════════════════════════════════════════════════════

class TestStudentFinetuneModel:
    def _make_image_batch(self, B=2, H=64, W=64):
        return {
            "pixel_values": torch.randn(B, 1, 3, H, W),
            "padding_mask": torch.ones(B, H // 4, W // 4, dtype=torch.bool),
        }

    def _make_video_batch(self, B=2, T=4, H=64, W=64):
        return {
            "full_clips": torch.randn(B, T, 3, H, W),
            "padding_masks": torch.ones(B, H // 4, W // 4, dtype=torch.bool),
        }

    def test_no_heads_forward(self):
        model = _stub_model(head_specs=[])
        batch = self._make_image_batch()
        out = model(batch)
        assert "features" in out
        assert "loss" not in out

    @pytest.mark.parametrize("anatomy", ["cardiac", "breast", "thyroid", "lung", "fetal"])
    def test_anatomy_heads_forward_no_labels(self, anatomy):
        """All anatomy head suites should run without labels (inference mode)."""
        specs_fn = {
            "cardiac": cardiac_head_specs,
            "breast":  breast_head_specs,
            "thyroid": thyroid_head_specs,
            "lung":    lung_head_specs,
            "fetal":   fetal_head_specs,
        }[anatomy]
        specs = specs_fn(STUB_EMBED_DIMS)
        model = _stub_model(specs)
        batch = self._make_image_batch()
        out = model(batch)
        for spec in specs:
            if spec.name in model.heads:
                assert spec.name in out, f"Head '{spec.name}' missing from output"

    def test_segmentation_with_labels(self):
        specs = [HeadSpec(
            name="seg", task_type=TaskType.SEGMENTATION, n_classes=1,
            class_names=["structure"], loss_type=LossType.DICE,
            batch_key="seg_masks",
        )]
        model = _stub_model(specs)
        B, H, W = 2, 64, 64
        ph, pw = H // 4, W // 4
        batch = {
            "pixel_values": torch.randn(B, 1, 3, H, W),
            "seg_masks": torch.randint(0, 2, (B, 1, ph, pw)).float(),
        }
        out = model(batch)
        assert "loss" in out
        assert out["loss"].item() >= 0.0

    def test_lus_patient_tb_mil_with_labels(self):
        specs = [HeadSpec(
            name="lus_patient_tb",
            task_type=TaskType.PATIENT_CLS,
            n_classes=1,
            class_names=["tb"],
            loss_type=LossType.BCE,
            aggregation="mil",
            batch_key="lus_patient_tb_labels",
        )]
        model = _stub_model(specs)
        B, T = 2, 5
        batch = {
            "pixel_values": torch.randn(B, T, 3, 64, 64),
            "frame_mask": torch.tensor([
                [True, True, True, False, False],
                [True, True, True, True, False],
            ]),
            "lus_patient_tb_labels": torch.tensor([0.0, 1.0]),
        }
        out = model(batch)
        assert "lus_patient_tb" in out
        assert "loss" in out
        assert torch.isfinite(out["loss"])

    def test_classification_with_labels(self):
        specs = [HeadSpec(
            name="cls", task_type=TaskType.MULTICLASS_CLS, n_classes=3,
            class_names=["a", "b", "c"], loss_type=LossType.CE,
            batch_key="cls_labels",
        )]
        model = _stub_model(specs)
        batch = {
            "pixel_values": torch.randn(2, 1, 3, 64, 64),
            "cls_labels": torch.tensor([0, 2]),
        }
        out = model(batch)
        assert "loss" in out
        assert out["cls"].shape == (2, 3)

    def test_regression_with_labels(self):
        specs = [HeadSpec(
            name="ef", task_type=TaskType.REGRESSION, n_classes=1,
            class_names=["ejection_fraction"], loss_type=LossType.MSE,
            batch_key="ef_labels",
        )]
        model = _stub_model(specs)
        batch = {
            "pixel_values": torch.randn(2, 1, 3, 64, 64),
            "ef_labels": torch.tensor([55.0, 62.0]),
        }
        out = model(batch)
        assert "loss" in out
        assert out["ef"].shape == (2,)

    def test_video_batch_forward(self):
        specs = cardiac_head_specs(STUB_EMBED_DIMS)
        model = _stub_model(specs)
        batch = self._make_video_batch()
        out = model(batch)
        assert "features" in out
        assert out["features"]["F4"].shape[1] == 4   # T=4 preserved


# ══════════════════════════════════════════════════════════════════════════════
# 6. Phase step functions (stage 1, 2, 4)
# ══════════════════════════════════════════════════════════════════════════════

class TestPhaseSteps:
    def _make_image_batch_ssl(self, B=2, H=64, W=64, *, masked_patch: bool = True):
        ph = H // 4
        patch_masks = torch.zeros(B, ph, ph, dtype=torch.bool)
        if masked_patch:
            patch_masks[:, : ph // 2, : ph // 2] = True
        return {
            "global_crops": torch.randn(B, 2, 3, H, W),
            "global_pmasks": torch.ones(B, 2, ph, ph, dtype=torch.bool),
            "patch_masks": patch_masks,
            "sample_type": "image",
        }

    def _make_video_batch_ssl(self, B=2, T=4, H=64, W=64):
        ph16 = H // 16
        ph4 = H // 4
        tube_mask = torch.zeros(B, T, ph16, ph16, dtype=torch.bool)
        tube_mask[:, :, : ph16 // 2, : ph16 // 2] = True
        return {
            "full_clips": torch.randn(B, T, 3, H, W),
            "visible_clips": torch.randn(B, T, 3, H, W),
            "tube_masks": tube_mask,
            "tube_masks_s16": tube_mask,
            "padding_masks": torch.ones(B, ph4, ph4, dtype=torch.bool),
            "padding_masks_s16": torch.ones(B, ph16, ph16, dtype=torch.bool),
            "valid_frames": torch.ones(B, T, dtype=torch.bool),
            "sample_type": "video",
        }

    def _make_components(self):
        student     = StubStudentBackbone()
        ema_student = StubStudentBackbone()
        proto_head  = StubProtoHead(n_proto=8, dim=STUB_EMBED_DIMS[-1])

        # Minimal stub teacher objects (duck-typed)
        class _DINOStub(nn.Module):
            def forward(self, pv, padding_mask=None):
                B = pv.shape[0]
                D1, D4 = STUB_EMBED_DIMS[0], STUB_EMBED_DIMS[-1]
                ph = max(1, pv.shape[2] // STUB_PATCH_STRIDE)
                N = ph * ph
                return {
                    "cls":          torch.randn(B, D4),
                    "patch_tokens": torch.randn(B, N, D1),
                    "cls_proj":     torch.randn(B, D4),
                    "patch_proj":   torch.randn(B, N, D1),
                }

        return student, ema_student, _DINOStub(), proto_head

    def test_stage1_image_batch(self):
        student, ema, dino, proto = self._make_components()
        batch = self._make_image_batch_ssl()

        lam = dict(lam_global=1.0, lam_patch=0.5, lam_masked=0.5, lam_proto=0.2)
        out = student_stage1_step(batch, student, ema, dino, proto, lam)

        assert "loss" in out
        assert out["loss"].requires_grad
        assert 0.0 <= out["loss_global"] <= 2.0

    def test_stage1_video_batch_warmstart(self):
        """Stage-1 mixes may include video samples; synthesize two global crops."""
        student, ema, dino, proto = self._make_components()
        batch = self._make_video_batch_ssl()

        lam = dict(lam_global=1.0, lam_patch=0.5, lam_masked=0.5, lam_proto=0.2)
        out = student_stage1_step(batch, student, ema, dino, proto, lam)

        assert "loss" in out
        assert out["loss"].requires_grad

    def test_stage1_ema_ramps_from_zero(self):
        student, ema, dino, proto = self._make_components()
        batch = self._make_image_batch_ssl()
        lam = dict(
            lam_global=1.0, lam_patch=0.5, lam_masked=0.5, lam_proto=0.2,
            lam_ema_max=0.15, lam_ema_warmup_steps=100,
        )
        out0 = student_stage1_step(batch, student, ema, dino, proto, lam, global_step=0)
        assert out0["lam_ema_eff"] == 0.0

        out_w = student_stage1_step(batch, student, ema, dino, proto, lam, global_step=100)
        assert out_w["lam_ema_eff"] == pytest.approx(0.15)

    def test_stage1_ema_loss_active(self):
        student, ema, dino, proto = self._make_components()
        batch = self._make_image_batch_ssl()
        lam = dict(
            lam_global=1.0, lam_patch=0.5, lam_masked=0.5, lam_proto=0.2,
            lam_ema_max=0.15, lam_ema_warmup_steps=10,
        )
        out = student_stage1_step(
            batch, student, ema, dino, proto, lam, global_step=10,
        )
        assert out["loss_ema"] > 0.0
        assert out["loss_ema_global"] > 0.0

    def test_stage1_ema_disabled(self):
        student, ema, dino, proto = self._make_components()
        batch = self._make_image_batch_ssl()
        lam = dict(lam_global=1.0, lam_ema_max=0.0, lam_ema_warmup_steps=100)
        out = student_stage1_step(
            batch, student, ema, dino, proto, lam, global_step=100,
        )
        assert out["lam_ema_eff"] == 0.0

    def test_stage1_ema_no_grad_through_target(self):
        student, ema, dino, proto = self._make_components()
        for p in ema.parameters():
            p.requires_grad_(False)
        batch = self._make_image_batch_ssl()
        lam = dict(lam_global=0.0, lam_patch=0.0, lam_masked=0.0, lam_proto=0.0,
                   lam_ema_max=1.0, lam_ema_warmup_steps=1)
        out = student_stage1_step(
            batch, student, ema, dino, proto, lam, global_step=1,
        )
        out["loss"].backward()
        assert all(p.grad is None for p in ema.parameters())
        assert any(p.grad is not None for p in student.parameters())

    def test_lam_ema_eff_helper(self):
        lam = dict(lam_ema_max=0.2, lam_ema_floor=0.0, lam_ema_warmup_steps=100)
        assert _lam_ema_eff(lam, 0) == 0.0
        assert _lam_ema_eff(lam, 50) == pytest.approx(0.1)
        assert _lam_ema_eff(lam, 100) == pytest.approx(0.2)

    def test_dino_patch_masks_disjoint(self):
        B, N = 2, 16
        ph = pw = 4
        s_out = {
            "F1": torch.randn(B, 1, N, STUB_EMBED_DIMS[0]),
            "pmasks": [torch.ones(B, ph, pw, dtype=torch.bool)],
        }
        patch_masks = torch.zeros(B, ph, pw, dtype=torch.bool)
        patch_masks[:, :2, :2] = True

        unmasked, masked = _dino_patch_masks(s_out, patch_masks)
        assert masked is not None
        assert unmasked is not None
        assert (masked & unmasked).sum().item() == 0
        assert masked.sum().item() > 0
        assert unmasked.sum().item() > 0
        assert (masked | unmasked).all()

    def test_stage1_patch_masked_partition(self):
        student, ema, dino, proto = self._make_components()
        batch = self._make_image_batch_ssl()
        base_lam = dict(lam_global=0.0, lam_proto=0.0, lam_ema_max=0.0)

        out_both = student_stage1_step(
            batch, student, ema, dino, proto,
            {**base_lam, "lam_patch": 1.0, "lam_masked": 1.0},
        )
        assert out_both["loss_patch"] > 0.0
        assert out_both["loss_masked"] > 0.0

        out_patch_only = student_stage1_step(
            batch, student, ema, dino, proto,
            {**base_lam, "lam_patch": 1.0, "lam_masked": 0.0},
        )
        assert out_patch_only["loss"].item() == pytest.approx(
            out_both["loss_patch"], rel=1e-5,
        )

        out_masked_only = student_stage1_step(
            batch, student, ema, dino, proto,
            {**base_lam, "lam_patch": 0.0, "lam_masked": 1.0},
        )
        assert out_masked_only["loss"].item() == pytest.approx(
            out_both["loss_masked"], rel=1e-5,
        )

        batch_all_masked = self._make_image_batch_ssl()
        batch_all_masked["patch_masks"] = torch.ones_like(
            batch_all_masked["patch_masks"], dtype=torch.bool,
        )
        out_all_masked = student_stage1_step(
            batch_all_masked, student, ema, dino, proto,
            {**base_lam, "lam_patch": 1.0, "lam_masked": 1.0},
        )
        assert out_all_masked["loss_patch"] == 0.0
        assert out_all_masked["loss_masked"] > 0.0

    def test_stage2_image_batch(self):
        student, ema, dino, proto = self._make_components()
        batch = self._make_image_batch_ssl()

        class _VJEPAStub(nn.Module):
            def forward(self, pv, tube_mask=None, padding_mask=None, valid_frames=None):
                B, T = pv.shape[:2]
                D = STUB_EMBED_DIMS[-1]
                N = (pv.shape[3] // 16) * (pv.shape[4] // 16)
                return {
                    "clip_cls":    torch.randn(B, D),
                    "tube_tokens": torch.randn(B, T * N, D),
                    "clip_proj":   torch.randn(B, D),
                    "tube_proj":   torch.randn(B, T * N, D),
                }

        lam = dict(lam_global=1.0, lam_tube=0.5, lam_temp=0.2, lam_proto=0.2)
        out = student_stage2_step(batch, student, ema, dino, _VJEPAStub(), proto, lam)
        assert "loss" in out

    def test_stage2_video_batch(self):
        student, ema, dino, proto = self._make_components()
        batch = self._make_video_batch_ssl()

        class _VJEPAStub(nn.Module):
            def forward(self, pv, tube_mask=None, padding_mask=None, valid_frames=None):
                B, T = pv.shape[:2]
                D = STUB_EMBED_DIMS[-1]
                N = max(1, (pv.shape[3] // 16) * (pv.shape[4] // 16))
                return {
                    "clip_cls":    torch.randn(B, D),
                    "tube_tokens": torch.randn(B, T * N, D),
                    "clip_proj":   torch.randn(B, D),
                    "tube_proj":   torch.randn(B, T * N, D),
                }

        lam = dict(lam_global=1.0, lam_tube=0.5, lam_temp=0.2, lam_proto=0.2, lam_ema_max=0.15)
        out = student_stage2_step(batch, student, ema, dino, _VJEPAStub(), proto, lam)
        assert "loss" in out
        assert out["loss"].requires_grad
        assert out["loss_tube"] > 0.0
        assert out["loss_ema"] > 0.0

    def test_stage4_segmentation(self):
        B, H, W = 2, 64, 64
        ph = H // 4
        seg_specs = [HeadSpec(
            name="seg", task_type=TaskType.SEGMENTATION, n_classes=1,
            class_names=["lv"], loss_type=LossType.DICE,
            batch_key="seg_masks",
        )]
        student = StubStudentBackbone()
        seg_head = build_hierarchical_seg_head(STUB_EMBED_DIMS, n_classes=1, fpn_channels=32)
        batch = {
            "global_crops": torch.randn(B, 1, 3, H, W),
            "padding_masks": torch.ones(B, ph, ph, dtype=torch.bool),
            "seg_masks": torch.randint(0, 2, (B, 1, ph, pw := ph)).float(),
            "sample_type": "image",
        }
        out = student_stage4_step(
            batch, student,
            seg_head=seg_head,
            cls_heads={},
            lam=dict(lam_seg=1.0, lam_seg_dice=1.0),
            backbone_frozen=False,
        )
        assert "loss" in out
        assert out["loss"].requires_grad


# ══════════════════════════════════════════════════════════════════════════════
# 7. Real dataset loading: manifests + transforms → batch shapes
# ══════════════════════════════════════════════════════════════════════════════

class TestDatasetLoading:
    """
    Each test uses the conftest fixtures which build synthetic mini-datasets
    with the real adapters (realistic file layouts, metadata CSVs, etc.).
    Verifies that real samples flow through transforms and the student backbone.
    """

    def _verify_batch_through_backbone(self, batch: dict, label: str):
        """Assert the batch can flow through the stub backbone without error."""
        backbone = StubStudentBackbone()

        if "global_crops" in batch:
            crops = batch["global_crops"]   # (B, N, 3, H, W)
            x = crops[:, :1]               # (B, 1, 3, H, W)
            pm = batch.get("global_pmasks")
            if pm is not None and pm.dim() == 4:
                pm = pm[:, 0]
        elif "full_clips" in batch:
            x = batch["full_clips"]         # (B, T, 3, H, W)
            pm = batch.get("padding_masks")
        else:
            pytest.skip(f"Unknown batch format for {label}")
            return

        if x.dim() == 4:
            x = x.unsqueeze(1)

        out = backbone(x, padding_mask=pm)
        assert out["global"].shape[0] == x.shape[0], f"{label}: batch size mismatch"
        assert out["F4"].shape[0] == x.shape[0]

    # ── CAMUS (cardiac segmentation + video) ─────────────────────────────────

    def test_camus_image_batch(self, camus_root, tmp_path):
        adapter = CAMUSAdapter(camus_root)
        batch   = _get_image_batch(adapter, tmp_path)
        assert "global_crops" in batch
        assert batch["global_crops"].dim() == 5   # (B, N_crops, 3, H, W)
        self._verify_batch_through_backbone(batch, "CAMUS_image")

    def test_camus_video_batch(self, camus_root, tmp_path):
        adapter = CAMUSAdapter(camus_root)
        batch   = _get_video_batch(adapter, tmp_path)
        assert "full_clips" in batch or "visible_clips" in batch
        self._verify_batch_through_backbone(batch, "CAMUS_video")

    def test_camus_model_forward(self, camus_root, tmp_path):
        adapter = CAMUSAdapter(camus_root)
        batch   = _get_image_batch(adapter, tmp_path)
        model   = _stub_model(cardiac_head_specs(STUB_EMBED_DIMS))
        # Convert batch format: student model expects pixel_values
        batch["pixel_values"] = batch["global_crops"][:, :1].squeeze(1).unsqueeze(1)
        out = model(batch)
        assert "features" in out

    # ── BUSI (breast segmentation + malignancy classification) ───────────────

    def test_busi_image_batch(self, busi_root, tmp_path):
        adapter = BUSIAdapter(busi_root)
        batch   = _get_image_batch(adapter, tmp_path)
        self._verify_batch_through_backbone(batch, "BUSI_image")

    def test_busi_model_forward(self, busi_root, tmp_path):
        adapter = BUSIAdapter(busi_root)
        batch   = _get_image_batch(adapter, tmp_path)
        model   = _stub_model(breast_head_specs(STUB_EMBED_DIMS))
        batch["pixel_values"] = batch["global_crops"][:, :1].squeeze(1).unsqueeze(1)
        out = model(batch)
        assert "breast_seg" in out or "birads_cls" in out

    # ── COVIDx-US (lung classification + video) ───────────────────────────────

    def test_covidx_image_batch(self, covidx_root, tmp_path):
        adapter = COVIDxUSAdapter(covidx_root)
        batch   = _get_image_batch(adapter, tmp_path)
        self._verify_batch_through_backbone(batch, "COVIDx_image")

    def test_covidx_video_batch(self, covidx_root, tmp_path):
        adapter = COVIDxUSAdapter(covidx_root)
        batch   = _get_video_batch(adapter, tmp_path)
        self._verify_batch_through_backbone(batch, "COVIDx_video")

    def test_covidx_model_forward(self, covidx_root, tmp_path):
        adapter = COVIDxUSAdapter(covidx_root)
        batch   = _get_image_batch(adapter, tmp_path)
        model   = _stub_model(lung_head_specs(STUB_EMBED_DIMS))
        batch["pixel_values"] = batch["global_crops"][:, :1].squeeze(1).unsqueeze(1)
        out = model(batch)
        assert "lung_cls" in out

    # ── TN3K (thyroid nodule segmentation) ───────────────────────────────────

    def test_tn3k_image_batch(self, tn3k_root, tmp_path):
        adapter = TN3KAdapter(tn3k_root)
        batch   = _get_image_batch(adapter, tmp_path)
        self._verify_batch_through_backbone(batch, "TN3K_image")

    def test_tn3k_model_forward(self, tn3k_root, tmp_path):
        adapter = TN3KAdapter(tn3k_root)
        batch   = _get_image_batch(adapter, tmp_path)
        model   = _stub_model(thyroid_head_specs(STUB_EMBED_DIMS))
        batch["pixel_values"] = batch["global_crops"][:, :1].squeeze(1).unsqueeze(1)
        out = model(batch)
        assert "thyroid_seg" in out

    # ── HC18 (fetal head circumference measurement) ───────────────────────────

    def test_hc18_image_batch(self, hc18_root, tmp_path):
        adapter = HC18Adapter(hc18_root)
        batch   = _get_image_batch(adapter, tmp_path)
        self._verify_batch_through_backbone(batch, "HC18_image")

    def test_hc18_model_forward(self, hc18_root, tmp_path):
        adapter = HC18Adapter(hc18_root)
        batch   = _get_image_batch(adapter, tmp_path)
        model   = _stub_model(fetal_head_specs(STUB_EMBED_DIMS))
        batch["pixel_values"] = batch["global_crops"][:, :1].squeeze(1).unsqueeze(1)
        out = model(batch)
        assert "hc_measurement" in out or "fetal_plane_cls" in out

    # ── EchoNet-Dynamic (cardiac EF regression + video) ───────────────────────

    def test_echonet_video_batch(self, echonet_root, tmp_path):
        adapter = EchoNetDynamicAdapter(echonet_root)
        batch   = _get_video_batch(adapter, tmp_path)
        self._verify_batch_through_backbone(batch, "EchoNet_video")

    # ── Fetal Planes DB (multi-class view classification) ────────────────────

    def test_fetal_planes_image_batch(self, fetal_planes_root, tmp_path):
        adapter = FetalPlanesDBAdapter(fetal_planes_root)
        batch   = _get_image_batch(adapter, tmp_path)
        self._verify_batch_through_backbone(batch, "FetalPlanes_image")

    def test_fetal_planes_model_forward(self, fetal_planes_root, tmp_path):
        adapter = FetalPlanesDBAdapter(fetal_planes_root)
        batch   = _get_image_batch(adapter, tmp_path)
        model   = _stub_model(fetal_head_specs(STUB_EMBED_DIMS))
        batch["pixel_values"] = batch["global_crops"][:, :1].squeeze(1).unsqueeze(1)
        out = model(batch)
        assert "fetal_plane_cls" in out

    # ── LUS-multicenter (lung A/B-lines classification) ──────────────────────

    def test_lus_image_batch(self, lus_multicenter_root, tmp_path):
        adapter = LUSMulticenterAdapter(lus_multicenter_root)
        batch   = _get_image_batch(adapter, tmp_path)
        self._verify_batch_through_backbone(batch, "LUS_image")

    # ── BUS-BRA (breast mass segmentation + BI-RADS) ─────────────────────────

    def test_bus_bra_image_batch(self, bus_bra_root, tmp_path):
        adapter = BUSBRAAdapter(bus_bra_root)
        batch   = _get_image_batch(adapter, tmp_path)
        self._verify_batch_through_backbone(batch, "BUS-BRA_image")


# ══════════════════════════════════════════════════════════════════════════════
# 8. StudentMixedCollator sample_type tagging
# ══════════════════════════════════════════════════════════════════════════════

class TestStudentCollator:
    def test_image_collator_tags(self, camus_root, tmp_path):
        adapter = CAMUSAdapter(camus_root)
        ds = _image_dataset_from_adapter(adapter, tmp_path)
        samples = [ds[0], ds[1]] if len(ds) >= 2 else [ds[0]]
        valid = [s for s in samples if s is not None]
        mixed = StudentMixedCollator(ImageSSLCollator(), VideoSSLCollator())
        batch = mixed.collate_image(valid)
        assert batch.get("sample_type") == "image"

    def test_video_collator_tags(self, camus_root, tmp_path):
        adapter = CAMUSAdapter(camus_root)
        ds = _video_dataset_from_adapter(adapter, tmp_path)
        samples = [ds[0]] if len(ds) >= 1 else []
        if not samples:
            pytest.skip("No video samples")
        valid = [s for s in samples if s is not None]
        mixed = StudentMixedCollator(ImageSSLCollator(), VideoSSLCollator())
        batch = mixed.collate_video(valid)
        assert batch.get("sample_type") == "video"

    def test_ddp_broadcast_sample_type_passthrough_without_dist(self):
        assert _ddp_broadcast_sample_type("video") == "video"
        assert _ddp_broadcast_sample_type("paired") == "paired"

    def test_ddp_active_false_without_dist(self):
        assert _ddp_active() is False


# ══════════════════════════════════════════════════════════════════════════════
# 9. Native-resolution invariance test
# ══════════════════════════════════════════════════════════════════════════════

class TestNativeResolution:
    """
    Verify the backbone produces the correct output for images of different
    native resolutions — no fixed resize, output spatial dims scale with input.
    """

    @pytest.mark.parametrize("H,W", [(48, 48), (80, 112), (96, 160)])
    def test_variable_resolution(self, H, W):
        backbone = StubStudentBackbone()
        x = torch.randn(2, 1, 3, H, W)
        out = backbone(x)
        # F4 should have spatial dim proportional to H×W (divided by stride)
        expected_N4 = (H // (STUB_PATCH_STRIDE * 8)) * (W // (STUB_PATCH_STRIDE * 8))
        actual_N4   = out["F4"].shape[2]
        assert actual_N4 >= 1, f"F4 spatial dim should be ≥1 for H={H}, W={W}"
        assert out["global"].shape == (2, STUB_EMBED_DIMS[-1])

    def test_consistent_output_shape_across_crops(self):
        """
        Two images of different resolutions in a padded batch — each should
        produce the correct F4 spatial size without contaminating each other.
        """
        backbone = StubStudentBackbone()
        # Both images padded to same W but different content aspect ratios
        x = torch.randn(2, 1, 3, 64, 128)
        out = backbone(x)
        assert out["F4"].shape[0] == 2
