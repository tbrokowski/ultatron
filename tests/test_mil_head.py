"""Tests for Gated Attention MIL and multi-finding binary heads."""
from __future__ import annotations

import torch

from models.heads.mil_head import GatedAttentionMILPool, PatientMILClsHead
from models.heads.multifinding_head import MultiFindingBinaryHead, DEFAULT_LUS_FINDINGS


class TestGatedAttentionMILPool:
    def test_single_bag_shape(self):
        pool = GatedAttentionMILPool(embed_dim=64, hidden_dim=32)
        H = torch.randn(5, 64)
        z = pool(H)
        assert z.shape == (64,)

    def test_single_bag_attention_sums_to_one(self):
        pool = GatedAttentionMILPool(embed_dim=64, hidden_dim=32)
        H = torch.randn(4, 64)
        w = pool.attention_weights(H)
        assert w.shape == (4,)
        assert torch.allclose(w.sum(), torch.tensor(1.0), atol=1e-5)

    def test_batched_with_mask(self):
        pool = GatedAttentionMILPool(embed_dim=64, hidden_dim=32)
        H = torch.randn(2, 5, 64)
        mask = torch.tensor([[True, True, True, False, False],
                             [True, True, False, False, False]])
        z = pool(H, mask=mask)
        assert z.shape == (2, 64)

    def test_gradients_flow(self):
        pool = GatedAttentionMILPool(embed_dim=32, hidden_dim=16)
        H = torch.randn(3, 32, requires_grad=True)
        z = pool(H)
        z.sum().backward()
        assert H.grad is not None


class TestPatientMILClsHead:
    def test_single_patient_output(self):
        head = PatientMILClsHead(embed_dim=64, hidden_dim=32, n_classes=1)
        H = torch.randn(6, 64)
        logit = head(H)
        assert logit.shape == (1,)

    def test_batched_output_and_loss(self):
        head = PatientMILClsHead(embed_dim=64, hidden_dim=32, n_classes=1)
        H = torch.randn(2, 4, 64)
        mask = torch.ones(2, 4, dtype=torch.bool)
        logits = head(H, mask=mask)
        assert logits.shape == (2, 1)
        targets = torch.tensor([0.0, 1.0])
        loss = head.loss(logits, targets)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


class TestMultiFindingBinaryHead:
    def test_forward_shape_mlp(self):
        head = MultiFindingBinaryHead(embed_dim=128, head_type="mlp")
        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, len(DEFAULT_LUS_FINDINGS))

    def test_forward_shape_linear(self):
        head = MultiFindingBinaryHead(embed_dim=128, head_type="linear")
        x = torch.randn(2, 128)
        out = head(x)
        assert out.shape == (2, len(DEFAULT_LUS_FINDINGS))

    def test_independent_heads(self):
        head = MultiFindingBinaryHead(embed_dim=64, head_type="mlp")
        x = torch.randn(1, 64)
        out = head(x)
        assert len(head.heads) == len(DEFAULT_LUS_FINDINGS)
        assert out.shape[-1] == len(DEFAULT_LUS_FINDINGS)
