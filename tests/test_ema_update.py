"""Unit tests for ema_update (name-dict matching, DDP unwrap, buffers)."""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from models.branches.shared import _ema_copy_buffers, ema_update


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=True)
        self.register_buffer("scale", torch.ones(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x) * self.scale


class _DDPWrapper(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.module = inner


def test_ema_update_changes_teacher_weights():
    student = _TinyNet()
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)

    w_before = teacher.fc.weight.clone()
    student.fc.weight.data.add_(1.0)
    ema_update(student, teacher, momentum=0.9)

    assert not torch.allclose(w_before, teacher.fc.weight)


def test_ema_update_exact_equation():
    student = _TinyNet()
    teacher = copy.deepcopy(student)
    momentum = 0.9

    student.fc.weight.data.add_(0.5)
    student.fc.bias.data.add_(0.25)

    expected = {}
    for name, t_p in teacher.named_parameters():
        s_p = dict(student.named_parameters())[name]
        expected[name] = momentum * t_p.data.clone() + (1.0 - momentum) * s_p.data

    ema_update(student, teacher, momentum=momentum)

    for name, t_p in teacher.named_parameters():
        assert torch.allclose(t_p.data, expected[name], atol=1e-6), name


def test_ema_update_strict_param_mismatch():
    student = _TinyNet()
    teacher = nn.Linear(4, 4)
    with pytest.raises(KeyError, match="missing parameters"):
        ema_update(student, teacher, momentum=0.9)


def test_ema_update_no_grad_on_teacher():
    student = _TinyNet()
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)

    ema_update(student, teacher, momentum=0.9)
    assert all(not p.requires_grad for p in teacher.parameters())
    assert all(p.grad is None for p in teacher.parameters())


def test_ema_update_ddp_unwrap():
    student = _DDPWrapper(_TinyNet())
    teacher = copy.deepcopy(student.module)
    for p in teacher.parameters():
        p.requires_grad_(False)

    w_before = teacher.fc.weight.clone()
    student.module.fc.weight.data.add_(1.0)
    ema_update(student, teacher, momentum=0.5)

    assert not torch.allclose(w_before, teacher.fc.weight)


def test_ema_copy_buffers():
    student = _TinyNet()
    teacher = copy.deepcopy(student)
    student.scale.fill_(3.0)
    _ema_copy_buffers(student, teacher)
    assert torch.allclose(teacher.scale, torch.full((4,), 3.0))


def test_ema_copy_buffers_mismatch_raises():
    student = _TinyNet()
    teacher = _TinyNet()
    teacher.register_buffer("extra", torch.zeros(1))
    with pytest.raises(KeyError, match="buffers"):
        _ema_copy_buffers(student, teacher)
