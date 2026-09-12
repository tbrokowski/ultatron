"""CLI surface of train.student_pretrain (no GPU)."""
from __future__ import annotations

import sys

import pytest

torch = pytest.importorskip("torch")

from train.student_pretrain import _parse_args, _stage_for_step
from tests.dataset_adapters.student_training_smoke import (
    _resolve_resume_stage_fracs,
    _stage_for_step as shim_stage,
)
from train.bench import apply_bench_overrides


def test_shim_exports_resume_helpers():
    assert callable(shim_stage)
    assert callable(_resolve_resume_stage_fracs)
    assert shim_stage is _stage_for_step
    assert _stage_for_step(0, 100_000, [0.25, 0.20, 0.20, 0.35]) == 1
    assert _stage_for_step(25_000, 100_000, [0.25, 0.20, 0.20, 0.35]) == 2


def test_parse_bench_flags(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        [
            "student_pretrain",
            "--bench-stage", "2",
            "--max-steps", "80",
            "--no-ckpt",
            "--per-step-timing",
            "--bench-window",
            "--gbs-img", "2048",
            "--gbs-vid", "32",
            "--num-workers", "8",
            "--video-manifest", "/tmp/enc_videos.jsonl",
        ],
    )
    args = _parse_args()
    assert args.bench_stage == 2
    assert args.max_steps == 80
    assert args.no_ckpt
    assert args.per_step_timing
    assert args.bench_window
    assert args.video_manifest == "/tmp/enc_videos.jsonl"
    cfg = {"training": {}, "loaders": {}, "student_data": {}, "pretrain": {}}
    apply_bench_overrides(cfg, args)
    assert cfg["training"]["total_steps"] == 80
    assert cfg["training"]["stage_fracs"] == [0.0, 1.0, 0.0, 0.0]
    assert cfg["training"]["ckpt_every"] == 0
    assert cfg["training"]["global_batch_image"] == 2048
    assert cfg["loaders"]["num_workers"] == 8
    assert cfg["manifest"]["video_path"] == "/tmp/enc_videos.jsonl"
