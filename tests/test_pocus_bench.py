"""Unit tests for WP5 / POCUS bench helpers (no GPU, no cluster)."""
from __future__ import annotations

import math

from train.bench import (
    accum_steps,
    amdahl_fit,
    efficiency,
    half_window_rel_diff,
    pin_bench_stage,
    resolve_grad_accum,
    select_measurement_window,
    speedup,
    throughput,
)


def test_accum_matches_spec_table():
    # GBS_img=2048, MBS=32 → accum at 4/8/16/32 GPUs = 16/8/4/2
    assert accum_steps(2048, 32, 4) == 16
    assert accum_steps(2048, 32, 8) == 8
    assert accum_steps(2048, 32, 16) == 4
    assert accum_steps(2048, 32, 32) == 2
    # GBS_vid=32, MBS=1 → 8/4/2/1
    assert accum_steps(32, 1, 4) == 8
    assert accum_steps(32, 1, 8) == 4
    assert accum_steps(32, 1, 16) == 2
    assert accum_steps(32, 1, 32) == 1


def test_accum_rejects_non_integer():
    import pytest
    with pytest.raises(ValueError):
        accum_steps(2048, 32, 3)


def test_resolve_grad_accum_by_type():
    cfg = {
        "training": {"global_batch_image": 2048, "global_batch_video": 32},
        "loaders": {"image_batch_size": 32, "video_batch_size": 1},
    }
    assert resolve_grad_accum(cfg, 4, "image") == 16
    assert resolve_grad_accum(cfg, 4, "video") == 8
    assert resolve_grad_accum(cfg, 32, "image") == 2
    assert resolve_grad_accum(cfg, 32, "video") == 1


def test_pin_bench_stage():
    cfg = {"training": {"stage_fracs": [0.25, 0.2, 0.2, 0.35]}}
    pin_bench_stage(cfg, 2)
    assert cfg["training"]["stage_fracs"] == [0.0, 1.0, 0.0, 0.0]


def test_measurement_window_too_short():
    wr = select_measurement_window([1.0] * 40)
    assert not wr.stable
    assert "shorter" in wr.reason or "warmup" in wr.reason


def test_measurement_window_stable():
    # 30 warmup steps of 1s (=30s) is not enough wall time; then 5 min of 1s
    # plus 15 min measurement.
    t = [1.0] * 1200  # 20 minutes of 1s steps
    wr = select_measurement_window(t)
    assert wr.warmup_discarded >= 30
    assert wr.n_steps >= 50
    assert wr.stable
    assert wr.half_rel_diff <= 0.03
    assert math.isclose(wr.t_mean, 1.0, rel_tol=1e-6)


def test_measurement_window_unstable_then_extend():
    # Warmup 300s + 30 steps, then first half 1s, second half 2s → 50% diff.
    warmup = [1.0] * 330
    first = [1.0] * 450
    second = [2.0] * 450
    t = warmup + first + second
    wr = select_measurement_window(t)
    assert wr.half_rel_diff > 0.03 or wr.extended or not wr.stable


def test_speedup_and_efficiency_ideal():
    rates = {4: 10.0, 8: 20.0, 16: 40.0, 32: 80.0}
    s = speedup(rates)
    e = efficiency(rates)
    assert s[4] == 1.0
    assert math.isclose(s[32], 8.0)
    for v in e.values():
        assert math.isclose(v, 1.0, rel_tol=1e-9)


def test_throughput():
    assert math.isclose(throughput(2048, 2.0), 1024.0)


def test_amdahl_perfect_parallel():
    n = [4, 8, 16, 32]
    s = [1, 2, 4, 8]
    f = amdahl_fit(n, s, 4)
    assert f is not None
    assert f > 0.9


def test_half_window():
    assert half_window_rel_diff([1, 1, 1, 1]) == 0.0
    d = half_window_rel_diff([1, 1, 3, 3])
    assert math.isclose(d, 1.0)  # |2-1|/2 wait: mean=2, |1-3|/2 = 1.0 yes
