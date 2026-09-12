"""
train/bench.py  ·  Strong-scaling benchmark helpers for WP5 / POCUS
===================================================================

Shared by ``train.student_pretrain`` and ``scripts/pocus/analyse.py``.

Protocol (spec §4.1)
--------------------
* Discard until both 30 optimizer steps **and** 5 minutes have passed.
* Then measure until both ≥50 steps **and** ≥15 minutes, hard cap 60 minutes.
* Half-window mean step times must agree within 3 %.  If not, extend the
  window by 50 % once, then mark the run unstable.
"""
from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SEED_DEFAULT = 1234

WARMUP_STEPS = 30
WARMUP_SECONDS = 5 * 60
MEASURE_STEPS = 50
MEASURE_SECONDS = 15 * 60
HARD_CAP_SECONDS = 60 * 60
HALF_WINDOW_TOL = 0.03
GBS_IMG_DEFAULT = 2048
GBS_VID_DEFAULT = 32


def set_global_seed(seed: int = SEED_DEFAULT) -> None:
    """Deterministic RNG for strong-scaling (same seed at every scale)."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def world_size() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except ImportError:
        pass
    for key in ("WORLD_SIZE", "SLURM_NTASKS"):
        if key in os.environ:
            # torchrun WORLD_SIZE is GPU count; SLURM_NTASKS is 1 per node here.
            return int(os.environ[key])
    nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    nproc = int(os.environ.get("SLURM_GPUS_ON_NODE", os.environ.get("GPUS_PER_NODE", "1")))
    if "LOCAL_WORLD_SIZE" in os.environ:
        return nnodes * int(os.environ["LOCAL_WORLD_SIZE"])
    return max(1, nnodes * nproc)


def gpu_count_from_env() -> int:
    """Best-effort GPU count for the job (used when dist is not initialised)."""
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    nnodes = int(os.environ.get("SLURM_NNODES", "1"))
    nproc = int(os.environ.get(
        "SLURM_GPUS_PER_NODE",
        os.environ.get("GPUS_PER_NODE", os.environ.get("LOCAL_WORLD_SIZE", "1")),
    ))
    return max(1, nnodes * nproc)


def accum_steps(gbs: int, mbs: int, n_gpus: int) -> int:
    """
    Gradient accumulation so that GBS = MBS × n_gpus × accum.

    Raises ValueError if the quotient is not an integer ≥ 1.
    """
    if mbs <= 0 or n_gpus <= 0:
        raise ValueError(f"mbs and n_gpus must be positive, got mbs={mbs} n_gpus={n_gpus}")
    denom = mbs * n_gpus
    if gbs % denom != 0:
        raise ValueError(
            f"GBS={gbs} is not divisible by MBS×n_gpus={mbs}×{n_gpus}={denom}"
        )
    acc = gbs // denom
    if acc < 1:
        raise ValueError(f"accumulation would be {acc} (GBS={gbs}, MBS={mbs}, n={n_gpus})")
    return acc


def resolve_grad_accum(
    cfg: dict,
    n_gpus: int,
    sample_type: str = "image",
) -> int:
    """Per-step-type accumulation from global-batch settings (spec §4.2)."""
    loaders = cfg.get("loaders") or {}
    tcfg = cfg.get("training") or {}
    if sample_type == "video":
        gbs = tcfg.get("global_batch_video")
        mbs = int(loaders.get("video_batch_size", 1))
        if gbs is not None:
            return accum_steps(int(gbs), mbs, n_gpus)
    else:
        gbs = tcfg.get("global_batch_image")
        mbs = int(loaders.get("image_batch_size", 1))
        if gbs is not None:
            return accum_steps(int(gbs), mbs, n_gpus)
    return max(1, int(tcfg.get("grad_accum_steps", 1)))


def pin_bench_stage(cfg: dict, stage: int) -> dict:
    """Force the entire run onto curriculum stage ``stage`` (1–4)."""
    if stage not in (1, 2, 3, 4):
        raise ValueError(f"bench stage must be 1–4, got {stage}")
    fracs = [0.0, 0.0, 0.0, 0.0]
    fracs[stage - 1] = 1.0
    cfg.setdefault("training", {})["stage_fracs"] = fracs
    return cfg


def apply_bench_overrides(cfg: dict, args: Any) -> dict:
    """Apply CLI benchmark flags to the loaded YAML config (in-place)."""
    tcfg = cfg.setdefault("training", {})
    loaders = cfg.setdefault("loaders", {})
    student = cfg.setdefault("student_data", {})
    pretrain = cfg.setdefault("pretrain", {})

    if getattr(args, "max_steps", None):
        tcfg["total_steps"] = int(args.max_steps)
    if getattr(args, "bench_stage", None):
        pin_bench_stage(cfg, int(args.bench_stage))
    if getattr(args, "no_ckpt", False):
        tcfg["ckpt_every"] = 0
        pretrain["save_stage_end_ckpts"] = False
    if getattr(args, "image_mbs", None):
        loaders["image_batch_size"] = int(args.image_mbs)
        student["image_batch_size"] = int(args.image_mbs)
    if getattr(args, "video_mbs", None):
        loaders["video_batch_size"] = int(args.video_mbs)
        student["video_batch_size"] = int(args.video_mbs)
    if getattr(args, "gbs_img", None):
        tcfg["global_batch_image"] = int(args.gbs_img)
    if getattr(args, "gbs_vid", None):
        tcfg["global_batch_video"] = int(args.gbs_vid)
    if getattr(args, "num_workers", None):
        loaders["num_workers"] = int(args.num_workers)
        student["num_workers"] = int(args.num_workers)
    if getattr(args, "prefetch_factor", None):
        loaders["prefetch_factor"] = int(args.prefetch_factor)
        student["prefetch_factor"] = int(args.prefetch_factor)
    if getattr(args, "seed", None) is not None:
        tcfg["seed"] = int(args.seed)
    tcfg.setdefault("global_batch_image", GBS_IMG_DEFAULT)
    tcfg.setdefault("global_batch_video", GBS_VID_DEFAULT)
    tcfg.setdefault("seed", SEED_DEFAULT)
    if getattr(args, "log_every", None):
        tcfg["log_every"] = int(args.log_every)
        tcfg["metrics_every"] = int(args.log_every)
    # Bench runs want every-step JSONL.
    if getattr(args, "per_step_timing", False) or getattr(args, "bench_window", False):
        tcfg["log_every"] = min(int(tcfg.get("log_every", 1)), 1)
        tcfg["metrics_every"] = 1
    return cfg


@dataclass
class WindowResult:
    start_idx: int
    end_idx: int
    n_steps: int
    t_mean: float
    t_std: float
    cv: float
    half_rel_diff: float
    stable: bool
    extended: bool
    warmup_discarded: int
    wall_s: float
    reason: str = ""


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var)


def half_window_rel_diff(times: Sequence[float]) -> float:
    if len(times) < 2:
        return float("nan")
    mid = len(times) // 2
    a, b = times[:mid], times[mid:]
    if not a or not b:
        return float("nan")
    m = _mean(times)
    if m == 0:
        return float("nan")
    return abs(_mean(a) - _mean(b)) / m


def select_measurement_window(
    t_steps: Sequence[float],
    timestamps: Optional[Sequence[float]] = None,
    *,
    warmup_steps: int = WARMUP_STEPS,
    warmup_seconds: float = WARMUP_SECONDS,
    measure_steps: int = MEASURE_STEPS,
    measure_seconds: float = MEASURE_SECONDS,
    hard_cap_seconds: float = HARD_CAP_SECONDS,
    tol: float = HALF_WINDOW_TOL,
) -> WindowResult:
    """
    Select the timing window from a sequence of per-step times.

    ``timestamps[i]`` is wall-clock seconds from the start of the run to the
    *end* of step i.  If omitted, a cumulative sum of ``t_steps`` is used.
    """
    n = len(t_steps)
    if n == 0:
        return WindowResult(
            0, 0, 0, float("nan"), float("nan"), float("nan"), float("nan"),
            False, False, 0, 0.0, "empty",
        )
    if timestamps is None:
        ts = []
        acc = 0.0
        for t in t_steps:
            acc += float(t)
            ts.append(acc)
    else:
        ts = [float(x) for x in timestamps]
        if len(ts) != n:
            raise ValueError("timestamps length must match t_steps")

    # Warm-up: first index that has both 30 steps and 5 minutes behind it.
    warmup_end = 0
    for i in range(n):
        steps_done = i + 1
        if steps_done >= warmup_steps and ts[i] >= warmup_seconds:
            warmup_end = i + 1
            break
    else:
        return WindowResult(
            0, n, n, _mean(t_steps), _std(t_steps),
            (_std(t_steps) / _mean(t_steps)) if _mean(t_steps) else float("nan"),
            half_window_rel_diff(t_steps),
            False, False, 0, ts[-1],
            "run ended before warmup (30 steps and 5 min)",
        )

    def _window_ok(start: int, end: int) -> bool:
        times = t_steps[start:end]
        wall = ts[end - 1] - (ts[start - 1] if start > 0 else 0.0)
        return len(times) >= measure_steps and wall >= measure_seconds

    start = warmup_end
    end = start
    cap_t = (ts[start - 1] if start > 0 else 0.0) + hard_cap_seconds
    while end < n and ts[end] <= cap_t:
        end += 1
        if _window_ok(start, end):
            break
    if end <= start:
        end = min(n, start + 1)

    times = list(t_steps[start:end])
    wall = ts[end - 1] - (ts[start - 1] if start > 0 else 0.0)
    rel = half_window_rel_diff(times)
    stable = math.isfinite(rel) and rel <= tol and _window_ok(start, end)
    extended = False

    if not stable and math.isfinite(rel) and rel > tol:
        extra = max(1, int(0.5 * (end - start)))
        new_end = min(n, end + extra)
        # also respect remaining cap
        while new_end > end and ts[new_end - 1] > cap_t:
            new_end -= 1
        if new_end > end:
            end = new_end
            times = list(t_steps[start:end])
            wall = ts[end - 1] - (ts[start - 1] if start > 0 else 0.0)
            rel = half_window_rel_diff(times)
            stable = math.isfinite(rel) and rel <= tol and _window_ok(start, end)
            extended = True

    mean = _mean(times)
    std = _std(times)
    cv = (std / mean) if mean else float("nan")
    if not _window_ok(start, end):
        reason = "window shorter than 50 steps / 15 min"
        stable = False
    elif not (math.isfinite(rel) and rel <= tol):
        reason = "unstable after +50% extension" if extended else "half-window difference > 3%"
        stable = False
    else:
        reason = "ok"
        stable = True
    return WindowResult(
        start_idx=start,
        end_idx=end,
        n_steps=len(times),
        t_mean=mean,
        t_std=std,
        cv=cv,
        half_rel_diff=rel,
        stable=stable,
        extended=extended,
        warmup_discarded=start,
        wall_s=wall,
        reason=reason,
    )


class BenchWindowController:
    """Runtime counterpart of :func:`select_measurement_window` for live jobs."""

    def __init__(
        self,
        enabled: bool = True,
        *,
        warmup_steps: int = WARMUP_STEPS,
        warmup_seconds: float = WARMUP_SECONDS,
        measure_steps: int = MEASURE_STEPS,
        measure_seconds: float = MEASURE_SECONDS,
        hard_cap_seconds: float = HARD_CAP_SECONDS,
        tol: float = HALF_WINDOW_TOL,
    ):
        self.enabled = enabled
        self.warmup_steps = warmup_steps
        self.warmup_seconds = warmup_seconds
        self.measure_steps = measure_steps
        self.measure_seconds = measure_seconds
        self.hard_cap_seconds = hard_cap_seconds
        self.tol = tol
        self.t0 = time.time()
        self.t_steps: List[float] = []
        self._extended = False
        self.stopped = False
        self.result: Optional[WindowResult] = None

    def observe(self, t_step: float) -> bool:
        """
        Record one optimizer-step duration.

        Returns True if the run should stop (window complete or cap hit).
        """
        if not self.enabled:
            return False
        self.t_steps.append(float(t_step))
        elapsed = time.time() - self.t0
        n = len(self.t_steps)
        if n < self.warmup_steps or elapsed < self.warmup_seconds:
            if elapsed >= self.hard_cap_seconds:
                self.stopped = True
                self.result = select_measurement_window(self.t_steps)
                return True
            return False
        measure_elapsed = elapsed - self.warmup_seconds
        enough = (
            (n - self.warmup_steps) >= self.measure_steps
            and measure_elapsed >= self.measure_seconds
        )
        if elapsed >= self.hard_cap_seconds or enough:
            wr = select_measurement_window(self.t_steps)
            if (not wr.stable) and (not self._extended) and elapsed < self.hard_cap_seconds:
                self._extended = True
                return False
            self.stopped = True
            self.result = wr
            return True
        return False


@dataclass
class StepTiming:
    t_step: float = 0.0
    t_data_wait: float = 0.0
    t_fwd_student: float = 0.0
    t_fwd_teachers: float = 0.0
    t_bwd: float = 0.0
    t_opt: float = 0.0
    t_allreduce: float = 0.0
    n_images: int = 0
    n_clips: int = 0
    n_frames: int = 0
    mem_peak_GB: float = float("nan")
    loss: float = float("nan")
    nonfinite: bool = False
    stage: int = 0
    sample_type: str = ""
    step: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "step": self.step,
            "stage": self.stage,
            "type": self.sample_type,
            "t_step": self.t_step,
            "t_data_wait": self.t_data_wait,
            "t_fwd_student": self.t_fwd_student,
            "t_fwd_teachers": self.t_fwd_teachers,
            "t_bwd": self.t_bwd,
            "t_opt": self.t_opt,
            "t_allreduce": self.t_allreduce,
            "n_images": self.n_images,
            "n_clips": self.n_clips,
            "n_frames": self.n_frames,
            "mem_peak_GB": self.mem_peak_GB,
            "loss": self.loss,
            "nonfinite": self.nonfinite,
        }
        d.update(self.extra)
        return d

    def unaccounted(self) -> float:
        accounted = (
            self.t_data_wait
            + self.t_fwd_student
            + self.t_fwd_teachers
            + self.t_bwd
            + self.t_opt
            + self.t_allreduce
        )
        return self.t_step - accounted


class CudaStepTimer:
    """CUDA-event timer for fwd/bwd/opt slices. Falls back to wall clock."""

    def __init__(self, enabled: bool = True, device: Optional[str] = None):
        self.enabled = enabled
        self._cuda = False
        self._events: Dict[str, Tuple[Any, Any]] = {}
        self._wall: Dict[str, float] = {}
        if not enabled:
            return
        try:
            import torch
            self._cuda = bool(
                device and str(device).startswith("cuda") and torch.cuda.is_available()
            )
        except ImportError:
            self._cuda = False

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        if self._cuda:
            import torch
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            self._events[name] = (s, e)
        else:
            self._wall[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if not self.enabled:
            return 0.0
        if self._cuda and name in self._events:
            import torch
            s, e = self._events.pop(name)
            e.record()
            e.synchronize()
            return s.elapsed_time(e) / 1000.0
        t0 = self._wall.pop(name, None)
        if t0 is None:
            return 0.0
        return time.perf_counter() - t0


def count_batch_items(batch: dict) -> Tuple[int, int, int]:
    """Return (n_images, n_clips, n_frames) for a collated student batch."""
    st = batch.get("sample_type", "image")
    if st == "video":
        clips = batch.get("full_clips")
        if clips is None:
            clips = batch.get("visible_clips")
        if clips is not None and hasattr(clips, "shape"):
            b, t = int(clips.shape[0]), int(clips.shape[1])
            return 0, b, b * t
        return 0, int(batch.get("n_clips", 0) or 0), int(batch.get("n_frames", 0) or 0)
    crops = batch.get("global_crops")
    if crops is not None and hasattr(crops, "shape"):
        # (B, n_crops, C, H, W) or (B, C, H, W)
        b = int(crops.shape[0])
        return b, 0, b
    frame = batch.get("frame")
    if frame is not None and hasattr(frame, "shape"):
        return int(frame.shape[0]), 0, int(frame.shape[0])
    return 0, 0, 0


def peak_mem_gb(device: str) -> float:
    if not str(device).startswith("cuda"):
        return float("nan")
    try:
        import torch
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        return float("nan")


def throughput(gbs: float, t_mean: float) -> float:
    if not t_mean or not math.isfinite(t_mean) or t_mean <= 0:
        return float("nan")
    return gbs / t_mean


def speedup(rates: Dict[int, float], n0: Optional[int] = None) -> Dict[int, float]:
    if not rates:
        return {}
    n0 = n0 if n0 is not None else min(rates)
    r0 = rates[n0]
    return {n: (rates[n] / r0 if r0 else float("nan")) for n in sorted(rates)}


def efficiency(rates: Dict[int, float], n0: Optional[int] = None) -> Dict[int, float]:
    if not rates:
        return {}
    n0 = n0 if n0 is not None else min(rates)
    r0 = rates[n0]
    out = {}
    for n in sorted(rates):
        if not r0 or not n0:
            out[n] = float("nan")
        else:
            out[n] = (rates[n] / n) / (r0 / n0)
    return out


def amdahl_fit(n_list: Sequence[int], s_list: Sequence[float], n0: int) -> Optional[float]:
    """
    Least-squares fit of S(n) = 1 / ((1-f) + f·n0/n) for parallel fraction f.

    Returns None if the fit is degenerate.
    """
    pairs = [(n, s) for n, s in zip(n_list, s_list) if s and math.isfinite(s) and n > 0]
    if len(pairs) < 2:
        return None
    # Linearise: 1/S = (1-f) + f·n0/n  →  1/S = a + b / n,  b = f·n0, a = 1-f
    xs = [1.0 / n for n, _ in pairs]
    ys = [1.0 / s for _, s in pairs]
    xbar = _mean(xs)
    ybar = _mean(ys)
    varx = sum((x - xbar) ** 2 for x in xs)
    if varx < 1e-18:
        return None
    cov = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = cov / varx
    a = ybar - b * xbar
    f = 1.0 - a
    # also f = b / n0
    f2 = b / n0 if n0 else f
    return float(max(0.0, min(1.0, 0.5 * (f + f2))))
