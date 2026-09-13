"""
train/ddp_hooks.py  ·  Timed DDP all-reduce communication hook
==============================================================

Registers a host-timed all-reduce hook so per-step JSONL can record
``t_allreduce``.  Sampled every ``every`` steps (spec §5: every 50 steps)
to keep the hook cheap; intervening steps reuse the last measurement.
"""
from __future__ import annotations

import time
from typing import Optional


class AllreduceTimer:
    """Accumulates wall-clock time spent in DDP gradient all-reduces."""

    def __init__(self, every: int = 50):
        self.every = max(1, int(every))
        self.t_allreduce = 0.0
        self.last_sample = 0.0
        self.n_calls = 0
        self._step = 0
        self._active = False

    def reset_step(self) -> None:
        self.t_allreduce = 0.0
        self._step += 1
        self._active = (self._step % self.every) == 0
        if not self._active:
            # Reuse last sample so the JSONL field is populated every step.
            self.t_allreduce = self.last_sample

    def hook(self, state, bucket):  # noqa: ANN001 — PyTorch comm-hook signature
        import torch.distributed as dist

        if not self._active:
            fut = dist.all_reduce(bucket.buffer(), op=dist.ReduceOp.SUM, async_op=True).get_future()
            return fut.then(lambda f: f.wait()[0])

        t0 = time.perf_counter()
        fut = dist.all_reduce(bucket.buffer(), op=dist.ReduceOp.SUM, async_op=True).get_future()

        def _then(f):
            dt = time.perf_counter() - t0
            self.t_allreduce += dt
            self.last_sample = self.t_allreduce
            self.n_calls += 1
            return f.wait()[0]

        return fut.then(_then)


def try_register_allreduce_timer(module, every: int = 50) -> Optional[AllreduceTimer]:
    """Register the hook on a DDP module. Returns None if not DDP."""
    import torch.nn as nn

    if not isinstance(module, nn.parallel.DistributedDataParallel):
        return None
    timer = AllreduceTimer(every=every)
    try:
        module.register_comm_hook(state=None, hook=timer.hook)
    except Exception:
        return None
    return timer
