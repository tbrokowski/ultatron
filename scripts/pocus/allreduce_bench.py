#!/usr/bin/env python3
"""
allreduce_bench.py  ·  E6 NCCL validation (Slingshot vs sockets)
================================================================

Measures all-reduce bus bandwidth in the Ultatron container.
WP1 reference: 152.9 GB/s over Slingshot, 11.3 GB/s over sockets.

Bus bandwidth for all-reduce:  2 · (n−1)/n · bytes / time
"""
from __future__ import annotations

import argparse
import json
import os
import time


def bus_bandwidth_GBps(nbytes: int, seconds: float, n_ranks: int) -> float:
    if seconds <= 0 or n_ranks <= 1:
        return float("nan")
    return 2.0 * (n_ranks - 1) / n_ranks * nbytes / seconds / 1e9


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bytes", type=int, default=1 << 30, help="payload size (default 1 GiB)")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()

    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    nbytes = args.bytes
    n_elem = nbytes // 4
    t = torch.empty(n_elem, device="cuda", dtype=torch.float32)
    dist.barrier()
    for _ in range(args.warmup):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / args.iters
    bw = bus_bandwidth_GBps(nbytes, dt, world)
    nccl_net = os.environ.get("NCCL_NET", "(unset — plugin default)")
    result = {
        "rank": rank,
        "world_size": world,
        "bytes": nbytes,
        "iters": args.iters,
        "seconds_per_allreduce": dt,
        "bus_bandwidth_GBps": bw,
        "NCCL_NET": nccl_net,
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG"),
        "hostname": os.uname().nodename,
        "wp1_reference": {"slingshot_GBps": 152.9, "socket_GBps": 11.3},
    }
    if rank == 0:
        print(json.dumps(result, indent=2))
        if args.out:
            Path = __import__("pathlib").Path
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
