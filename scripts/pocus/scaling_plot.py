#!/usr/bin/env python3
"""
scaling_plot.py  ·  Template-style strong-scaling plots for WP5
===============================================================

Reads runs.csv (workload,gpus,throughput,unit,parallel_setup,jobid,log)
and writes PNG/PDF under plots/.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train.bench import efficiency, speedup


def load_runs(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                r["gpus"] = int(float(r["gpus"]))
                r["throughput"] = float(r["throughput"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(r["throughput"]):
                rows.append(r)
    return rows


def plot_series(ax, xs, ys, label, ylabel):
    ax.plot(xs, ys, marker="o", label=label)
    ax.set_xlabel("GPUs")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    ax.legend()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plots", file=sys.stderr)
        return

    rows = load_runs(args.runs)
    by_wl = defaultdict(list)
    for r in rows:
        by_wl[r["workload"]].append(r)

    args.out.mkdir(parents=True, exist_ok=True)
    for wl, items in sorted(by_wl.items()):
        items = sorted(items, key=lambda r: r["gpus"])
        xs = [r["gpus"] for r in items]
        ys = [r["throughput"] for r in items]
        unit = items[0].get("unit") or "1/s"
        rates = {r["gpus"]: r["throughput"] for r in items}
        s = speedup(rates)
        e = efficiency(rates)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        plot_series(axes[0], xs, ys, wl, f"throughput ({unit})")
        plot_series(axes[1], list(s), list(s.values()), wl, "speed-up S(n)")
        if len(xs) >= 2:
            axes[1].plot(xs, [x / xs[0] for x in xs], ls="--", color="gray", label="ideal")
            axes[1].legend()
        plot_series(axes[2], list(e), list(e.values()), wl, "efficiency E(n)")
        axes[2].axhline(0.70, ls=":", color="tab:red", label="n* threshold")
        axes[2].legend()
        fig.suptitle(wl)
        fig.tight_layout()
        fig.savefig(args.out / f"{wl}_scaling.png", dpi=140)
        fig.savefig(args.out / f"{wl}_scaling.pdf")
        plt.close(fig)

        # component bars if extra columns exist
        print(f"wrote {args.out / (wl + '_scaling.png')}")


if __name__ == "__main__":
    main()
