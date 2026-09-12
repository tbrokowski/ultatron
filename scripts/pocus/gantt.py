#!/usr/bin/env python3
"""
gantt.py  ·  Feed job-level GPUh and wall-clock into the WP5 Gantt fields
=========================================================================

Reads analysis.json and writes gantt.json + a markdown fragment with
T_enc, T_RL, monthly GPUh vs the 47,500 GPUh/month pro-rata share.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

MONTHLY = 47_500


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    gpuh = analysis.get("gpuh") or {}
    enc = gpuh.get("encoder") or {}
    rl = gpuh.get("rl") or {}
    t_enc = enc.get("T_enc_days_with_queue_margin")
    t_rl = rl.get("T_RL_days_with_queue_margin")
    gpuh_enc = enc.get("GPUh_enc_total")
    gpuh_rl = rl.get("GPUh_RL_total")
    monthly = (gpuh_enc or 0) + (gpuh_rl or 0)
    out = {
        "T_enc_days": t_enc,
        "T_RL_days": t_rl,
        "GPUh_enc_total": gpuh_enc,
        "GPUh_RL_total": gpuh_rl,
        "monthly_gpuh_implied": monthly,
        "monthly_pro_rata": MONTHLY,
        "monthly_fraction_of_pro_rata": (monthly / MONTHLY) if MONTHLY else None,
        "note": (
            "Place WP5 encoder then RL on the Gantt using T_enc and T_RL "
            "(already include +20 % queueing margin). Compare monthly GPUh "
            f"with the pro-rata {MONTHLY:,} GPUh/month."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    md = args.out.with_suffix(".md")
    md.write_text(
        "\n".join([
            "### WP5 timeline (from gantt.py)",
            "",
            f"- Encoder wall-clock (with 20 % queue margin): {t_enc} days",
            f"- RL wall-clock (with 20 % queue margin): {t_rl} days",
            f"- Encoder GPUh request: {gpuh_enc}",
            f"- RL GPUh request: {gpuh_rl}",
            f"- Monthly GPUh implied: {monthly} / {MONTHLY:,} pro-rata",
            "",
        ]),
        encoding="utf-8",
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
