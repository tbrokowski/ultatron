#!/usr/bin/env python3
"""Quick sanity check for LUS patient-level MIL dataset and throughput estimates."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from data.adapters.lung.benin_lus import BeninLUSAdapter

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEFAULT_BENIN = "/capstor/store/cscs/swissai/a127/ultrasound/raw/lung/Benin_Videos"


def benin_split_stats(root: str) -> dict[str, dict]:
    adapter = BeninLUSAdapter(root)
    by_split: dict[str, dict] = defaultdict(lambda: {"patients": set(), "clips": 0, "tb_pos": 0})
    patient_tb: dict[str, int] = {}

    for entry in adapter.iter_entries():
        if entry.modality_type != "video":
            continue
        split = entry.split
        pid = entry.study_id
        by_split[split]["patients"].add(pid)
        by_split[split]["clips"] += 1
        pl = (entry.source_meta or {}).get("patient_labels") or {}
        patient_tb[pid] = int(pl.get("tb", 0))

    out = {}
    for split, stats in by_split.items():
        patients = stats["patients"]
        out[split] = {
            "patients": len(patients),
            "clips": stats["clips"],
            "tb_pos": sum(patient_tb[p] for p in patients),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug LUS patient MIL dataset")
    parser.add_argument("--benin-root", default=DEFAULT_BENIN)
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument("--load-batch", type=int, default=4, help="Patients to decode for timing")
    args = parser.parse_args()

    root = args.benin_root
    if not Path(root).exists():
        log.error("Benin root not found: %s", root)
        sys.exit(1)

    log.info("=== Benin patient bags (adapter 80/10/10 splits) ===")
    stats = benin_split_stats(root)
    for split in ("train", "val", "test"):
        s = stats.get(split, {"patients": 0, "clips": 0, "tb_pos": 0})
        avg = s["clips"] / max(s["patients"], 1)
        log.info(
            "%s: %d patients, %d clips (avg %.1f/patient), TB+=%d",
            split, s["patients"], s["clips"], avg, s["tb_pos"],
        )

    train_clips = stats.get("train", {}).get("clips", 0)
    eff_clips = min(args.max_clips, train_clips) if train_clips else 0
    train_patients = stats.get("train", {}).get("patients", 0)
    if train_patients:
        avg_train = train_clips / train_patients
        eff = train_patients * min(args.max_clips, int(avg_train + 0.5))
        log.info(
            "\nWith max_clips=%d: ~%d clips/epoch vs %d (all clips), batch_size=8 → ~%d steps",
            args.max_clips, eff, train_clips, (train_patients + 7) // 8,
        )

    try:
        import torch  # noqa: F401
        from finetune.experiments.lus_patient import LUSPatientBagDataset
    except ImportError:
        log.info("\n(torch not available — skipping decode timing)")
        return

    log.info("\n=== Decode timing (%d patients) ===", args.load_batch)
    ds = LUSPatientBagDataset(
        root_benin=root,
        split="train",
        include_rsa=False,
        max_clips_per_patient=args.max_clips,
    )
    ds.set_epoch(0)
    n = min(args.load_batch, len(ds))
    t0 = time.perf_counter()
    for i in range(n):
        _ = ds[i]
    elapsed = time.perf_counter() - t0
    per_patient = elapsed / max(n, 1)
    log.info("Decoded %d patients in %.1fs (%.2fs/patient)", n, elapsed, per_patient)
    log.info("Rough decode-only train epoch: %.0f min", (per_patient * len(ds)) / 60)


if __name__ == "__main__":
    main()
