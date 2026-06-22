#!/usr/bin/env python3
"""Generate finetune comparison reports and charts."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from finetune.experiment_registry import DASHBOARD_DIR, RESULTS_ROOT, SWEEPS
from finetune.report import generate_comparison_report, generate_dashboard
from finetune.results_collector import (
    collect_all_sweeps,
    collect_sweep,
    write_backfilled_results,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ultatron finetune report generator")
    parser.add_argument("--sweep", choices=list(SWEEPS.keys()), default=None)
    parser.add_argument("--all-sweeps", action="store_true")
    parser.add_argument("--from-logs", action="store_true", help="Include log fallback / backfill")
    parser.add_argument("--logs-only", action="store_true", help="Parse logs only, write results.json")
    parser.add_argument("--dashboard", action="store_true", help="Only regenerate master dashboard")
    args = parser.parse_args()

    if args.dashboard:
        tree = collect_all_sweeps(from_logs=True)
        generate_dashboard(tree, DASHBOARD_DIR)
        log.info("Dashboard → %s", DASHBOARD_DIR)
        return

    if args.logs_only or args.from_logs:
        tree = collect_all_sweeps(from_logs=True, logs_only=args.logs_only)
        n = write_backfilled_results(tree)
        log.info("Backfilled %d results.json files", n)

    if args.all_sweeps:
        for name, sweep_dir in SWEEPS.items():
            generate_comparison_report(
                sweep_dir,
                from_logs=args.from_logs or args.logs_only,
            )
            log.info("Sweep report → %s", sweep_dir)
        tree = collect_all_sweeps(from_logs=args.from_logs or args.logs_only)
        generate_dashboard(tree, DASHBOARD_DIR)
        log.info("Dashboard → %s", DASHBOARD_DIR)
        return

    sweep = args.sweep or "representative"
    sweep_dir = SWEEPS[sweep]
    generate_comparison_report(sweep_dir, from_logs=args.from_logs or args.logs_only)
    tree = collect_all_sweeps(from_logs=args.from_logs)
    generate_dashboard(tree, DASHBOARD_DIR)


if __name__ == "__main__":
    main()
