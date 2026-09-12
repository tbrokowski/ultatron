#!/usr/bin/env python3
"""
inspect_us365k.py  ·  Does the HF US-365K release carry structured attributes?

Writes a JSON report: keys present, whether captions-only, sample records.
Used before R4 to decide whether the teacher must extract attributes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.pocus.paths import DATASET_DEFAULTS
from scripts.pocus.sampling import load_jsonl, metadata_field_report


def _find_metadata(root: Path) -> list[Path]:
    meta = root / "metadata"
    files = []
    for split in ("train", "val", "test"):
        p = meta / f"{split}.jsonl"
        if p.exists():
            files.append(p)
    if not files:
        files = sorted(root.rglob("*.jsonl"))[:6]
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DATASET_DEFAULTS["US-365K"])
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    files = _find_metadata(args.root)
    if not files:
        report = {
            "root": str(args.root),
            "exists": args.root.exists(),
            "error": "no jsonl metadata found",
            "captions_only": None,
        }
    else:
        rows = []
        for f in files:
            part = load_jsonl(f)
            rows.extend(part[: args.max_rows])
            if len(rows) >= args.max_rows:
                rows = rows[: args.max_rows]
                break
        report = metadata_field_report(rows)
        report["root"] = str(args.root)
        report["files"] = [str(f) for f in files]
        report["sample"] = rows[:2]
        if report["captions_only"]:
            report["r4_action"] = (
                "US-365K has captions only. R4 teacher extracts structured "
                "attributes from each caption; exact-match reward limited to "
                "organ and diagnosis parsed from captions."
            )
        else:
            report["r4_action"] = (
                "Structured attribute fields present; teacher uses them as "
                "reference and exact-match applies to organ/diagnosis."
            )
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
