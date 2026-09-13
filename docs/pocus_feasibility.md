# WP5 POCUS scaling experiments — runbook

Version 1 harness in this repo (11 Sep 2026 spec, T. Brokowski).

This tree cannot submit to Clariden from CI.  On a login node:

1. Copy `edf/ultatron.toml.example` to `~/.edf/ultatron.toml` and set `image`.
   Confirm `[annotations] com.hooks.aws_ofi_nccl.enabled = "true"`.
2. Account is `a0238` (Slurm `--account` and writable store). Optional:
   `export POCUS_ACCOUNT=a0238`. Evidence lands under
   `/capstor/store/cscs/swissai/a0238/meditron-feasibility-review/pocus`.
   Existing US-365K / video copies / student ckpts are still read from a127.
3. Licence checks `[TBC]` for US-365K and CardiacUDC; CAMUS is the CardiacUDC fallback.
4. Download / reuse Alps copies: `bash scripts/pocus/download.sh`
5. Inspect US-365K attributes: `python3 scripts/pocus/inspect_us365k.py --out manifests/us365k_fields.json`
6. Manifests (seed 1234): `python3 scripts/pocus/build_manifests.py --out $MANIFESTS`
7. Shards + drop unreadable files: `python3 scripts/pocus/build_shards.py`
8. Data facts: `python3 scripts/pocus/data_facts.py --out $EVIDENCE/data_facts.json`
9. E6 NCCL (Slingshot, then `--sockets`): `bash scripts/pocus/submit_nccl.sh`
10. Launch everything (or step through):
    `bash scripts/pocus/launch_all.sh --dry-run`   # print the campaign
    `bash scripts/pocus/launch_all.sh`             # E0–E6 and R0–R5 with Slurm deps
    `bash scripts/pocus/launch_all.sh --prep-data` # download + manifests + shards first
    `bash scripts/pocus/submit_encoder.sh E0`      # 1 GPU MBS probe only
    Repeats at 1 and 8 nodes are included in `launch_all.sh`.
11. After logs land: `python3 scripts/pocus/analyse.py --evidence $EVIDENCE`
    then `scaling_plot.py`, `gantt.py`, `wp5_results.py`
12. ACLs: `bash scripts/pocus/set_acls.sh`

`--video-manifest` is merged into a combined JSONL (`ssl_stream=image|video`) so
stage-2/3/4 runs actually see the three video datasets. E1 pins `--forced-type image`.

Encoder entry point: `python -m train.student_pretrain` (Slingshot, rank-0 logs,
`--bench-stage`, `--bench-window`, `--per-step-timing`, `--no-ckpt`,
`--ckpt-probe`, `--loader-only`).

Open decisions remain marked `[TBC]` / `[DEFAULT]` in the spec: GBS vs
production yaml, 100k from step 0 vs continue from 30,800, RL prompt set P
and epochs, 8B vs 32B teacher.
