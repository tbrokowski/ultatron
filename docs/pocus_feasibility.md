# WP5 POCUS scaling experiments — runbook

Version 1 harness in this repo (11 Sep 2026 spec, T. Brokowski).

This tree cannot submit to Clariden from CI.  On a login node:

1. Copy `edf/ultatron.toml.example` to `~/.edf/ultatron.toml` and set `image`.
   Confirm `[annotations] com.hooks.aws_ofi_nccl.enabled = "true"`.
2. Confirm account `[TBC: a127 or infra01]`: `export POCUS_ACCOUNT=...`
3. Licence checks `[TBC]` for US-365K and CardiacUDC; CAMUS is the CardiacUDC fallback.
4. Download / reuse Alps copies: `bash scripts/pocus/download.sh`
5. Inspect US-365K attributes: `python3 scripts/pocus/inspect_us365k.py --out manifests/us365k_fields.json`
6. Manifests (seed 1234): `python3 scripts/pocus/build_manifests.py --out $MANIFESTS`
7. Shards + drop unreadable files: `python3 scripts/pocus/build_shards.py`
8. Data facts: `python3 scripts/pocus/data_facts.py --out $EVIDENCE/data_facts.json`
9. E6 NCCL (Slingshot, then `--sockets`): `bash scripts/pocus/submit_nccl.sh`
10. E0 / R0 OOM probes, then E1/E2 in parallel with R1/R2.
    `bash scripts/pocus/submit_encoder.sh E1 --nodes 1`
    Repeats at smallest and largest scale: `--repeat`
11. After logs land: `python3 scripts/pocus/analyse.py --evidence $EVIDENCE`
    then `scaling_plot.py`, `gantt.py`, `wp5_results.py`
12. ACLs: `bash scripts/pocus/set_acls.sh`

Encoder entry point: `python -m train.student_pretrain` (Slingshot, rank-0 logs,
`--bench-stage`, `--bench-window`, `--per-step-timing`, `--no-ckpt`,
`--ckpt-probe`, `--loader-only`).

Open decisions remain marked `[TBC]` / `[DEFAULT]` in the spec: account, GBS
vs production yaml, 100k from step 0 vs continue from 30,800, RL prompt set P
and epochs, 8B vs 32B teacher.
