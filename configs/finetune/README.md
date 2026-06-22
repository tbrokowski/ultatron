# Finetune comparison configs

All finetune metrics and charts live under `results/finetune/`.

## Sweeps

| Config | Output dir | Experiments |
|---|---|---|
| `comparison_representative.yaml` | `results/finetune/representative` | busi, busi_multitask, camus, echonet, lus, lus_video |
| `openus_segmentation.yaml` | `results/finetune/openus_seg` | busbra, tn3k |
| `comparison_fetal_planes.yaml` | `results/finetune/fetal_planes` | fetal_planes_db |

## Run finetune

```bash
python scripts/finetune/run_all.py
python scripts/finetune/tn3k.py --comparison-config configs/finetune/openus_segmentation.yaml
bash scripts/finetune/submit_tn3k.sh
```

## Generate reports

```bash
# Backfill from Slurm logs + build unified dashboard
python scripts/finetune/report.py --from-logs --all-sweeps

# Refresh dashboard only
python scripts/finetune/report.py --dashboard
```

Outputs:
- Per-sweep: `results/finetune/{sweep}/comparison_report.md` and `charts/`
- Unified: `results/finetune/_dashboard/summary_primary_metric.png`

## Cleanup legacy outputs

```bash
bash scripts/finetune/clean_legacy_outputs.sh
```
