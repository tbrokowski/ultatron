# Frozen-backbone segmentation head comparison

All finetune runs keep the encoder **frozen** (`freeze_backbone: true`). Only the task head
(and its adapters) are trained. This document defines the two enhanced head families and
how they map to backbones in the comparison sweep.

## Head types

| `head_type` | Backbone requirement | Input features | Architecture |
|-------------|---------------------|----------------|--------------|
| `upernet` | Hierarchical (`embed_dims` with 4 stages) | F1, F2, F3, F4 | `UPerNetDecoder`: SegAdapters → FPN + attention gates → ASPP fusion → optional `refine_up` |
| `dpt` | Any (patch-token encoder) | `patch_tokens` | `EnhancedDPTSegHead`: SegAdapter → conv neck → ASPP → optional `refine_up` |

### Student encoder (Hiera)

`StudentEncoder.encode_image()` exposes both paths:

- **UPerNet** (`head_type: upernet`): uses all four scales F1–F4 (strides 4/8/16/32).
- **DPT equivalent** (`head_type: dpt`): uses F4 only via `patch_tokens` — single-scale ablation
  on the same frozen backbone. Same enhancements (adapter, ASPP, boundary loss, refine_up) but
  without multi-scale fusion.

### Other backbones (DINOv3, ViT, USFM, OpenUS, …)

- **`dpt`**: `EnhancedDPTSegHead` on final patch tokens (standard ViT dense head path).
- **`upernet`**: skipped automatically (not hierarchical).

## Shared training protocol (segmentation)

Applied via `smoke_overrides` in `comparison_representative.yaml` and per-dataset YAML:

| Setting | Default | Effect |
|---------|---------|--------|
| `boundary_loss_weight` | `0.5` | BCE on dilation−erosion boundary ring |
| `refine_up` | `true` | Learned 2× upsample before final bilinear to GT size |
| `seg_use_adapters` | `true` | Per-scale / token SegAdapters (zero-init residual) |
| `seg_use_aspp` | `true` | ASPP context module in decoder neck |
| `seg_use_attention_gates` | `true` | FPN gating (UPerNet only) |

Loss (binary seg): **BCE + Dice + boundary BCE** (`models/heads/seg_losses.py`).

## Comparison sweep layout

Results are written to:

```
{output_dir}/{backbone_key}/{experiment}/{head_type}/results.json
```

### BUSI / BUSI-multitask

```yaml
head_types:
  busi: [upernet, dpt]
```

For each student checkpoint you get two rows:

- `student_stage1/busi/upernet` — primary multi-scale head (target ~0.84 Dice)
- `student_stage1/busi/dpt` — F4-only ablation on same backbone

For DINOv3 / USFM / etc.:

- `dinov3_l/busi/dpt` only (`upernet` skipped)

### CAMUS

```yaml
head_types:
  camus: [dpt, upernet]
```

Same pattern: all backbones get `dpt`; student additionally runs `upernet`.

## Running

```bash
# Full representative sweep (BUSI gets upernet + dpt per backbone)
python scripts/finetune/run_all.py

# BUSI only, student backbones
python scripts/finetune/busi.py --backbones student_stage1 student_stage3

# Standalone BUSI finetune (UPerNet, 500px)
python -m finetune.experiments.busi --config configs/finetune/busi.yaml ...
```

## Legacy heads

- `dpt_legacy` → original `DPTSegHead` (no adapter/ASPP/refine)
- `linear` → linear probe baseline
