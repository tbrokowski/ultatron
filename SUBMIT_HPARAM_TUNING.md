# `submit_hparam_tuning.sh`

Generates per-variant training YAMLs and Slurm launchers for Ultratron hyperparameter ablations. Generated files, logs, checkpoints, submission manifests, and summaries are written to Capstor by default.

## Output Layout

Default roots use `/capstor/scratch/cscs/$USER/ultrasound`.

```text
hparam_tuning/
  configs/<sweep_id>/<run>.yaml
  slurm/<sweep_id>/<run>.sbatch
  slurm/<sweep_id>/<run>.inner.sh
  submissions/<sweep_id>/jobs.tsv
  summaries/<sweep_id>/summary.md
  summaries/<sweep_id>/summary.json
checkpoints/hparam_stability/<sweep_id>/<run>/
logs/hparam_stability/<sweep_id>/<run>/
```

`summary.md` is created by a dependent `afterany` Slurm job. It reports failed jobs, average/median Slurm runtime, finish times, last logged step/phase, and best observed metrics from `metrics.jsonl`.

## Examples

Generate a small stability set without submitting:

```bash
./submit_hparam_tuning.sh --dry-run \
  --variants stable_baseline,lr_5e6,lr_5e5,video_mask_065,cross_loss_half,gram_w2
```

Submit the same hand-picked set and summarize after all jobs finish:

```bash
./submit_hparam_tuning.sh --submit --summary \
  --variants stable_baseline,lr_5e6,lr_5e5,video_mask_065,cross_loss_half,gram_w2 \
  --run-prefix curated_screen_8k
```

Submit baseline plus every configured ablation in parallel:

```bash
./submit_hparam_tuning.sh --all-parallel \
  --run-prefix curated_all_8k
```

Run a short reduced-crop screen:

```bash
./submit_hparam_tuning.sh --submit --summary \
  --variants stable_baseline,lr_5e6,lr_5e5,video_mask_065,cross_loss_half,gram_w2 \
  --steps 2400 \
  --time 04:00:00 \
  --phase-split 0.15,0.25,0.60 \
  --reduced-image-crops \
  --num-workers 2
```

## Launcher Arguments

| params | short and precise explanation | values |
|---|---|---|
| `--submit` | Submit generated Slurm jobs. | default `false`; flag sets `true` |
| `--all-parallel` | Submit baseline plus every configured isolated ablation as independent jobs and add summary dependency. | default `false`; equivalent to `--submit --summary` over all variants |
| `--summary` / `--no-summary` | Add or suppress dependent summary job. | default `false`; `--all-parallel` sets `true` |
| `--dry-run` | Print selected jobs and paths without writing files. | default `false` |
| `--variants` | Explicit comma-separated variants. | default empty; if omitted runs only `stable_baseline` |
| `--list-variants` | Print available variants. | no default; exits after listing |
| `--steps` | Override `curriculum.total_training_steps`. | default from YAML: `8000` |
| `--phase-split` | Set phase lengths; converted to cumulative phase ends. | default `0.15,0.25,0.60` via YAML `0.15/0.40/1.0` |
| `--reduced-image-crops` | Use fewer image crops for cheaper screens. | default `false`; sets global `2`, local `4` unless explicit |
| `--image-global-crops` | Override global image crop count. | YAML default `2` |
| `--image-local-crops` | Override local image crop count. | YAML default `8` |
| `--num-workers` | Override DataLoader workers per rank. | YAML default `4` |
| `--hparam-config` | Hparam YAML path. | default `configs/hparam_tuning.yaml` |
| `--manifest` | Training JSONL manifest passed to `scripts/train.py`. | default `/capstor/scratch/cscs/$USER/ultrasound/manifests/cardiac_breast_brain_maternal_lung_train.jsonl` |
| `--base-configs` | Optional generated config `_base_` list. | default empty |
| `--repo-dir` | Repo path used inside generated launchers. | default script directory |
| `--run-prefix` | Prefix for run names. | default `family` |
| `--sweep-id` | Stable ID used as Capstor subdirectory. | default UTC timestamp + prefix + `selected` or `all` |
| `--generated-root` | Root for generated configs, Slurm scripts, manifests, summaries. | default `/capstor/scratch/cscs/$USER/ultrasound/hparam_tuning` |
| `--config-dir` | Base dir for generated YAMLs; `<sweep_id>` is appended. | default `$generated_root/configs` |
| `--slurm-dir` | Base dir for generated Slurm scripts; `<sweep_id>` is appended. | default `$generated_root/slurm` |
| `--ckpt-root` | Base checkpoint root; `<sweep_id>/<run>` is appended. | default `/capstor/scratch/cscs/$USER/ultrasound/checkpoints/hparam_stability` |
| `--log-root` | Base log root; `<sweep_id>/<run>` is appended. | default `/capstor/scratch/cscs/$USER/ultrasound/logs/hparam_stability` |
| `--nodes` | Slurm nodes per training job. | default `8` |
| `--gpus-per-node` | GPUs per node for torchrun. | default `4` |
| `--cpus-per-task` | CPUs per Slurm task. | default `64` |
| `--time` | Slurm wall time per training job. | default `12:00:00` |
| `--partition` | Slurm partition. | default `normal` |
| `--account` | Slurm account; empty omits directive. | default `a127` |
| `--edf-env` | EDF environment used by `srun --environment`. | default `$HOME/.edf/ultatron.toml` |
| `--train-python` | Python executable inside EDF. | default `python3` |
| `--job-prefix` | Slurm job-name prefix. | default `uhp` |
| `--with-7b` | Enable frozen 7B teacher by not passing `--no-7b`. | default disabled; flag enables |

## Config Parameters And Ablations

This is the single source of truth for the baseline and the one-change-at-a-time ablations. `default` is what `stable_baseline` uses. `possible ablations` are the variant jobs that change that parameter. `--all-parallel` launches `stable_baseline` plus every listed ablation as separate Slurm jobs; each ablation changes one parameter unless the parameters must move together as one policy, such as image/video mask difficulty or heads-only Phase 3.

| params | short and precise explanation | default | possible ablations | references |
|---|---|---|---|---|
| `seed` | Random seed. | `42` | none configured | Local reproducibility |
| `manifest.path` | Fallback training manifest if no `--manifest` is passed. | `manifests/cardiac_breast_brain_maternal_lung_train.jsonl` | CLI `--manifest` | Local data layout |
| `manifest.val_path` | Optional validation manifest. | `null` | none configured | Local datamodule |
| `manifest.root_remap` | Path remapping for manifest entries. | `{}` | none configured | CSCS storage layout |
| `curriculum.total_training_steps` | Total optimizer steps. | `8000` | CLI `--steps` | Local pilot budget; DINO/V-JEPA scaled down |
| `curriculum.image_samples_per_epoch` | Image sampler epoch length. | `120000` | none configured | Local datamodule |
| `curriculum.video_samples_per_epoch` | Video sampler epoch length. | `120000` | none configured | Local datamodule |
| `loaders.image_batch_size` | Per-rank image batch size. | `64` | none configured | USF-MAE batch 64; local memory |
| `loaders.video_batch_size` | Per-rank video batch size. | `16` | none configured | V-JEPA/DISCOVR video cost; local memory |
| `loaders.num_workers` | DataLoader workers per rank. | `4` | CLI `--num-workers` | Local datamodule |
| `loaders.pin_memory` | Use pinned host memory. | `true` | none configured | PyTorch input pipeline |
| `transforms.patch_size` | Mask/padding patch size for current backbones. | `16` | none; patch 8 not wired for current DINO/V-JEPA wrappers | DINOv3/V-JEPA2 patch 16; local code |
| `transforms.image.n_global_crops` | Number of global image crops. | `2` | CLI `--image-global-crops`; reduced mode keeps `2` | DINOv3/OpenUS: 2 global crops/views |
| `transforms.image.n_local_crops` | Number of local image crops. | `8` | CLI `--image-local-crops`; reduced mode `4` | DINOv3/OpenUS: 8 local crops/views |
| `transforms.image.max_global_crop_px` | Starting image crop cap. | `256` | resolution curriculum reaches `384`, `512` | DINOv3 HR adaptation; EchoCare/USFM 224/256 |
| `transforms.image.mask_strategy` | Image masking implementation. | `freq` | none configured; code supports `{freq, spatial, both}` | USFM frequency masking; local transforms |
| `transforms.image.freq_mask.mask_ratio` | Image frequency mask ratio. | `0.40` | `img_mask_025=0.25`, `img_mask_030=0.30`, `img_mask_050=0.50`, `img_mask_080=0.80` | USFM `0.4`; USF-MAE `0.25`; OpenUS `0.80` |
| `transforms.image.freq_mask.n_bands` | Number of frequency bands masked. | `1` | `image_bands_2=2` | USFM 2-of-7 frequency-band sampling |
| `transforms.image.freq_mask.use_alp_bias` | Bias frequency masks with ALP difficulty. | `false` | none configured | OpenUS ALP; local transform |
| `transforms.image.spatial_mask_ratio` | Random spatial image mask ratio. | `0.40` | `img_mask_025=0.25`, `img_mask_030=0.30`, `img_mask_050=0.50`, `img_mask_080=0.50 cap` | DINOv3 iBOT `[0.1,0.5]`; USFM `0.4`; USF-MAE high-mask caution |
| `transforms.video.n_frames` | Frames per clip. | `16` | `video_frames_32=32` | V-JEPA2 16 primary / 64 cooldown; DISCOVR 64 |
| `transforms.video.temporal_stride` | Frame stride / fps proxy. | `4` | `video_stride_2=2` | V-JEPA2 4 fps; cardiac/fetal motion tuning |
| `transforms.video.tube_size` | Temporal grouping for video mask spans. | `2` | none configured | V-JEPA2/DISCOVR tubelet anchor `2 x 16 x 16` |
| `transforms.video.tube_mask_ratio` | Video tube mask ratio. | `0.75` | `video_mask_050=0.50`, `video_mask_065=0.65`, `video_mask_080=0.80`, `video_mask_090=0.90` | V-JEPA2, DISCOVR, UltraFedFM |
| `transforms.video.mask_strategy` | Video masking implementation. | `freq` | none configured; code supports `{freq, spatial, both}` | USFM frequency prior; V-JEPA masked latent prediction |
| `transforms.video.max_crop_px` | Starting video crop cap. | `112` | resolution curriculum reaches `128`, `160` | DISCOVR fast path; V-JEPA high-res reserved |
| `transforms.video.spatial_mask_ratio` | Spatial part of video masking. | `0.40` | `video_mask_050=0.30`, `video_mask_065=0.35`, `video_mask_080=0.45`, `video_mask_090=0.50` | V-JEPA block masks; OpenUS adaptive-mask caution |
| `transforms.video.freq_mask.mask_ratio` | Video frequency mask ratio. | `0.75` | `video_mask_050=0.50`, `video_mask_065=0.65`, `video_mask_080=0.80`, `video_mask_090=0.90` | USFM frequency masking extended to video |
| `transforms.video.freq_mask.n_bands` | Number of video frequency bands masked. | `1` | `video_bands_2=2` | USFM frequency masking extended to video |
| `train.phase1_frac` | End of image warm-start phase. | `0.15` | CLI `--phase-split`; `heads_only_phase3=0.0` | Oura image warm start |
| `train.phase2_frac` | Cumulative end of video warm-start phase. | `0.40` | CLI `--phase-split`; `heads_only_phase3=0.0` | Oura/V-JEPA video warm start |
| `train.phase3_frac` | Cumulative end of hybrid phase. | `1.0` | CLI `--phase-split` | Oura cross-branch coupling |
| `train.base_lr` | AdamW learning rate. | `1e-5` | `lr_5e6=5e-6`, `lr_2e5=2e-5`, `lr_5e5=5e-5`, `lr_1e4=1e-4` | DINOv3, V-JEPA2, URFM, UltraFedFM |
| `train.weight_decay` | AdamW weight decay. | `0.04` | `wd_001=0.01`, `wd_008=0.08` | DINOv3/V-JEPA2 `0.04`; URFM/UltraFedFM `0.05` |
| `train.beta1`, `train.beta2` | AdamW betas. | `(0.9, 0.95)` | none configured | V-JEPA2/DISCOVR/URFM/UltraFedFM |
| `train.grad_clip` | Max gradient norm. | `1.0` | `clip_05=0.5` | USF-MAE `1.0`; local stability |
| `train.warmup_steps_p1/p2/p3` | Warmup steps per phase. | `500/500/500` | none configured | DINOv3/V-JEPA warmup principle |
| `train.ema_momentum` | EMA teacher momentum. | `0.999` | `ema_9995=0.9995`, `ema_9998=0.9998` | DINOv3 `0.999`; V-JEPA2 `0.99925`; OpenUS/DISCOVR `0.996` |
| `train.lam1` | Image CLS/global DINO-style loss weight. | `1.0` | none configured | DINOv3 global DINO loss |
| `train.lam2` | Dense iBOT/image patch loss weight. | `1.0` | none configured | DINOv3 iBOT; dense ultrasound semantics |
| `train.lam3` | Local crop CLS weight. | `0.5` | none configured | DINOv3/OpenUS local crops/views |
| `train.lam4` | Video CLS/clip self-distillation weight. | `1.0` | none configured | V-JEPA2 video branch |
| `train.lam5` | Video masked tube prediction weight. | `1.0` | none configured | V-JEPA2 masked latent prediction |
| `train.lam6` | Cross patch-to-tube distillation weight. | `1.0` | `cross_loss_half=0.5`, `cross_loss_025=0.25` | DISCOVR; Oura cross-branch distillation |
| `train.lam6_nce` | Cross-branch InfoNCE weight. | `0.5` | `nce_025=0.25` | Oura contrastive coupling |
| `train.lam6_nce_temp` | InfoNCE temperature. | `0.07` | `nce_temp_010=0.10` | DINO/OpenUS temperature scale; local contrastive loss |
| `train.lam7` | Prototype consistency weight. | `0.5` | `no_proto=0`, `proto_025=0.25` | DISCOVR SCD/prototypes; Oura prototype consistency |
| `train.lam_7b` | Frozen DINOv3-7B teacher loss weight. | `0.0` | none configured; `--with-7b` only enables loading | DINOv3 distillation optionality |
| `train.lam_gram` | Gram anchoring loss weight. | `1.0` | `gram_off=0`, `gram_w05=0.5`, `gram_w2=2.0` | DINOv3 `wGram=2`; dense-feature safeguard |
| `train.lam_koleo` | Single-branch KoLeo weight. | `0.1` but inactive because `use_koleo=false` | none configured | DINOv3 KoLeo |
| `train.lam_koleo_cross` | Cross-branch KoLeo uniformity weight. | `0.1` | `cross_koleo_005=0.05`, `no_cross_koleo=0` | DINOv3 KoLeo idea extended to Oura |
| `train.use_koleo` | Enable single-branch KoLeo in Phase 1. | `false` | none configured | Local stability choice; DINOv3 source prior |
| `train.gram_start_step` | First step where Gram anchoring starts. | `4000` | `gram_start_30=2400`, `gram_start_70=5600` | DINOv3 early Gram teacher; short-run scaling |
| `train.gram_refresh_interval` | Gram teacher refresh period. | `2000` | `gram_refresh_1000=1000` | DINOv3 10k refresh scaled down |
| `train.res_step_1/2` | Image resolution curriculum boundaries. | `6000`, `7600` | none configured | DINOv3 HR adaptation; V-JEPA cooldown idea |
| `train.res_px_1/2/3` | Image crop cap schedule. | `256 -> 384 -> 512` | none configured | DINOv3 HR adaptation; EchoCare/USFM |
| `train.res_vid_step_1/2` | Video resolution curriculum boundaries. | `6000`, `7600` | none configured | V-JEPA cooldown structure |
| `train.res_vid_px_1/2/3` | Video crop cap schedule. | `112 -> 128 -> 160` | none configured | DISCOVR low-res fast path |
| `train.checkpoint_every` | Checkpoint interval. | `1000` | none configured | Local training loop |
| `train.log_every` | Metric logging interval. | `25` | none configured | Local training loop |
| `model.image_backbone` | Image backbone. | `dinov3_s` | `image_backbone_l=dinov3_l` | Cached DINOv3-S pilot baseline; DINOv3-L cached scale-up |
| `model.video_backbone` | Video backbone. | `vjepa2_l` | none configured | V-JEPA2 ViT-L pilot anchor |
| `model.frozen_teacher` | Optional frozen DINO teacher. | `null` | none configured; script default passes `--no-7b` | DINOv3 7B distillation optionality |
| `model.ema_momentum` | Model-side EMA value for consistency. | `0.999` | `ema_9995=0.9995`, `ema_9998=0.9998` | DINOv3/V-JEPA/OpenUS EMA anchors |
| `model.n_prototypes` | Number of prototype vectors. | `256` | `proto_128=128`, `proto_512=512` | DISCOVR prototypes; Oura prototype search |
| `model.align_dim` | Cross-modal projection dimension. | `1024` | `align_dim_512=512`, `align_dim_256=256` | Oura alignment head; memory/regularization |
| `model.dtype` | Training dtype. | `bfloat16` | none configured | DINOv3 bf16; CSCS H100 path |
| `model.hf_cache_dir` | Explicit Hugging Face cache override. | `null` | none configured | Local model loading |
| `model.use_gradient_checkpointing` | Activation checkpointing. | `true` | none configured | Local memory requirement for large ViTs |
| `model.trainable_image_layers` | Image backbone trainability. | `4` | `image_full_backbone=null`, `image_last8_layers=8`, `heads_only_phase3=0` | URFM/UltraFedFM fine-tuning; local last-K support |
| `model.trainable_video_layers` | Video backbone trainability. | `4` | `video_full_backbone=null`, `video_last8_layers=8`, `heads_only_phase3=0` | V-JEPA scaling cost; local last-K support |
| `anatomy_weights` | Anatomy-family sampler weights. | cardiac `0.8`, breast/brain `1.0`, fetal/intrapartum `1.2`, lung `1.3` | none configured | USFM organ balancing; URFM/EchoCare multi-organ caution |

## Removed Or Not Yet Configured

| params | short and precise explanation | values | references |
|---|---|---|---|
| `train.force_stage` | Removed because it was not consumed by `TrainConfig`; `heads_only_phase3` now uses phase fractions only. | removed | Local code audit |
| DINOv3-B image backbone | Curated as the first serious image-branch target, but not configured in this runnable sweep because the gated HF model is not present in the shared cache. | not configured until cached/authenticated | DINOv3; local CSCS cache audit |
| Patch size 8 | Curated as high-value for ultrasound fine structures, but not enabled here because current DINOv3/V-JEPA wrappers are patch-16. | not configured | DINOv3/V-JEPA2 patch-16 compatibility |
| V-JEPA 2.1 dense variant | Worth tracking for dense ultrasound tokens, but no local `vjepa2_1_*` registry key exists yet. | not configured | V-JEPA2 GitHub update |
| MedSAM-3 / UltraSam knobs | Downstream Phase 4 segmentation/agent settings, not part of this SSL launcher. | not configured here | MedSAM-3, UltraSam |

## Source Notes

References in the tables use the curated notes from the current request plus local code inspection. Direct links provided in the request:

- V-JEPA2 config: https://raw.githubusercontent.com/facebookresearch/vjepa2/main/configs/train/vitg16/pretrain-256px-16f.yaml
- V-JEPA2 repo: https://github.com/facebookresearch/vjepa2
- DISCOVR arXiv: https://arxiv.org/html/2506.11777v3
- DISCOVR repo: https://github.com/mdivyanshu97/DISCOVR
- DISCOVR pretraining parser: https://raw.githubusercontent.com/mdivyanshu97/DISCOVR/master/scripts/run_mae_pretraining.py
- OpenUS repo: https://github.com/XZheng0427/OpenUS

Other source labels refer to the user-curated paper notes for DINOv3, UltraSam, MedSAM-3, USFM, UltraFedFM, URFM, EchoCare, and USF-MAE.
