# `submit_hparam_tuning.sh`

Generate per-variant training YAMLs and Slurm launchers for Ultratron
hyperparameter ablations. Add `--submit` to enqueue the generated jobs.

Generated configs and Slurm files go to Capstor by default. Checkpoints and
logs also go to Capstor, not the repo.

## Arguments

| Argument | Meaning |
|---|---|
| `--submit` | Submit generated Slurm jobs with `sbatch`. |
| `--dry-run` | Print selected settings without writing files. |
| `--profile` | Use a variant group from `configs/hparam_tuning.yaml`. |
| `--variants` | Run explicit comma-separated variants. |
| `--list-variants` | Print all available variants and exit. |
| `--steps` | Override total training steps. |
| `--phase-split` | Phase lengths, e.g. `0.15,0.25,0.60`. |
| `--reduced-image-crops` | Use `2` global and `4` local image crops unless overridden. |
| `--image-global-crops` | Override global image crop count. |
| `--image-local-crops` | Override local image crop count. |
| `--num-workers` | Override DataLoader workers. |
| `--hparam-config` | Use a different hparam YAML. |
| `--manifest` | Training JSONL manifest. |
| `--base-configs` | Optional generated config `_base_` list. |
| `--repo-dir` | Repo path used inside Slurm scripts. |
| `--run-prefix` | Prefix for generated run names. |
| `--generated-root` | Root for generated configs and Slurm scripts. |
| `--config-dir` | Directory for generated train YAMLs. |
| `--slurm-dir` | Directory for generated Slurm scripts. |
| `--ckpt-root` | Root checkpoint directory. |
| `--log-root` | Root log directory. |
| `--nodes` | Number of Slurm nodes. |
| `--gpus-per-node` | GPUs per node. |
| `--cpus-per-task` | CPUs per Slurm task. |
| `--time` | Slurm wall time. |
| `--partition` | Slurm partition. |
| `--account` | Slurm account. Empty string omits account. |
| `--edf-env` | CSCS EDF environment file. |
| `--train-python` | Python executable inside the EDF container. |
| `--job-prefix` | Slurm job-name prefix. |
| `--with-7b` | Enable the 7B teacher for training. |
| `-h`, `--help`, `help` | Show help. |

## Config

`configs/hparam_tuning.yaml` has three top-level parts:

| Section | Meaning |
|---|---|
| `profiles` | Named groups of variants used by `--profile`. |
| `defaults` | Base training config used for every generated run. |
| `variants` | Named deep overrides applied on top of `defaults`. |

Main default parameters:

| Parameter | Meaning |
|---|---|
| `seed` | Random seed. |
| `manifest.path` | Fallback manifest path if `--manifest` is not passed. |
| `manifest.val_path` | Optional validation manifest. |
| `manifest.root_remap` | Optional path remapping for manifest paths. |
| `curriculum.total_training_steps` | Total optimizer steps. |
| `curriculum.image_samples_per_epoch` | Image sampler epoch length. |
| `curriculum.video_samples_per_epoch` | Video sampler epoch length. |
| `loaders.image_batch_size` | Per-rank image batch size. |
| `loaders.video_batch_size` | Per-rank video batch size. |
| `loaders.num_workers` | DataLoader workers per rank. |
| `loaders.pin_memory` | Use pinned host memory. |
| `transforms.patch_size` | Patch size used by masks/backbones. |
| `transforms.image.n_global_crops` | Number of global image crops. |
| `transforms.image.n_local_crops` | Number of local image crops. |
| `transforms.image.max_global_crop_px` | Max global image crop size. |
| `transforms.image.mask_strategy` | Image masking mode. |
| `transforms.image.freq_mask.mask_ratio` | Image frequency-mask ratio. |
| `transforms.image.freq_mask.n_bands` | Number of frequency bands to mask. |
| `transforms.image.freq_mask.use_alp_bias` | Bias masks with ALP difficulty. |
| `transforms.image.spatial_mask_ratio` | Image spatial-mask ratio. |
| `transforms.video.n_frames` | Frames sampled per video clip. |
| `transforms.video.temporal_stride` | Frame sampling stride. |
| `transforms.video.tube_size` | Temporal length of each tube mask group. |
| `transforms.video.tube_mask_ratio` | Video tube-mask ratio. |
| `transforms.video.mask_strategy` | Video masking mode. |
| `transforms.video.max_crop_px` | Max video crop size. |
| `transforms.video.spatial_mask_ratio` | Video spatial-mask ratio. |
| `transforms.video.freq_mask.mask_ratio` | Video frequency-mask ratio. |
| `train.phase1_frac` | Cumulative end of image-only Phase 1. |
| `train.phase2_frac` | Cumulative end of video-only Phase 2. |
| `train.phase3_frac` | Cumulative end of paired Phase 3. |
| `train.base_lr` | Base learning rate. |
| `train.weight_decay` | AdamW weight decay. |
| `train.beta1`, `train.beta2` | AdamW beta values. |
| `train.grad_clip` | Gradient norm clipping threshold. |
| `train.warmup_steps_p1/p2/p3` | Warmup steps per phase. |
| `train.ema_momentum` | EMA teacher momentum. |
| `train.lam1`-`train.lam7` | Main loss weights. |
| `train.lam6_nce` | InfoNCE loss weight. |
| `train.lam6_nce_temp` | InfoNCE temperature. |
| `train.lam_7b` | 7B teacher loss weight. |
| `train.lam_gram` | Gram loss weight. |
| `train.lam_koleo` | Single-branch KoLeo weight. |
| `train.lam_koleo_cross` | Cross-branch KoLeo weight. |
| `train.use_koleo` | Enable single-branch KoLeo. |
| `train.gram_start_step` | First step where Gram loss can run. |
| `train.gram_refresh_interval` | Gram target refresh interval. |
| `train.res_*` | Image resolution curriculum. |
| `train.res_vid_*` | Video resolution curriculum. |
| `train.checkpoint_every` | Periodic checkpoint interval. |
| `train.log_every` | Metric logging interval. |
| `train.force_stage` | Force a training stage, or `null`. |
| `model.image_backbone` | Image backbone name. |
| `model.video_backbone` | Video backbone name. |
| `model.frozen_teacher` | Optional frozen teacher backbone. |
| `model.ema_momentum` | Model-side EMA momentum. |
| `model.n_prototypes` | Number of prototype vectors. |
| `model.align_dim` | Cross-modal projection dimension. |
| `model.dtype` | Training dtype. |
| `model.hf_cache_dir` | Explicit Hugging Face cache dir. |
| `model.use_gradient_checkpointing` | Enable activation checkpointing. |
| `model.trainable_image_layers` | `null` full tune, `0` frozen, `K` last K image layers. |
| `model.trainable_video_layers` | `null` full tune, `0` frozen, `K` last K video layers. |
| `anatomy_weights` | Sampling weights by anatomy family. |

Current variant groups:

| Profile | Purpose |
|---|---|
| `single` | Default baseline only. |
| `sentinel` | Small first-pass stability checks. |
| `optimizer` | LR, weight decay, clipping, EMA. |
| `masking` | Image/video mask difficulty and video length. |
| `loss` | Cross-modal and regularization loss weights. |
| `architecture` | Projection/prototype size. |
| `trainability` | Full, last-K, or frozen backbone tuning. |
| `all` | Every configured variant. |

## Examples

4-hour reduced-crop stability screen:

```bash
./submit_hparam_tuning.sh --submit \
  --variants nce_025,nce_temp_010,cross_koleo_005,no_cross_koleo \
  --run-prefix stability_screen_4h \
  --steps 3600 \
  --time 04:00:00 \
  --phase-split 0.15,0.25,0.60 \
  --reduced-image-crops \
  --num-workers 2
```

8-hour reduced-crop confirm run:

```bash
./submit_hparam_tuning.sh --submit \
  --variants nce_temp_010,cross_koleo_005 \
  --run-prefix stability_confirm_8h \
  --steps 10000 \
  --time 08:00:00 \
  --phase-split 0.15,0.25,0.60 \
  --reduced-image-crops \
  --num-workers 2
```
