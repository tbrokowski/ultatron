"""
finetune/experiments/busi.py  ·  BUSI breast tumour segmentation finetune
====================================================================================

Frozen-backbone protocol (target ~0.84 tumour Dice on USFM split):
  - Encoder/backbone fully frozen; only DPT/UPerNet task head is trained
  - seg_only: tumour segmentation head only (no cls multitask)
  - tumor_only_train: benign + malignant images only
  - USFM 70/15/15 split via split_manifest.json
  - eval_tumor_only: primary metric dice_tumor_mean

Full backbone fine-tuning (USFM paper UPerNet @ 512px) is a separate path:
  scripts/run_busi_finetune.sh — not used here.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from finetune.base import FinetuneExperiment, FinetuneConfig
from finetune.datasets.busi_splits import (
    collect_busi_samples,
    default_repo_manifest_path,
    load_split_manifest,
    split_busi_samples,
)
from finetune.datasets.native_batch import (
    extract_native_region,
    finetune_img_size,
    finetune_native_collate,
    native_crop_shape,
    padding_mask_for_shape,
    uses_native_resolution,
)
from finetune.datasets.seg_augment import augment_breast_us
from models.heads import build_finetune_seg_head, forward_seg_head, build_cls_head
from finetune.seg_common import build_seg_finetune_head
from models.heads.seg_losses import binary_segmentation_loss
from eval.metrics import dice_score, iou_score, auc_roc
from eval.benchmarks.busi import BUSIBenchmark

log = logging.getLogger(__name__)

CLASS_NAMES = ["benign", "malignant", "normal"]


class BUSIFinetuneDataset(Dataset):
    """
    BUSI finetune dataset.

    Layout:
        {root}/benign/benign (1).png
        {root}/benign/benign (1)_mask.png
        ...

    Returns:
        image     : (3, 224, 224) float32 [0, 1]
        mask      : (1, 224, 224) float32 binary  (zeros for 'normal')
        cls_label : int  0=benign, 1=malignant, 2=normal
        sample_id : str
        cls_name  : str
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        *,
        test_frac: float = 0.20,
        val_frac: float = 0.20,
        split_manifest: str | Path | None = None,
        tumor_only_train: bool = False,
        img_size: int = 224,
        native: bool = False,
        native_max_px: int = 512,
    ):
        self.root = Path(root)
        self.split = split
        self.tumor_only_train = tumor_only_train
        self.img_size = img_size
        self.native = native
        self.native_max_px = native_max_px
        manifest = load_split_manifest(split_manifest) if split_manifest else None
        all_samples = collect_busi_samples(self.root)
        self.samples = split_busi_samples(
            all_samples,
            split,
            test_frac=test_frac,
            val_frac=val_frac,
            manifest=manifest,
            tumor_only_train=tumor_only_train,
        )

    def _collect(self, split: str, test_frac: float) -> list[dict]:
        """Legacy hook — samples built in __init__."""
        return self.samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = np.array(Image.open(s["img"]).convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1)        # (3, H, W)

        if self.native:
            _, h0, w0 = img_t.shape
            th, tw = native_crop_shape(h0, w0, max_px=self.native_max_px)
            img_t = extract_native_region(img_t, th, tw)
            if s["mask"]:
                m = np.array(Image.open(s["mask"]).convert("L"), dtype=np.float32) / 255.0
                m_t = torch.from_numpy(m).unsqueeze(0)
                mask = extract_native_region(m_t, th, tw)
                mask = (mask > 0.5).float()
            else:
                mask = torch.zeros(1, th, tw)
            if self.split == "train":
                img_t, mask = augment_breast_us(img_t, mask)
            padding_mask = padding_mask_for_shape(th, tw, valid_h=th, valid_w=tw)
            return {
                "image": img_t,
                "mask": mask,
                "padding_mask": padding_mask,
                "cls_label": torch.tensor(s["cls_label"], dtype=torch.long),
                "cls_name": s["cls_name"],
                "sample_id": s["sample_id"],
            }

        img_t = F.interpolate(img_t.unsqueeze(0), size=(self.img_size, self.img_size),
                               mode="bilinear", align_corners=False).squeeze(0)

        mask = torch.zeros(1, self.img_size, self.img_size)
        if s["mask"]:
            m   = np.array(Image.open(s["mask"]).convert("L"), dtype=np.float32) / 255.0
            m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
            m_t = F.interpolate(m_t, size=(self.img_size, self.img_size), mode="nearest").squeeze(0)
            mask = (m_t > 0.5).float()

        if self.split == "train":
            img_t, mask = augment_breast_us(img_t, mask)

        return {
            "image":     img_t,
            "mask":      mask,
            "cls_label": torch.tensor(s["cls_label"], dtype=torch.long),
            "cls_name":  s["cls_name"],
            "sample_id": s["sample_id"],
        }

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for balanced CrossEntropy."""
        counts = torch.zeros(len(CLASS_NAMES))
        for s in self.samples:
            counts[s["cls_label"]] += 1
        counts = counts.clamp(min=1)
        weights = counts.sum() / (len(CLASS_NAMES) * counts)
        return weights

    def sampling_weights(self) -> list[float]:
        """Oversample tumour images so most batches include segmentation signal."""
        counts = torch.zeros(len(CLASS_NAMES))
        for s in self.samples:
            counts[s["cls_label"]] += 1
        counts = counts.clamp(min=1)
        inv = counts.sum() / (len(CLASS_NAMES) * counts)
        return [float(inv[s["cls_label"]]) for s in self.samples]


class BUSIFinetune(FinetuneExperiment):
    """
    BUSI joint segmentation + classification finetune.

    Two heads:
      self.head  = DPTSegHead (binary tumour segmentation, uses patch_tokens)
      self.head2 = LinearClsHead (3-class classification, uses CLS token)

    Both are trained simultaneously with a weighted loss sum.
    """

    EXPERIMENT_NAME = "busi_seg"
    DATASET_ID      = "BUSI"
    TASK            = "segmentation"
    BENCHMARK_CLS   = BUSIBenchmark

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        """Primary head: binary tumour segmentation (UPerNet or DPT by encoder)."""
        return build_seg_finetune_head(self.encoder, cfg, n_classes=1)

    def setup(self, img_branch=None, device="cuda", vid_branch=None, encoder=None):
        super().setup(img_branch, device, vid_branch, encoder=encoder)
        assert self.cfg.freeze_backbone, (
            "[BUSI] freeze_backbone must be true — use scripts/run_busi_finetune.sh "
            "for full USFM backbone fine-tuning"
        )
        if self.cfg.seg_only:
            self.head2 = None
            log.info("[BUSI] seg_only=True — tumour segmentation head only (frozen backbone)")
            return
        # Infer dtype from the encoder
        try:
            backbone_dtype = next(
                p for mod in self.encoder._nn_modules()
                for p in mod.parameters()
            ).dtype
        except StopIteration:
            backbone_dtype = torch.bfloat16
        self.head2 = build_cls_head(
            self.encoder.embed_dim, n_classes=3, head_type="linear"
        ).to(device=device, dtype=backbone_dtype)
        log.info(f"[BUSI] Cls head: {self.head2} (dtype={backbone_dtype})")

    def _resolve_split_manifest(self) -> Path | None:
        if self.cfg.split_manifest:
            path = Path(self.cfg.split_manifest)
            if not path.is_absolute():
                repo = Path(__file__).resolve().parents[2]
                path = repo / path
            return path if path.is_file() else None
        default = default_repo_manifest_path()
        return default if default.is_file() else None

    def build_dataloader(self, split: str) -> DataLoader:
        manifest_path = self._resolve_split_manifest()
        native = uses_native_resolution(self.cfg, self.encoder)
        sz = finetune_img_size(self.cfg)
        ds = BUSIFinetuneDataset(
            str(self.data_root),
            split,
            split_manifest=manifest_path,
            tumor_only_train=self.cfg.tumor_only_train,
            img_size=sz,
            native=native,
            native_max_px=self.cfg.native_max_px,
        )
        if split == "train" and not hasattr(self, "_busi_resolution_logged"):
            self._busi_resolution_logged = True
            if native:
                log.info(
                    "[BUSI] Native resolution: patch-stride padding, max_px=%d",
                    self.cfg.native_max_px,
                )
            else:
                log.info("[BUSI] Fixed resize: %dx%d for all backbones", sz, sz)
        if split == "train" and not hasattr(self, "_cls_weights"):
            self._cls_weights = ds.class_weights().to(self.device)
            log.info(f"[BUSI] class weights: {self._cls_weights.tolist()}")
        collate = finetune_native_collate if native else None
        if split == "train":
            weights = ds.sampling_weights()
            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True,
            )
            return DataLoader(
                ds, batch_size=self.cfg.batch_size, sampler=sampler,
                num_workers=self.cfg.num_workers, pin_memory=True,
                collate_fn=collate, drop_last=True,
            )
        return DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.num_workers, pin_memory=True,
            collate_fn=collate,
        )

    def _tumor_seg_loss(self, pred_logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """BCE + soft Dice + boundary BCE with foreground reweighting for small tumours."""
        return binary_segmentation_loss(
            pred_logits,
            target_mask,
            boundary_weight=getattr(self.cfg, "boundary_loss_weight", 0.5),
            use_pos_weight=True,
        )

    def compute_loss(self, batch, feats, head_output) -> torch.Tensor:
        target_mask = batch["mask"]
        target_cls  = batch["cls_label"]

        pred_logits = F.interpolate(
            head_output,
            size=target_mask.shape[-2:],
            mode="bilinear", align_corners=False,
        )

        if self.cfg.seg_only and self.cfg.tumor_only_train:
            return self.cfg.seg_loss_weight * self._tumor_seg_loss(pred_logits, target_mask)

        has_tumour = target_cls < 2
        is_normal  = target_cls == 2
        seg_terms  = []

        if has_tumour.any():
            seg_terms.append(
                self._tumor_seg_loss(pred_logits[has_tumour], target_mask[has_tumour])
            )

        if is_normal.any() and not self.cfg.tumor_only_train:
            bg_bce = F.binary_cross_entropy_with_logits(
                pred_logits[is_normal], target_mask[is_normal],
            )
            seg_terms.append(bg_bce * self.cfg.bg_seg_loss_weight)

        seg_loss = sum(seg_terms) if seg_terms else target_mask.new_tensor(0.0)

        cls_w = self.cfg.cls_loss_weight
        if self.cfg.seg_only or getattr(self, "_current_epoch", 0) < self.cfg.seg_only_epochs:
            cls_w = 0.0

        if self.head2 is None or cls_w == 0.0:
            return self.cfg.seg_loss_weight * seg_loss

        cls_logits = self.head2(feats["cls"])
        weight = getattr(self, "_cls_weights", None)
        if weight is not None:
            weight = weight.to(device=cls_logits.device, dtype=cls_logits.dtype)
        cls_loss = F.cross_entropy(cls_logits, target_cls, weight=weight)

        return self.cfg.seg_loss_weight * seg_loss + cls_w * cls_loss

    def _save_head(self, name: str = "best_head.pt"):
        """Save seg head (and cls head when present) to Capstor checkpoint dir."""
        path = self._head_checkpoint_path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"head": self.head.state_dict()}
        if self.head2 is not None:
            payload["head2"] = self.head2.state_dict()
        torch.save(payload, path)
        log.info(f"[BUSI] Checkpoint → {path}")

    def load_head(self, path: str):
        """Load both seg head and cls head."""
        ckpt = torch.load(path, map_location=self.device)
        if isinstance(ckpt, dict) and "head" in ckpt:
            self.head.load_state_dict(ckpt["head"])
            if "head2" in ckpt and self.head2 is not None:
                self.head2.load_state_dict(ckpt["head2"])
        else:
            self.head.load_state_dict(ckpt)

    def _reload_best_heads(self):
        """Called after training to restore best checkpoint into both heads."""
        best_path = self._head_checkpoint_path("best_head.pt")
        if best_path.exists():
            self.load_head(str(best_path))

    def _ensure_train_optim(self, scaler: GradScaler):
        """Head-only AdamW + cosine schedule (backbone stays frozen)."""
        if hasattr(self, "_busi_optim"):
            return self._busi_optim, self._busi_scheduler, scaler
        params = list(self._iter_trainable_params())
        n_params = sum(p.numel() for p in params)
        log.info("[BUSI] Optimizer: AdamW on %d head parameters (%.2fM)", len(params), n_params / 1e6)
        self._busi_optim = torch.optim.AdamW(
            params, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
        )
        self._busi_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._busi_optim,
            T_max=self.cfg.max_epochs,
            eta_min=self.cfg.lr * 0.01,
        )
        return self._busi_optim, self._busi_scheduler, scaler

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        """Train segmentation head only; backbone features are no-grad."""
        optim, scheduler, scaler = self._ensure_train_optim(scaler)

        self.head.train()
        if self.head2 is not None:
            self.head2.train()
        self.encoder.eval()

        total_loss = 0.0
        n          = 0
        clip_params = list(self._iter_trainable_params())

        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=torch.cuda.is_available()):
                pmask = batch.get("padding_mask")
                if pmask is not None:
                    feats = self.encoder.encode_image(batch["image"], padding_mask=pmask)
                else:
                    feats = self.encoder.encode_image(batch["image"])
                head_out = forward_seg_head(self.head, feats, padding_mask=pmask)
                loss     = self.compute_loss(batch, feats, head_out)

            self._backward_step_with_scaler(loss, optim, scaler, clip_params)
            optim.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n          += 1

        scheduler.step()
        return total_loss / max(n, 1)

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        if self.head2 is not None:
            self.head2.eval()
        self.encoder.eval()

        per_sample   = []
        cls_correct  = 0
        cls_total    = 0
        total_loss   = 0.0
        n            = 0

        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pmask = batch.get("padding_mask")
            if pmask is not None:
                feats = self.encoder.encode_image(batch["image"], padding_mask=pmask)
            else:
                feats = self.encoder.encode_image(batch["image"])
            logits   = forward_seg_head(self.head, feats, padding_mask=pmask)
            cls_out  = self.head2(feats["cls"]) if self.head2 is not None else None

            loss = self.compute_loss(batch, feats, logits)
            total_loss += loss.item()
            n          += 1

            if cls_out is not None:
                preds_cls = cls_out.argmax(-1)
                cls_correct += (preds_cls == batch["cls_label"]).sum().item()
                cls_total   += len(batch["cls_label"])

            # Segmentation Dice
            pred_seg = (torch.sigmoid(
                F.interpolate(logits, size=batch["mask"].shape[-2:],
                              mode="bilinear", align_corners=False)
            ) > 0.5).cpu().numpy()
            gt_seg   = (batch["mask"] > 0.5).cpu().numpy()

            for i, sid in enumerate(batch["sample_id"]):
                per_sample.append({
                    "sample_id": sid,
                    "cls_name":  batch.get("cls_name", ["?"] * len(batch["sample_id"]))[i],
                    "dice":      dice_score(pred_seg[i, 0], gt_seg[i, 0]),
                })

        dices_b = [s["dice"] for s in per_sample if s.get("cls_name") == "benign"]
        dices_m = [s["dice"] for s in per_sample if s.get("cls_name") == "malignant"]
        dices_tumor = dices_b + dices_m
        dice_all = [s["dice"] for s in per_sample]
        primary_dice = dices_tumor if self.cfg.eval_tumor_only else dice_all

        out = {
            "val_loss":       round(total_loss / max(n, 1), 4),
            "val_dice":       round(float(np.mean(primary_dice)), 4) if primary_dice else float("nan"),
            "dice_tumor_mean": round(float(np.mean(dices_tumor)), 4) if dices_tumor else float("nan"),
            "dice_benign":    round(float(np.mean(dices_b)), 4) if dices_b else float("nan"),
            "dice_malignant": round(float(np.mean(dices_m)), 4) if dices_m else float("nan"),
        }
        if cls_total:
            out["cls_accuracy"] = round(cls_correct / cls_total, 4)
        return out

    def evaluate(self, split: str = "test") -> dict:
        """Evaluate with USFM split manifest and tumour-only metrics when configured."""
        assert self.head is not None, "Call setup() and run() first"
        self.head.eval()
        if self.head2 is not None:
            self.head2.eval()

        manifest = self._resolve_split_manifest()
        benchmark = self.BENCHMARK_CLS(
            encoder=self.encoder,
            img_branch=self.img_branch,
            head=self.head,
            head2=self.head2,
            device=self.device,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            split_manifest=str(manifest) if manifest else None,
            eval_tumor_only=self.cfg.eval_tumor_only,
            img_size=finetune_img_size(self.cfg),
            native=uses_native_resolution(self.cfg, self.encoder),
            native_max_px=self.cfg.native_max_px,
        )
        results = benchmark.run(str(self.data_root), split=split)
        results["experiment"] = self.EXPERIMENT_NAME
        results["eval_tumor_only"] = self.cfg.eval_tumor_only
        results["tumor_only_train"] = self.cfg.tumor_only_train
        results["seg_only"] = self.cfg.seg_only
        if manifest:
            results["split_manifest"] = str(manifest)
        self._save_results(results)
        self.run_viz(results, self.output_dir)
        return results


class BUSIMultitaskFinetune(BUSIFinetune):
    """BUSI frozen-backbone finetune with separate seg + cls heads (all 3 classes)."""

    EXPERIMENT_NAME = "busi_multitask"


if __name__ == "__main__":
    BUSIFinetune.main()
