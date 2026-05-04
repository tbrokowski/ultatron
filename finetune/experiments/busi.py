"""
finetune/experiments/busi.py  ·  BUSI breast tumour segmentation + classification finetune
====================================================================================

Task:    Segment breast tumour regions AND classify benign vs malignant.
Dataset: BUSI — 780 images, 3 classes (benign, malignant, normal).
Heads:   DPTSegHead (segmentation) + LinearClsHead (3-class classification).
Loss:    seg: BCE+Dice  |  cls: CrossEntropy  (both active, weighted sum)
Metric:  Dice per class (benign/malignant), 3-class accuracy, AUC.

Two-head note
-------------
BUSI uniquely benefits from training segmentation and classification
jointly because tumour morphology predicts malignancy.  The base class
FinetuneExperiment only manages one head directly — we override _train_epoch
to handle both heads.

Head parameters
---------------
  seg_head: DPTSegHead(embed_dim, n_classes=1)         — tumour vs background
  cls_head: LinearClsHead(embed_dim, n_classes=3)      — benign/malignant/normal

The seg head uses patch_tokens; the cls head uses the CLS token.
No changes to the generic head classes needed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from finetune.base import FinetuneExperiment, FinetuneConfig
from models.heads import build_seg_head, build_cls_head
from eval.metrics import dice_score, iou_score, auc_roc
from eval.benchmarks.busi import BUSIBenchmark

log = logging.getLogger(__name__)

CLASS_NAMES = ["benign", "malignant", "normal"]
IMG_SIZE    = 224


def _augment(img: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply consistent spatial + colour augmentations to image+mask pairs.
    All transforms that change geometry are applied identically to both.
    """
    # Random horizontal flip
    if random.random() < 0.5:
        img  = TF.hflip(img)
        mask = TF.hflip(mask)

    # Random vertical flip
    if random.random() < 0.3:
        img  = TF.vflip(img)
        mask = TF.vflip(mask)

    # Random rotation ±15°
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        img  = TF.rotate(img,  angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)

    # Random brightness / contrast (image only)
    if random.random() < 0.5:
        img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
    if random.random() < 0.5:
        img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))

    return img, mask


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

    def __init__(self, root: str, split: str = "train", test_frac: float = 0.20):
        self.root    = Path(root)
        self.split   = split
        self.samples = self._collect(split, test_frac)

    def _collect(self, split: str, test_frac: float) -> list[dict]:
        samples = []
        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                continue
            imgs = sorted(p for p in cls_dir.glob("*.png") if "_mask" not in p.name)
            n_test = max(1, int(len(imgs) * test_frac))
            n_val  = max(1, int(len(imgs) * test_frac))
            if split == "test":
                subset = imgs[-n_test:]
            elif split == "val":
                subset = imgs[-(n_test + n_val):-n_test]
            else:
                subset = imgs[:-(n_test + n_val)]

            for img_path in subset:
                mask_path = img_path.parent / img_path.name.replace(".png", "_mask.png")
                samples.append({
                    "img":       str(img_path),
                    "mask":      str(mask_path) if mask_path.exists() else None,
                    "cls_label": cls_idx,
                    "cls_name":  cls_name,
                    "sample_id": f"{cls_name}_{img_path.stem}",
                })
        log.info(f"BUSI {split}: {len(samples)} samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = np.array(Image.open(s["img"]).convert("RGB"), dtype=np.float32) / 255.0
        img_t = torch.from_numpy(img).permute(2, 0, 1)        # (3, H, W)
        img_t = F.interpolate(img_t.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE),
                               mode="bilinear", align_corners=False).squeeze(0)

        mask = torch.zeros(1, IMG_SIZE, IMG_SIZE)
        if s["mask"]:
            m   = np.array(Image.open(s["mask"]).convert("L"), dtype=np.float32) / 255.0
            m_t = torch.from_numpy(m).unsqueeze(0).unsqueeze(0)
            m_t = F.interpolate(m_t, size=(IMG_SIZE, IMG_SIZE), mode="nearest").squeeze(0)
            mask = (m_t > 0.5).float()

        if self.split == "train":
            img_t, mask = _augment(img_t, mask)

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


class BUSIFinetune(FinetuneExperiment):
    """
    BUSI joint segmentation + classification finetune.

    Two heads:
      self.head  = DPTSegHead (binary tumour segmentation, uses patch_tokens)
      self.head2 = LinearClsHead (3-class classification, uses CLS token)

    Both are trained simultaneously with a weighted loss sum.
    """

    EXPERIMENT_NAME = "busi_seg_cls"
    DATASET_ID      = "BUSI"
    TASK            = "segmentation+classification"
    BENCHMARK_CLS   = BUSIBenchmark

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        """Primary head: binary tumour segmentation."""
        return build_seg_head(embed_dim, n_classes=1, head_type=cfg.head_type)

    def setup(self, img_branch, device="cuda", vid_branch=None):
        super().setup(img_branch, device, vid_branch)
        backbone_dtype = next(img_branch.parameters()).dtype
        self.head2 = build_cls_head(
            img_branch.embed_dim, n_classes=3, head_type="linear"
        ).to(device=device, dtype=backbone_dtype)
        log.info(f"[BUSI] Cls head: {self.head2} (dtype={backbone_dtype})")

    def build_dataloader(self, split: str) -> DataLoader:
        ds = BUSIFinetuneDataset(str(self.data_root), split)
        if split == "train" and not hasattr(self, "_cls_weights"):
            self._cls_weights = ds.class_weights().to(self.device)
            log.info(f"[BUSI] class weights: {self._cls_weights.tolist()}")
        return DataLoader(
            ds, batch_size=self.cfg.batch_size, shuffle=(split == "train"),
            num_workers=self.cfg.num_workers, pin_memory=True,
        )

    def compute_loss(self, batch, feats, head_output) -> torch.Tensor:
        target_mask = batch["mask"]                           # (B, 1, H, W)
        target_cls  = batch["cls_label"]                      # (B,)

        # Segmentation loss — skip for 'normal' samples (no tumour)
        has_tumour = (target_cls < 2)                         # benign=0, malignant=1
        seg_loss   = target_mask.new_tensor(0.0)
        if has_tumour.any():
            pred_seg = F.interpolate(head_output[has_tumour],
                                     size=target_mask.shape[-2:],
                                     mode="bilinear", align_corners=False)
            bce  = F.binary_cross_entropy_with_logits(pred_seg, target_mask[has_tumour])
            p_s  = torch.sigmoid(pred_seg)
            inter= (p_s * target_mask[has_tumour]).sum(dim=(1, 2, 3))
            denom= p_s.sum(dim=(1, 2, 3)) + target_mask[has_tumour].sum(dim=(1, 2, 3))
            seg_loss = bce + (1.0 - (2 * inter + 1) / (denom + 1)).mean()

        # Classification loss — class-weighted to counter benign-heavy imbalance
        cls_logits = self.head2(feats["cls"])
        weight = getattr(self, "_cls_weights", None)
        if weight is not None:
            weight = weight.to(device=cls_logits.device, dtype=cls_logits.dtype)
        cls_loss = F.cross_entropy(cls_logits, target_cls, weight=weight)

        # Weighted sum: equal weight by default
        return seg_loss + cls_loss

    def _save_head(self, name: str = "best_head.pt"):
        """Save both seg head and cls head together."""
        path = self.output_dir / name
        torch.save({
            "head":  self.head.state_dict(),
            "head2": self.head2.state_dict(),
        }, path)

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
        best_path = self.output_dir / "best_head.pt"
        if best_path.exists():
            self.load_head(str(best_path))

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        """Override to include head2 in the optimiser with cosine LR schedule."""
        # Rebuild optimiser to include both heads on first call
        if not hasattr(self, "_multi_optim"):
            self._multi_optim = torch.optim.AdamW(
                list(self.head.parameters()) + list(self.head2.parameters()),
                lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
            )
            self._multi_scaler = scaler
            self._multi_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._multi_optim,
                T_max=self.cfg.max_epochs,
                eta_min=self.cfg.lr * 0.01,
            )

        self.head.train()
        self.head2.train()
        total_loss = 0.0
        n          = 0

        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            with torch.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=torch.cuda.is_available()):
                with torch.no_grad():
                    feats = self.img_branch.forward_teacher(batch["image"])
                head_out = self.head(feats["patch_tokens"])
                loss     = self.compute_loss(batch, feats, head_out)

            self._backward_step_with_scaler(
                loss,
                self._multi_optim,
                self._multi_scaler,
                list(self.head.parameters()) + list(self.head2.parameters()),
            )
            self._multi_optim.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n          += 1

        self._multi_scheduler.step()
        return total_loss / max(n, 1)

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.head2.eval()
        self.img_branch.teacher.eval()

        per_sample   = []
        cls_correct  = 0
        cls_total    = 0
        total_loss   = 0.0
        n            = 0

        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            feats    = self.img_branch.forward_teacher(batch["image"])
            logits   = self.head(feats["patch_tokens"])
            cls_out  = self.head2(feats["cls"])

            loss = self.compute_loss(batch, feats, logits)
            total_loss += loss.item()
            n          += 1

            # Classification accuracy
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

        return {
            "val_loss":       round(total_loss / max(n, 1), 4),
            "val_dice":       round(float(np.mean([s["dice"] for s in per_sample])), 4),
            "dice_benign":    round(float(np.mean(dices_b)), 4) if dices_b else float("nan"),
            "dice_malignant": round(float(np.mean(dices_m)), 4) if dices_m else float("nan"),
            "cls_accuracy":   round(cls_correct / max(cls_total, 1), 4),
        }

    def run_viz(self, results: dict, output_dir: Path) -> None:
        try:
            from viz.segmentation import plot_segmentation_grid, plot_dice_distribution
            from viz.core import save_figure
        except ImportError:
            return

        test_loader = self.build_dataloader("test")
        images, preds, gts, ids = [], [], [], []

        self.head.eval()
        self.img_branch.teacher.eval()
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                feats  = self.img_branch.forward_teacher(batch["image"])
                logits = self.head(feats["patch_tokens"])
                pred   = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE),
                                       mode="bilinear", align_corners=False)
                pred_np = (torch.sigmoid(pred) > 0.5).cpu().numpy()[:, 0]
                gt_np   = (batch["mask"] > 0.5).cpu().numpy()[:, 0]
                img_np  = (batch["image"].cpu().permute(0, 2, 3, 1).numpy() * 255
                           ).astype(np.uint8)
                for i in range(len(pred_np)):
                    images.append(img_np[i]); preds.append(pred_np[i])
                    gts.append(gt_np[i]);     ids.append(batch["sample_id"][i])
                if len(images) >= 24:
                    break

        fig1 = plot_segmentation_grid(images[:24], preds[:24], gts[:24],
                                       sample_ids=ids[:24],
                                       title="BUSI Tumour Segmentation — Test Set")
        save_figure(fig1, output_dir / "busi_seg_grid.png")

        dices = [dice_score(preds[i].astype(float), gts[i].astype(float))
                 for i in range(len(preds))]
        fig2 = plot_dice_distribution(np.array(dices), title="BUSI Dice Distribution")
        save_figure(fig2, output_dir / "busi_dice_hist.png")


if __name__ == "__main__":
    BUSIFinetune.main()
