"""
finetune/experiments/tn3k.py  ·  TN3K thyroid nodule segmentation finetune
====================================================================

Task:    Segment thyroid nodule boundaries from ultrasound images.
Dataset: TN3K — 3,493 images with expert boundary annotations.
Head:    DPTSegHead (default) — boundary-sensitive structures benefit from DPT.
Loss:    BCE + Dice.
Metric:  Dice, IoU, S-measure (structural similarity).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from finetune.base import FinetuneExperiment, FinetuneConfig
from models.heads import build_seg_head
from eval.metrics import dice_score, iou_score
from eval.benchmarks.tn3k import TN3KBenchmark

log = logging.getLogger(__name__)
IMG_SIZE = 224


class TN3KFinetuneDataset(Dataset):
    def __init__(self, root: str, split: str = "train", test_frac: float = 0.15):
        self.root    = Path(root)
        img_dir      = self.root / "image"
        lbl_dir      = self.root / "label"
        all_imgs     = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        n_test       = max(1, int(len(all_imgs) * test_frac))
        n_val        = max(1, int(len(all_imgs) * test_frac))
        if split == "test":
            subset = all_imgs[-n_test:]
        elif split == "val":
            subset = all_imgs[-(n_test + n_val):-n_test]
        else:
            subset = all_imgs[:-(n_test + n_val)]
        self.samples = [
            {"img": str(p), "lbl": str(lbl_dir / (p.stem + ".png")),
             "sample_id": p.stem}
            for p in subset if (lbl_dir / (p.stem + ".png")).exists()
        ]
        log.info(f"TN3K {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = np.array(Image.open(s["img"]).convert("RGB"), dtype=np.float32) / 255.0
        lbl = np.array(Image.open(s["lbl"]).convert("L"),   dtype=np.float32) / 255.0

        img_t = F.interpolate(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0),
                               size=(IMG_SIZE, IMG_SIZE), mode="bilinear",
                               align_corners=False).squeeze(0)
        lbl_t = F.interpolate(torch.from_numpy(lbl).unsqueeze(0).unsqueeze(0),
                               size=(IMG_SIZE, IMG_SIZE), mode="nearest").squeeze(0)
        return {"image": img_t, "mask": (lbl_t > 0.5).float(),
                "sample_id": s["sample_id"]}


class TN3KFinetune(FinetuneExperiment):
    EXPERIMENT_NAME = "tn3k_thyroid_segmentation"
    DATASET_ID      = "TN3K"
    TASK            = "segmentation"
    BENCHMARK_CLS   = TN3KBenchmark

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        return build_seg_head(embed_dim, n_classes=1, head_type=cfg.head_type)

    def build_dataloader(self, split: str) -> DataLoader:
        return DataLoader(TN3KFinetuneDataset(str(self.data_root), split),
                          batch_size=self.cfg.batch_size, shuffle=(split == "train"),
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def compute_loss(self, batch, feats, head_output) -> torch.Tensor:
        target = batch["mask"]
        pred   = F.interpolate(head_output, size=target.shape[-2:],
                               mode="bilinear", align_corners=False)
        bce   = F.binary_cross_entropy_with_logits(pred, target)
        p_s   = torch.sigmoid(pred)
        inter = (p_s * target).sum(dim=(1, 2, 3))
        denom = p_s.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice  = (1.0 - (2.0 * inter + 1.0) / (denom + 1.0)).mean()
        return bce + dice

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.img_branch.teacher.eval()
        per_sample = []
        total_loss = 0.0
        n          = 0

        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            feats   = self.img_branch.forward_teacher(batch["image"])
            logits  = self.head(feats["patch_tokens"])
            pred    = F.interpolate(logits, size=batch["mask"].shape[-2:],
                                    mode="bilinear", align_corners=False)
            loss    = self.compute_loss(batch, feats, logits)
            total_loss += loss.item(); n += 1

            pred_b = (torch.sigmoid(pred) > 0.5).cpu().numpy()
            gt_b   = (batch["mask"] > 0.5).cpu().numpy()
            for i in range(len(pred_b)):
                per_sample.append({
                    "dice": dice_score(pred_b[i, 0], gt_b[i, 0]),
                    "iou":  iou_score(pred_b[i, 0], gt_b[i, 0]),
                })

        return {
            "val_loss": round(total_loss / max(n, 1), 4),
            "val_dice": round(float(np.mean([s["dice"] for s in per_sample])), 4),
            "val_iou":  round(float(np.mean([s["iou"]  for s in per_sample])), 4),
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
                feats   = self.img_branch.forward_teacher(batch["image"])
                logits  = self.head(feats["patch_tokens"])
                pred    = F.interpolate(logits, size=(IMG_SIZE, IMG_SIZE),
                                        mode="bilinear", align_corners=False)
                pred_np = (torch.sigmoid(pred) > 0.5).cpu().numpy()[:, 0]
                gt_np   = (batch["mask"] > 0.5).cpu().numpy()[:, 0]
                img_np  = (batch["image"].cpu().permute(0, 2, 3, 1).numpy() * 255
                           ).astype(np.uint8)
                for i in range(len(pred_np)):
                    images.append(img_np[i]); preds.append(pred_np[i])
                    gts.append(gt_np[i]);     ids.append(batch["sample_id"][i])
                if len(images) >= 24: break
        fig = plot_segmentation_grid(images[:24], preds[:24], gts[:24],
                                      sample_ids=ids[:24],
                                      title="TN3K Thyroid Nodule Segmentation")
        save_figure(fig, output_dir / "tn3k_seg_grid.png")
        dices = [dice_score(preds[i].astype(float), gts[i].astype(float))
                 for i in range(len(preds))]
        fig2 = plot_dice_distribution(np.array(dices), title="TN3K Dice Distribution")
        save_figure(fig2, output_dir / "tn3k_dice_hist.png")


if __name__ == "__main__":
    TN3KFinetune.main()
