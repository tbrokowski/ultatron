"""
finetune/experiments/camus.py  ·  CAMUS LV segmentation finetune
==========================================================

Task:    Segment left-ventricular myocardium from apical 2CH and 4CH views.
Dataset: CAMUS — 500 patients, ED+ES frames, .mhd format.
Head:    DPTSegHead (default) or LinearSegHead.
Loss:    BCE + Dice (combined), weighted equally.
Metric:  Dice (primary), IoU, Hausdorff-95 per view/phase.

Run:
    python -m finetune.experiments.camus \\
        --checkpoint checkpoints/phase3_end.pt \\
        --data-root  /capstor/store/cscs/swissai/a127/ultrasound/CAMUS \\
        --config     configs/finetune/camus.yaml \\
        --output-dir results/finetune/camus/

What's dataset-specific here vs generic
----------------------------------------
Dataset-specific (lives in this file):
  - CAMUS file layout parsing (.mhd + .zraw, patient dirs, view/phase naming)
  - head instantiation parameters: n_classes=1, binary BCE+Dice loss
  - Dice stratification by view (2CH/4CH) and phase (ED/ES)
  - viz: segmentation overlays for all 4 view/phase combinations

Generic (reused from models/heads/ without modification):
  - DPTSegHead, LinearSegHead — just instantiated with the right n_classes
  - The training loop in FinetuneExperiment.run()
  - BCE, Dice loss functions from models/losses/ (or torch.nn.functional)
"""
from __future__ import annotations

import logging          
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from finetune.base import FinetuneExperiment, FinetuneConfig
from models.heads import build_seg_head
from eval.metrics import dice_score, iou_score, hausdorff_95
from eval.benchmarks.camus import CAMUSBenchmark
from data.pipeline.transforms import to_canonical_tensor

log = logging.getLogger(__name__)


# ── CAMUS finetune dataset ────────────────────────────────────────────────────

class CAMUSFinetuneDataset(Dataset):
    """
    CAMUS dataset for supervised finetune.

    Returns per-sample dicts:
        image     : (3, 256, 256) float32 [0, 1]  RGB (greyscale repeated)
        mask      : (1, 256, 256) float32 binary
        sample_id : str
        view      : "2CH" | "4CH"
        phase     : "ED"  | "ES"
    """

    IMG_SIZE = 256

    def __init__(self, root: str, split: str = "train"):
        self.root    = Path(root)
        self.samples = self._collect(split)

    def _patients_dir(self) -> Path:
        """Handle both extracted layouts: database_nifti/ (.nii.gz) or root/ (.mhd)."""
        nifti_dir = self.root / "database_nifti"
        return nifti_dir if nifti_dir.exists() else self.root

    def _collect(self, split: str) -> list[dict]:
        pdir_root = self._patients_dir()
        patients  = sorted(pdir_root.glob("patient*/"))
        n         = len(patients)

        # Detect file extension
        ext    = ".nii.gz" if any(pdir_root.rglob("*.nii.gz")) else ".mhd"
        gt_sfx = f"_gt{ext}"

        split_map = {}
        for i, p in enumerate(patients):
            frac = i / max(n - 1, 1)
            if frac < 0.80:   split_map[p.name] = "train"
            elif frac < 0.90: split_map[p.name] = "val"
            else:              split_map[p.name] = "test"

        out = []
        for pdir in patients:
            if split_map.get(pdir.name) != split:
                continue
            pid = pdir.name
            for view in ("2CH", "4CH"):
                for phase in ("ED", "ES"):
                    img = pdir / f"{pid}_{view}_{phase}{ext}"
                    msk = pdir / f"{pid}_{view}_{phase}{gt_sfx}"
                    if img.exists() and msk.exists():
                        out.append({"img": str(img), "msk": str(msk),
                                    "id": f"{pid}_{view}_{phase}",
                                    "view": view, "phase": phase,
                                    "ext": ext})
        log.info(f"CAMUS {split}: {len(out)} samples (ext={ext})")
        return out

    def _load_volume(self, path: str) -> np.ndarray:
        """Load .mhd or .nii.gz via SimpleITK; return 2-D float32 slice."""
        import SimpleITK as sitk
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        # SimpleITK reads (Z, Y, X) for 3-D; take middle slice if volume
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        return arr

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s   = self.samples[idx]
        img = self._load_volume(s["img"])
        msk = self._load_volume(s["msk"])

        # Normalise image
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        # Resize to fixed size for batching
        sz  = self.IMG_SIZE
        img_t = F.interpolate(
            torch.from_numpy(img).unsqueeze(0).unsqueeze(0),
            size=(sz, sz), mode="bilinear", align_corners=False
        ).squeeze()                                          # (sz, sz)
        msk_t = F.interpolate(
            torch.from_numpy(msk).unsqueeze(0).unsqueeze(0),
            size=(sz, sz), mode="nearest"
        ).squeeze()                                          # (sz, sz)

        # Binary LV mask: any non-zero label = LV region
        mask_bin = (msk_t > 0).float().unsqueeze(0)         # (1, sz, sz)

        # Convert to canonical 3-channel RGB tensor
        image_rgb = to_canonical_tensor(img_t)              # (3, sz, sz)

        return {
            "image":     image_rgb,
            "mask":      mask_bin,
            "sample_id": s["id"],
            "view":      s["view"],
            "phase":     s["phase"],
        }


# ── CAMUS finetune experiment ─────────────────────────────────────────────────

class CAMUSFinetune(FinetuneExperiment):
    """
    CAMUS LV segmentation finetune experiment.

    Wires together:
      - CAMUSFinetuneDataset (dataset-specific loading)
      - DPTSegHead or LinearSegHead (generic head, correct params for binary seg)
      - BCE + Dice loss (standard for binary medical segmentation)
      - Per-view/phase Dice reporting
      - Segmentation viz on test set completion
    """

    EXPERIMENT_NAME = "camus_lv_segmentation"
    DATASET_ID      = "CAMUS"
    TASK            = "segmentation"
    BENCHMARK_CLS   = CAMUSBenchmark

    # ── Build head ─────────────────────────────────────────────────────────────
    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        """
        Instantiate a generic segmentation head with CAMUS-specific parameters.
        n_classes=1 because LV segmentation is binary (LV vs background).
        The head type (linear vs dpt) is controlled by cfg.head_type.
        """
        return build_seg_head(
            embed_dim  = embed_dim,
            n_classes  = 1,
            head_type  = cfg.head_type,
            patch_size = 16,
        )

    # ── Dataloader ─────────────────────────────────────────────────────────────
    def build_dataloader(self, split: str) -> DataLoader:
        ds = CAMUSFinetuneDataset(str(self.data_root), split)
        return DataLoader(
            ds,
            batch_size  = self.cfg.batch_size,
            shuffle     = (split == "train"),
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
            drop_last   = (split == "train"),
        )

    # ── Loss ───────────────────────────────────────────────────────────────────
    def compute_loss(
        self,
        batch:       dict,
        feats:       dict,
        head_output: torch.Tensor,    # (B, 1, ph, pw) logits
    ) -> torch.Tensor:
        """
        Combined BCE + Dice loss.
        Both terms equally weighted — standard for binary medical segmentation.
        """
        target = batch["mask"]                              # (B, 1, H, W)

        # Upsample predictions to target resolution
        pred = F.interpolate(head_output, size=target.shape[-2:],
                             mode="bilinear", align_corners=False)

        # BCE loss
        bce = F.binary_cross_entropy_with_logits(pred, target)

        # Soft Dice loss
        pred_sig = torch.sigmoid(pred)
        inter    = (pred_sig * target).sum(dim=(1, 2, 3))
        union    = pred_sig.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice     = 1.0 - (2.0 * inter + 1.0) / (union + 1.0)

        return bce + dice.mean()

    # ── Validation ─────────────────────────────────────────────────────────────
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

            feats    = self.img_branch.forward_teacher(batch["image"])
            logits   = self.head(feats["patch_tokens"])
            pred     = F.interpolate(logits, size=batch["mask"].shape[-2:],
                                     mode="bilinear", align_corners=False)

            loss = F.binary_cross_entropy_with_logits(pred, batch["mask"])
            total_loss += loss.item()
            n          += 1

            pred_bin   = (torch.sigmoid(pred) > 0.5).cpu().numpy()
            target_bin = (batch["mask"] > 0.5).cpu().numpy()

            for i, sid in enumerate(batch["sample_id"]):
                parts = sid.split("_") if isinstance(sid, str) else ["?", "?", "?"]
                per_sample.append({
                    "sample_id": sid,
                    "view":  batch.get("view",  ["?"] * len(batch["sample_id"]))[i],
                    "phase": batch.get("phase", ["?"] * len(batch["sample_id"]))[i],
                    "dice":  dice_score(pred_bin[i, 0], target_bin[i, 0]),
                    "iou":   iou_score(pred_bin[i, 0], target_bin[i, 0]),
                })

        dices = [s["dice"] for s in per_sample]

        def _mean_subset(view, phase):
            sub = [s["dice"] for s in per_sample
                   if s.get("view") == view and s.get("phase") == phase]
            return round(float(np.mean(sub)), 4) if sub else float("nan")

        return {
            "val_loss":    round(total_loss / max(n, 1), 4),
            "val_dice":    round(float(np.mean(dices)), 4),
            "val_iou":     round(float(np.mean([s["iou"] for s in per_sample])), 4),
            "dice_2ch_ed": _mean_subset("2CH", "ED"),
            "dice_2ch_es": _mean_subset("2CH", "ES"),
            "dice_4ch_ed": _mean_subset("4CH", "ED"),
            "dice_4ch_es": _mean_subset("4CH", "ES"),
        }

    # ── Visualisation ──────────────────────────────────────────────────────────
    def run_viz(self, results: dict, output_dir: Path) -> None:
        """
        After training: produce a segmentation grid and Dice histogram.
        Uses oura.viz.segmentation — no viz logic lives here.
        """
        try:
            from viz.segmentation import (
                plot_segmentation_grid, plot_dice_distribution
            )
            from viz.core import save_figure
        except ImportError:
            log.warning("viz module not available — skipping figures")
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
                pred   = F.interpolate(logits, size=(256, 256),
                                       mode="bilinear", align_corners=False)
                pred_np = (torch.sigmoid(pred) > 0.5).cpu().numpy()[:, 0]
                gt_np   = (batch["mask"] > 0.5).cpu().numpy()[:, 0]
                img_np  = (batch["image"].cpu().permute(0, 2, 3, 1).numpy() * 255
                           ).astype(np.uint8)

                for i in range(len(pred_np)):
                    images.append(img_np[i])
                    preds.append(pred_np[i])
                    gts.append(gt_np[i])
                    ids.append(batch["sample_id"][i] if isinstance(batch["sample_id"][i], str)
                               else str(i))

                if len(images) >= 24:
                    break

        fig1 = plot_segmentation_grid(images[:24], preds[:24], gts[:24],
                                       sample_ids=ids[:24],
                                       title="CAMUS LV Segmentation — Test Set")
        save_figure(fig1, output_dir / "camus_seg_grid.png")

        dices = [dice_score(preds[i].astype(float), gts[i].astype(float))
                 for i in range(len(preds))]
        fig2 = plot_dice_distribution(np.array(dices),
                                       title="CAMUS LV Dice Distribution")
        save_figure(fig2, output_dir / "camus_dice_hist.png")
        log.info(f"[CAMUS] Viz saved to {output_dir}")


if __name__ == "__main__":
    CAMUSFinetune.main()
