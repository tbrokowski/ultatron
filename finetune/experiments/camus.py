"""
finetune/experiments/camus.py  ·  CAMUS multi-structure segmentation finetune
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from finetune.base import FinetuneExperiment, FinetuneConfig
from finetune.datasets.camus_io import (
    aggregate_camus_metrics,
    collect_camus_frames,
    encode_lv_mask,
    get_camus_variant_spec,
    load_camus_slice,
    verify_camus_protocol,
)
from finetune.datasets.native_batch import (
    extract_native_region,
    finetune_img_size,
    finetune_native_collate,
    native_crop_shape,
    padding_mask_for_shape,
    uses_native_resolution,
)
from finetune.seg_common import build_seg_finetune_head, compute_binary_seg_loss
from models.heads import forward_seg_head
from eval.metrics import dice_per_class, dice_score, iou_score
from eval.benchmarks.camus import CAMUSBenchmark
from data.pipeline.transforms import to_canonical_tensor

log = logging.getLogger(__name__)


class CAMUSFinetuneDataset(Dataset):
    """CAMUS dataset for supervised finetune (binary or multiclass)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        lv_target: str = "lv_structures",
        quality_filter: bool = False,
        *,
        training_mode: str = "binary",
        img_size: int = 256,
        native: bool = False,
        native_max_px: int = 512,
    ):
        self.root = Path(root)
        self.split = split
        self.lv_target = lv_target
        self.quality_filter = quality_filter
        self.training_mode = training_mode
        self.img_size = img_size
        self.native = native
        self.native_max_px = native_max_px
        self.samples = collect_camus_frames(
            self.root, split, lv_target=lv_target, quality_filter=quality_filter,
        )
        if self.samples:
            patients = sorted({s.patient for s in self.samples})
            verify_camus_protocol(
                patients, lv_target=lv_target, quality_filter=quality_filter,
            )

    def _load_volume(self, path: str) -> np.ndarray:
        return load_camus_slice(path)

    def __len__(self):
        return len(self.samples)

    def _prepare_mask(self, label_np: np.ndarray) -> np.ndarray:
        if self.training_mode == "multiclass":
            return label_np.astype(np.int64)
        return encode_lv_mask(label_np, self.lv_target)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = self._load_volume(s.img)
        msk = self._load_volume(s.msk)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        label_np = msk.astype(np.int64)
        mask_arr = self._prepare_mask(label_np)

        if self.native:
            h0, w0 = img.shape
            th, tw = native_crop_shape(h0, w0, max_px=self.native_max_px)
            img_2d = extract_native_region(
                torch.from_numpy(img).unsqueeze(0).float(), th, tw,
            ).squeeze(0)
            if self.training_mode == "multiclass":
                mask_t = extract_native_region(
                    torch.from_numpy(mask_arr).unsqueeze(0).float(), th, tw,
                ).squeeze(0).long().unsqueeze(0)
            else:
                mask_t = extract_native_region(
                    torch.from_numpy(mask_arr).unsqueeze(0).float(), th, tw,
                )
            label_t = extract_native_region(
                torch.from_numpy(label_np).unsqueeze(0).float(), th, tw,
            ).long()
            image_rgb = to_canonical_tensor(img_2d)
            padding_mask = padding_mask_for_shape(th, tw, valid_h=th, valid_w=tw)
            return {
                "image": image_rgb,
                "mask": mask_t,
                "label_map": label_t,
                "padding_mask": padding_mask,
                "sample_id": s.sample_id,
                "view": s.view,
                "phase": s.phase,
            }

        sz = self.img_size
        img_t = F.interpolate(
            torch.from_numpy(img).unsqueeze(0).unsqueeze(0),
            size=(sz, sz), mode="bilinear", align_corners=False,
        ).squeeze()
        if self.training_mode == "multiclass":
            mask_t = F.interpolate(
                torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0).float(),
                size=(sz, sz), mode="nearest",
            ).squeeze(0).long().unsqueeze(0)
        else:
            mask_t = F.interpolate(
                torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0),
                size=(sz, sz), mode="nearest",
            ).squeeze(0)
        label_t = F.interpolate(
            torch.from_numpy(label_np).unsqueeze(0).unsqueeze(0).float(),
            size=(sz, sz), mode="nearest",
        ).squeeze(0).long()
        image_rgb = to_canonical_tensor(img_t)

        return {
            "image": image_rgb,
            "mask": mask_t,
            "label_map": label_t,
            "sample_id": s.sample_id,
            "view": s.view,
            "phase": s.phase,
        }


class CAMUSFinetune(FinetuneExperiment):
    EXPERIMENT_NAME = "camus_lv_segmentation"
    DATASET_ID = "CAMUS"
    TASK = "segmentation"
    BENCHMARK_CLS = CAMUSBenchmark

    def _variant_spec(self) -> dict:
        return get_camus_variant_spec(self.cfg.camus_variant)

    def _is_multiclass(self) -> bool:
        return self.cfg.camus_training_mode == "multiclass"

    def evaluate(self, split: str = "test") -> dict:
        assert self.head is not None, "Call setup() and run() first"
        self.head.eval()

        benchmark = self.BENCHMARK_CLS(
            encoder=self.encoder,
            img_branch=self.img_branch,
            head=self.head,
            device=self.device,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            lv_target=self.cfg.lv_target,
            quality_filter=self.cfg.camus_quality_filter,
            camus_variant=self.cfg.camus_variant,
            camus_training_mode=self.cfg.camus_training_mode,
            img_size=finetune_img_size(self.cfg, default=256),
            native=uses_native_resolution(self.cfg, self.encoder),
            native_max_px=self.cfg.native_max_px,
        )
        results = benchmark.run(str(self.data_root), split=split)
        results["experiment"] = self.EXPERIMENT_NAME
        results["camus_variant"] = self.cfg.camus_variant
        results["camus_training_mode"] = self.cfg.camus_training_mode
        results["lv_target"] = self.cfg.lv_target
        results["camus_quality_filter"] = self.cfg.camus_quality_filter
        self._save_results(results)
        self.run_viz(results, self.output_dir)
        return results

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        n_classes = 4 if cfg.camus_training_mode == "multiclass" else 1
        return build_seg_finetune_head(self.encoder, cfg, n_classes=n_classes)

    def build_dataloader(self, split: str) -> DataLoader:
        native = uses_native_resolution(self.cfg, self.encoder)
        ds = CAMUSFinetuneDataset(
            str(self.data_root),
            split,
            lv_target=self.cfg.lv_target,
            quality_filter=self.cfg.camus_quality_filter,
            training_mode=self.cfg.camus_training_mode,
            img_size=finetune_img_size(self.cfg, default=256),
            native=native,
            native_max_px=self.cfg.native_max_px,
        )
        collate = finetune_native_collate if native else None
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
            collate_fn=collate,
        )

    def _multiclass_dice_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        tgt = target.squeeze(1).long()
        losses = []
        for c in (1, 2, 3):
            pred_c = probs[:, c]
            tgt_c = (tgt == c).float()
            inter = (pred_c * tgt_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + tgt_c.sum(dim=(1, 2))
            losses.append(1.0 - (2.0 * inter + 1.0) / (union + 1.0))
        return torch.stack(losses, dim=1).mean()

    def compute_loss(self, batch: dict, feats: dict, head_output: torch.Tensor) -> torch.Tensor:
        target = batch["mask"]
        pred = F.interpolate(
            head_output, size=target.shape[-2:],
            mode="bilinear", align_corners=False,
        )
        if self._is_multiclass():
            tgt = target.squeeze(1).long()
            ce = F.cross_entropy(pred, tgt)
            return ce + self._multiclass_dice_loss(pred, target)

        return compute_binary_seg_loss(
            head_output, target, self.cfg, use_pos_weight=False,
        )

    def _binary_metric_row(self, p_pred, p_tgt, sid, view, phase) -> dict:
        spec = self._variant_spec()
        dice_key = spec["dice_key"]
        return {
            "sample_id": sid,
            "view": view,
            "phase": phase,
            dice_key: dice_score(p_pred, p_tgt),
            "iou": iou_score(p_pred, p_tgt),
        }

    def _multiclass_metric_row(self, pred_labels, tgt_labels, sid, view, phase) -> dict:
        per_cls = dice_per_class(pred_labels, tgt_labels)
        row = {
            "sample_id": sid,
            "view": view,
            "phase": phase,
            "iou": float("nan"),
        }
        for c, d in per_cls.items():
            row[f"dice_class_{c}"] = d
        return row

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.encoder.eval()

        per_sample = []
        total_loss = 0.0
        n = 0

        for batch in val_loader:
            batch = {
                k: v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            pmask = batch.get("padding_mask")
            if pmask is not None:
                pmask = pmask.to(self.device, non_blocking=True)
            feats = (
                self.encoder.encode_image(batch["image"], padding_mask=pmask)
                if pmask is not None
                else self.encoder.encode_image(batch["image"])
            )
            logits = forward_seg_head(self.head, feats, padding_mask=pmask)
            pred = F.interpolate(
                logits, size=batch["mask"].shape[-2:],
                mode="bilinear", align_corners=False,
            )
            total_loss += self.compute_loss(batch, feats, logits).item()
            n += 1

            views = batch.get("view", ["?"] * len(batch["sample_id"]))
            phases = batch.get("phase", ["?"] * len(batch["sample_id"]))

            if self._is_multiclass():
                pred_labels = pred.argmax(dim=1).cpu().numpy()
                tgt_labels = batch["mask"].squeeze(1).cpu().numpy()
                for i, sid in enumerate(batch["sample_id"]):
                    per_sample.append(self._multiclass_metric_row(
                        pred_labels[i], tgt_labels[i], sid, views[i], phases[i],
                    ))
            else:
                pred_bin = (torch.sigmoid(pred) > 0.5).cpu().numpy()
                target_bin = (batch["mask"] > 0.5).cpu().numpy()
                for i, sid in enumerate(batch["sample_id"]):
                    per_sample.append(self._binary_metric_row(
                        pred_bin[i, 0], target_bin[i, 0], sid, views[i], phases[i],
                    ))

        agg = aggregate_camus_metrics(per_sample, self.cfg.camus_variant)
        out = {
            "val_loss": round(total_loss / max(n, 1), 4),
            "val_dice": agg.get("dice_mean", float("nan")),
            **{k: v for k, v in agg.items() if k not in ("camus_variant", "camus_training_mode", "lv_target")},
        }
        return out


if __name__ == "__main__":
    CAMUSFinetune.main()
