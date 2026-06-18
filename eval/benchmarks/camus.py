"""
eval/benchmarks/camus.py  ·  CAMUS LV segmentation benchmark
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eval.benchmarks.base import BaseBenchmark
from eval.metrics import dice_per_class, dice_score, hausdorff_95, iou_score
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
    finetune_native_collate,
    native_crop_shape,
    padding_mask_for_shape,
)
from data.pipeline.transforms import to_canonical_tensor

log = logging.getLogger(__name__)


class CAMUSBenchmarkDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "test",
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

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = load_camus_slice(s.img)
        msk = load_camus_slice(s.msk)

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        label_np = msk.astype(np.int64)
        if self.training_mode == "multiclass":
            mask_arr = label_np
        else:
            mask_arr = encode_lv_mask(msk, self.lv_target)

        if self.native:
            h0, w0 = img.shape
            th, tw = native_crop_shape(h0, w0, max_px=self.native_max_px)
            img_2d = extract_native_region(
                torch.from_numpy(img).unsqueeze(0).float(), th, tw,
            ).squeeze(0)
            if self.training_mode == "multiclass":
                msk_t = extract_native_region(
                    torch.from_numpy(mask_arr).unsqueeze(0).float(), th, tw,
                ).squeeze(0).long().unsqueeze(0)
            else:
                msk_t = extract_native_region(
                    torch.from_numpy(mask_arr).unsqueeze(0).float(), th, tw,
                )
            label_t = extract_native_region(
                torch.from_numpy(label_np).unsqueeze(0).float(), th, tw,
            ).long()
            img_rgb = to_canonical_tensor(img_2d)
            padding_mask = padding_mask_for_shape(th, tw, valid_h=th, valid_w=tw)
            return {
                "image": img_rgb,
                "mask": msk_t,
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
        ).squeeze(0)
        if self.training_mode == "multiclass":
            msk_t = F.interpolate(
                torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0).float(),
                size=(sz, sz), mode="nearest",
            ).squeeze(0).long().unsqueeze(0)
        else:
            msk_t = F.interpolate(
                torch.from_numpy(mask_arr).unsqueeze(0).unsqueeze(0),
                size=(sz, sz), mode="nearest",
            ).squeeze(0)
        label_t = F.interpolate(
            torch.from_numpy(label_np).unsqueeze(0).unsqueeze(0).float(),
            size=(sz, sz), mode="nearest",
        ).squeeze(0).long()
        img_rgb = to_canonical_tensor(img_t.squeeze(0))

        return {
            "image": img_rgb,
            "mask": msk_t,
            "label_map": label_t,
            "sample_id": s.sample_id,
            "view": s.view,
            "phase": s.phase,
        }


class CAMUSBenchmark(BaseBenchmark):
    BENCHMARK_NAME = "camus_segmentation"
    DATASET_ID = "CAMUS"
    ANATOMY_FAMILY = "cardiac"
    TASK = "segmentation"

    def __init__(
        self,
        img_branch=None,
        head: Optional[nn.Module] = None,
        device: str = "cuda",
        batch_size: int = 16,
        num_workers: int = 4,
        encoder=None,
        lv_target: str = "lv_structures",
        quality_filter: bool = False,
        camus_variant: str = "lv_epi",
        camus_training_mode: str = "binary",
        img_size: int = 256,
        native: bool = False,
        native_max_px: int = 512,
        **kwargs,
    ):
        super().__init__(
            img_branch=img_branch,
            head=head,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            encoder=encoder,
            **kwargs,
        )
        self.lv_target = lv_target
        self.quality_filter = quality_filter
        self.camus_variant = camus_variant
        self.camus_training_mode = camus_training_mode
        self.img_size = img_size
        self.native = native
        self.native_max_px = native_max_px
        self._spec = get_camus_variant_spec(camus_variant)

    def build_dataloader(self, root: str, split: str = "test") -> DataLoader:
        dataset = CAMUSBenchmarkDataset(
            root,
            split,
            lv_target=self.lv_target,
            quality_filter=self.quality_filter,
            training_mode=self.camus_training_mode,
            img_size=self.img_size,
            native=self.native,
            native_max_px=self.native_max_px,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=finetune_native_collate if self.native else None,
        )

    def predict(self, batch: dict) -> dict:
        images = batch["image"]
        pmask = batch.get("padding_mask")
        feats = self._extract_features(images, padding_mask=pmask)
        logits = self._predict_seg_logits(feats, padding_mask=pmask)
        pred = F.interpolate(
            logits, size=images.shape[-2:], mode="bilinear", align_corners=False,
        )
        if self.camus_training_mode == "multiclass":
            return {"pred": pred, "pred_labels": pred.argmax(dim=1)}
        return {"pred": torch.sigmoid(pred)}

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_ids: list,
        *,
        label_maps: torch.Tensor | None = None,
        views: list | None = None,
        phases: list | None = None,
        pred_labels: torch.Tensor | None = None,
    ) -> list[dict]:
        target_np = target.cpu().numpy()
        results = []

        for i, sid in enumerate(sample_ids):
            view = views[i] if views else "?"
            phase = phases[i] if phases else "?"
            row = {"sample_id": sid, "view": view, "phase": phase}

            if self.camus_training_mode == "multiclass":
                pl = pred_labels[i].cpu().numpy()
                tl = target_np[i, 0]
                per_cls = dice_per_class(pl, tl)
                for c, d in per_cls.items():
                    row[f"dice_class_{c}"] = d
            else:
                p = pred[i, 0].cpu().float().numpy()
                t = target_np[i, 0]
                dice_key = self._spec["dice_key"]
                row[dice_key] = dice_score(p, t)
                row["iou"] = iou_score(p, t)
                row["hd95"] = hausdorff_95(p >= 0.5, t >= 0.5)
            results.append(row)
        return results

    def aggregate(self, per_sample: list[dict]) -> dict:
        return aggregate_camus_metrics(per_sample, self.camus_variant)

    def run(self, root: str, split: str = "test") -> dict:
        t0 = time.time()
        log.info(
            "[%s] Running variant=%s mode=%s split=%s ...",
            self.BENCHMARK_NAME, self.camus_variant, self.camus_training_mode, split,
        )

        loader = self.build_dataloader(root, split)
        per_sample = []

        if self.encoder is not None:
            self.encoder.eval()
        elif self.img_branch is not None:
            mod = self.img_branch.teacher if hasattr(self.img_branch, "teacher") else self.img_branch
            mod.eval()
        if self.head is not None:
            self.head.eval()

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                outputs = self.predict(batch)
                pred = outputs["pred"]
                metrics = self.compute_metrics(
                    pred,
                    batch["mask"],
                    batch.get("sample_id", [""] * pred.shape[0]),
                    label_maps=batch.get("label_map"),
                    views=batch.get("view"),
                    phases=batch.get("phase"),
                    pred_labels=outputs.get("pred_labels"),
                )
                per_sample.extend(metrics)

        results = self.aggregate(per_sample)
        results.update({
            "benchmark": self.BENCHMARK_NAME,
            "dataset_id": self.DATASET_ID,
            "task": self.TASK,
            "split": split,
            "camus_variant": self.camus_variant,
            "camus_training_mode": self.camus_training_mode,
            "lv_target": self.lv_target,
            "camus_quality_filter": self.quality_filter,
            "n_samples": len(per_sample),
            "elapsed_sec": round(time.time() - t0, 1),
        })
        spec = self._spec
        if spec["mode"] == "binary":
            prefix = spec["metric_prefix"]
            log.info(
                "[%s] Done. variant=%s %s_mean=%s %s_ed=%s %s_es=%s",
                self.BENCHMARK_NAME,
                self.camus_variant,
                prefix, results.get(f"{prefix}_mean"),
                prefix, results.get(f"{prefix}_ed"),
                prefix, results.get(f"{prefix}_es"),
            )
        else:
            log.info(
                "[%s] Done. variant=multiclass macro_fg=%s class1=%s class2=%s class3=%s",
                self.BENCHMARK_NAME,
                results.get("dice_macro_fg_mean"),
                results.get("dice_class_1_mean"),
                results.get("dice_class_2_mean"),
                results.get("dice_class_3_mean"),
            )
        return results
