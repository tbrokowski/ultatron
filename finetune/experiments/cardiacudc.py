"""
finetune/experiments/cardiacudc.py  ·  CardiacUDC normal/disease classification
================================================================================

Task:    Binary classification — normal (0) vs cardiac disease (1).
Dataset: CardiacUDC — A4C NIfTI cine volumes, labelled subset from label_all_frame/.
Branch:  Video branch (NIfTI volumes treated as cine clips).
Head:    LinearClsHead on clip_cls token (2 classes).
Loss:    Binary cross-entropy.
Metric:  AUC-ROC, accuracy, F1.

Layout:
  {root}/cardiacUDC_dataset/
      label_all_frame/
          normal-{id}-4_image.nii.gz   → label 0
          patient-{id}-4_image.nii.gz  → label 1
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from finetune.base import FinetuneExperiment, FinetuneConfig
from models.heads import build_cls_head
from eval.metrics import binary_auc, binary_f1, binary_accuracy

log = logging.getLogger(__name__)
N_FRAMES  = 16
CLIP_SIZE = 112


def _load_nifti_clip(path: str, n_frames: int) -> torch.Tensor:
    """Load a NIfTI volume as a (T, 3, H, W) float tensor."""
    try:
        import SimpleITK as sitk
        arr = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        # SimpleITK: (Z, Y, X) — Z acts as temporal axis
        if arr.ndim == 2:
            arr = arr[None]
        elif arr.ndim == 4:
            arr = arr[..., 0]   # drop channel dim if present
        indices = np.linspace(0, len(arr) - 1, n_frames, dtype=int)
        frames  = arr[indices]
        pmin, pmax = frames.min(), frames.max()
        frames = (frames - pmin) / (pmax - pmin + 1e-8)
    except Exception:
        frames = np.zeros((n_frames, CLIP_SIZE, CLIP_SIZE), dtype=np.float32)

    clip = []
    for f in frames:
        t = torch.from_numpy(f).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, (CLIP_SIZE, CLIP_SIZE), mode="bilinear", align_corners=False)
        clip.append(t.squeeze(0).repeat(3, 1, 1))
    return torch.stack(clip)


class CardiacUDCDataset(Dataset):
    """
    CardiacUDC labelled subset (label_all_frame/).

    Returns:
        clip      : (T, 3, 112, 112) float32
        target    : int64  (0=normal, 1=disease)
        sample_id : str
    """

    def __init__(self, root: str, split: str, n_frames: int = N_FRAMES):
        self.n_frames = n_frames
        self.samples  = self._load(Path(root), split)

    def _load(self, root: Path, split: str) -> list[dict]:
        ds_dir    = root / "cardiacUDC_dataset" if (root / "cardiacUDC_dataset").exists() else root
        label_dir = ds_dir / "label_all_frame"
        if not label_dir.exists():
            log.warning(f"CardiacUDC label_all_frame/ not found at {label_dir}")
            return []

        all_files = sorted(label_dir.glob("*_image.nii.gz"))
        n = len(all_files)
        samples = []
        for i, fp in enumerate(all_files):
            frac = i / max(n - 1, 1)
            row_split = "train" if frac < 0.70 else ("val" if frac < 0.85 else "test")
            if row_split != split.lower():
                continue
            label = 0 if "normal" in fp.name.lower() else 1
            samples.append({"path": str(fp), "label": label,
                             "sample_id": fp.stem.replace("_image", "")})

        log.info(f"CardiacUDC {split}: {len(samples)} samples "
                 f"(normal={sum(s['label']==0 for s in samples)}, "
                 f"disease={sum(s['label']==1 for s in samples)})")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {"clip": _load_nifti_clip(s["path"], self.n_frames),
                "target": torch.tensor(s["label"], dtype=torch.long),
                "sample_id": s["sample_id"]}


class CardiacUDCFinetune(FinetuneExperiment):
    """CardiacUDC normal vs cardiac disease binary classification."""

    EXPERIMENT_NAME = "cardiacudc_disease_classification"
    DATASET_ID      = "CardiacUDC"
    TASK            = "classification"
    BENCHMARK_CLS   = None

    def build_head(self, embed_dim: int, cfg: FinetuneConfig):
        return build_cls_head(embed_dim, n_classes=2, head_type="mlp")

    def setup(self, img_branch, device="cuda", vid_branch=None):
        self.img_branch = img_branch
        self.vid_branch = vid_branch
        self.device     = device
        assert vid_branch is not None, "CardiacUDCFinetune requires vid_branch"
        if self.cfg.freeze_backbone:
            for p in vid_branch.parameters():
                p.requires_grad_(False)
            vid_branch.eval()
        embed_dim      = vid_branch.embed_dim
        backbone_dtype = next(vid_branch.parameters()).dtype
        self.head = self.build_head(embed_dim, self.cfg).to(device=device, dtype=backbone_dtype)
        log.info(f"[CardiacUDC] head={self.head}  dtype={backbone_dtype}")

    def build_dataloader(self, split: str) -> DataLoader:
        ds = CardiacUDCDataset(str(self.data_root), split, n_frames=N_FRAMES)
        return DataLoader(ds, batch_size=self.cfg.batch_size,
                          shuffle=(split == "train"),
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def compute_loss(self, batch, feats, head_output):
        return F.cross_entropy(head_output, batch["target"])

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        self.head.train(); self.vid_branch.teacher.eval()
        total_loss, n = 0.0, 0
        for batch in loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                 enabled=torch.cuda.is_available()):
                with torch.no_grad():
                    vid_out = self.vid_branch.forward_teacher(batch["clip"])
                logits = self.head(vid_out["clip_cls"])
                loss   = self.compute_loss(batch, {}, logits)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(self.head.parameters(), 1.0)
            scaler.step(optimiser); scaler.update()
            optimiser.zero_grad(set_to_none=True)
            total_loss += loss.item(); n += 1
        return total_loss / max(n, 1)

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval(); self.vid_branch.teacher.eval()
        all_probs, all_labels = [], []
        total_loss, n = 0.0, 0
        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            vid_out = self.vid_branch.forward_teacher(batch["clip"])
            logits  = self.head(vid_out["clip_cls"])
            loss    = self.compute_loss(batch, {}, logits)
            total_loss += loss.item(); n += 1
            probs = torch.softmax(logits.float(), dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(batch["target"].cpu().tolist())

        p, t = np.array(all_probs), np.array(all_labels)
        preds = (p >= 0.5).astype(int)
        return {"val_loss": round(total_loss / max(n, 1), 4),
                "val_auc":  round(binary_auc(p, t), 4),
                "val_acc":  round(binary_accuracy(preds, t), 4),
                "val_f1":   round(binary_f1(preds, t), 4)}

    def run_viz(self, results: dict, output_dir: Path) -> None:
        pass


if __name__ == "__main__":
    CardiacUDCFinetune.main()
