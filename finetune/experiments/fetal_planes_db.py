"""
finetune/experiments/fetal_planes_db.py  ·  FETAL_PLANES_DB plane classification
================================================================================

Task:    6-class fetal standard-plane classification.
Dataset: FETAL_PLANES_DB — 12,400 2-D ultrasound frames, 1,792 patients.
         Official train/test split from the Train column; val carved from train
         patients (10%) to avoid temporal leakage.
Branch:  Image branch.
Head:    Linear or MLP classifier on CLS token.
Loss:    Cross-entropy.
Metrics: Accuracy, macro F1, macro AUC (OvR).

Reference: https://zenodo.org/record/3904280
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from data.adapters.maternal_fetal.fetal_planes_db import FetalPlanesDBAdapter, PLANE_LABELS
from data.schema.manifest import USManifestEntry
from eval.metrics import auc_roc, binary_accuracy
from finetune.base import FinetuneConfig, FinetuneExperiment
from models.heads import build_cls_head

log = logging.getLogger(__name__)

N_CLASSES = len(PLANE_LABELS)
VAL_PATIENT_FRAC = 0.10


def _val_patient_ids(train_patients: List[str], frac: float = VAL_PATIENT_FRAC) -> Set[str]:
    """Deterministic patient-level val holdout from official train patients."""
    patients = sorted(set(train_patients))
    n_val = max(1, int(len(patients) * frac))
    return set(patients[:n_val])


def _resolve_split(entry: USManifestEntry, val_patients: Set[str]) -> str:
    if entry.split == "test":
        return "test"
    if entry.study_id in val_patients:
        return "val"
    return "train"


def _load_rgb_tensor(path: str, size: int, augment: bool) -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    if augment and torch.rand(()) < 0.5:
        tensor = torch.flip(tensor, dims=[-1])
    tensor = F.interpolate(
        tensor,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return tensor


class FetalPlanesDBDataset(Dataset):
    """Image-level fetal plane classification samples."""

    def __init__(
        self,
        root: str,
        split: str,
        img_size: int = 224,
        val_patient_frac: float = VAL_PATIENT_FRAC,
    ):
        self.split = split
        self.img_size = img_size
        self.samples = self._load_samples(Path(root), split, val_patient_frac)
        log.info(
            "FETAL_PLANES_DB %s: %d samples",
            split,
            len(self.samples),
        )

    def _load_samples(
        self,
        root: Path,
        split: str,
        val_patient_frac: float,
    ) -> List[Dict[str, object]]:
        entries = list(FetalPlanesDBAdapter(root).iter_entries())
        train_patients = [e.study_id for e in entries if e.split == "train" and e.study_id]
        val_patients = _val_patient_ids(train_patients, frac=val_patient_frac)

        samples: List[Dict[str, object]] = []
        for entry in entries:
            row_split = _resolve_split(entry, val_patients)
            if row_split != split:
                continue
            inst = entry.instances[0]
            label = inst.classification_label
            if label is None:
                continue
            samples.append({
                "image_path": entry.image_paths[0],
                "label": int(label),
                "sample_id": entry.series_id,
                "patient_id": entry.study_id,
                "plane": inst.label_raw,
            })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image = _load_rgb_tensor(
            sample["image_path"],
            self.img_size,
            augment=(self.split == "train"),
        )
        return {
            "image": image,
            "target": torch.tensor(sample["label"], dtype=torch.long),
            "sample_id": sample["sample_id"],
            "patient_id": sample["patient_id"],
            "plane": sample["plane"],
        }


class FetalPlanesDBFinetune(FinetuneExperiment):
    """FETAL_PLANES_DB 6-class fetal plane classification."""

    EXPERIMENT_NAME = "fetal_planes_db_classification"
    DATASET_ID      = "FETAL_PLANES_DB"
    TASK            = "classification"
    BENCHMARK_CLS   = None

    def build_head(self, embed_dim: int, cfg: FinetuneConfig):
        head_type = cfg.head_type if cfg.head_type in ("linear", "mlp", "attentive_pool") else "linear"
        return build_cls_head(embed_dim, n_classes=N_CLASSES, head_type=head_type)

    def build_dataloader(self, split: str) -> DataLoader:
        ds = FetalPlanesDBDataset(
            str(self.data_root),
            split,
            img_size=self.cfg.input_size,
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def compute_loss(self, batch, feats, head_output):
        return F.cross_entropy(head_output, batch["target"])

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.encoder.eval()

        all_logits: List[np.ndarray] = []
        all_labels: List[int] = []
        total_loss = 0.0
        n_batches = 0

        for batch in val_loader:
            batch = {
                k: v.to(self.device, non_blocking=True)
                if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            feats = self.encoder.encode_image(batch["image"])
            logits = self.head(feats["cls"])
            loss = self.compute_loss(batch, feats, logits)
            total_loss += loss.item()
            n_batches += 1
            all_logits.append(logits.float().cpu().numpy())
            all_labels.extend(batch["target"].cpu().tolist())

        if not all_labels:
            return {
                "val_loss": float("nan"),
                "val_acc": float("nan"),
                "val_f1": float("nan"),
                "val_auc": float("nan"),
            }

        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.asarray(all_labels, dtype=int)
        preds_np = logits_np.argmax(axis=1)
        probs_np = torch.softmax(torch.from_numpy(logits_np), dim=-1).numpy()

        try:
            from sklearn.metrics import f1_score
            macro_f1 = float(f1_score(labels_np, preds_np, average="macro", zero_division=0))
        except ImportError:
            macro_f1 = float("nan")

        return {
            "val_loss": round(total_loss / max(n_batches, 1), 4),
            "val_acc":  round(binary_accuracy(preds_np, labels_np), 4),
            "val_f1":   round(macro_f1, 4),
            "val_auc":  round(auc_roc(labels_np, probs_np, average="macro"), 4),
        }

    def run_viz(self, results: dict, output_dir: Path | None = None) -> None:
        pass


if __name__ == "__main__":
    FetalPlanesDBFinetune.main()
