"""
finetune/experiments/lus_patient.py  ·  LUS patient-level TB/Pneumonia/COVID finetune
======================================================================================

Task:    Binary classification per task: TB, Pneumonia, COVID.
Dataset: Benin-LUS + RSA-LUS (pooled, patient-level train/val/test split).
Branch:  Video branch (V-JEPA2 teacher, frozen) — clip_cls token.
Head:    Linear multi-label head: vid_embed_dim → 3 logits (sigmoid + BCE).
Metric:  Patient-level AUC per task + macro-average AUC.

Multi-instance note
-------------------
Each patient has multiple video clips (different lung sites/depths).
Training: treat each clip independently with patient-level labels.
Evaluation: aggregate clip-level predictions per patient (mean pooling),
            then compute AUC against patient-level ground truth.
"""
from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from finetune.base import FinetuneExperiment, FinetuneConfig
from eval.metrics import auc_roc

log = logging.getLogger(__name__)

TASK_KEYS   = ["tb", "pneumonia", "covid"]
N_FRAMES    = 8
CLIP_SIZE   = 224


# ── Dataset ──────────────────────────────────────────────────────────────────

class LUSPatientDataset(Dataset):
    """
    One sample = one video clip.  Label = patient-level (tb, pneumonia, covid).

    Layout:
        {root}/cleaned/videos/{PatientID}_{Site}_{Depth}_{Count}.mp4
        {root}/cleaned/labels_multidiagnosis.csv   (Benin)  or
        {root}/cleaned/rsa_pathology_labels.csv    (RSA)

    Returns:
        clip      : (n_frames, 3, H, W) float32 [0, 1]
        label     : (3,) float32  [tb, pneumonia, covid]
        patient_id: str
        source    : str  "benin" | "rsa"
    """

    def __init__(
        self,
        root_benin: str,
        root_rsa:   str,
        split:      str  = "train",
        n_frames:   int  = N_FRAMES,
        img_size:   int  = CLIP_SIZE,
        seed:       int  = 42,
        val_frac:   float = 0.15,
        test_frac:  float = 0.15,
    ):
        self.n_frames = n_frames
        self.img_size = img_size
        self.samples  = self._collect(root_benin, root_rsa, split, seed,
                                      val_frac, test_frac)
        log.info(f"[LUS] {split}: {len(self.samples)} clips from "
                 f"{len({s['patient_id'] for s in self.samples})} patients")

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _read_labels(labels_csv: Path) -> Dict[str, Dict[str, int]]:
        """Return {patient_id: {tb:0|1, pneumonia:0|1, covid:0|1}}."""
        labels: Dict[str, Dict[str, int]] = {}
        with labels_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("record_id", "").strip()
                if not pid:
                    continue
                try:
                    labels[pid] = {
                        "tb":        int(float(row.get("TB Label", 0) or 0)),
                        "pneumonia": int(float(row.get("Pneumonia",  0) or 0)),
                        "covid":     int(float(row.get("Covid",      0) or 0)),
                    }
                except (ValueError, KeyError):
                    pass
        return labels

    def _collect(
        self,
        root_benin: str,
        root_rsa:   str,
        split:      str,
        seed:       int,
        val_frac:   float,
        test_frac:  float,
    ) -> List[dict]:
        all_samples: List[dict] = []

        for source, root_str, labels_name in [
            ("benin", root_benin, "labels_multidiagnosis.csv"),
            ("rsa",   root_rsa,   "rsa_pathology_labels.csv"),
        ]:
            root        = Path(root_str) / "cleaned"
            labels_csv  = root / labels_name
            videos_dir  = root / "videos"

            if not labels_csv.exists():
                log.warning(f"[LUS] Labels not found: {labels_csv}. Skipping {source}.")
                continue
            if not videos_dir.exists():
                log.warning(f"[LUS] Videos dir not found: {videos_dir}. Skipping {source}.")
                continue

            labels = self._read_labels(labels_csv)
            clips_by_patient: Dict[str, List[Path]] = defaultdict(list)
            for vid_path in sorted(videos_dir.glob("*.mp4")):
                pid = vid_path.stem.split("_")[0]
                if pid in labels:
                    clips_by_patient[pid].append(vid_path)

            for pid, clips in clips_by_patient.items():
                for clip_path in clips:
                    all_samples.append({
                        "patient_id": f"{source}_{pid}",
                        "source":     source,
                        "path":       clip_path,
                        "labels":     labels[pid],
                    })

        # Patient-level split (stratify by TB label to preserve prevalence)
        patient_ids = sorted({s["patient_id"] for s in all_samples})
        rng = random.Random(seed)
        rng.shuffle(patient_ids)
        n_test = max(1, int(len(patient_ids) * test_frac))
        n_val  = max(1, int(len(patient_ids) * val_frac))
        test_set = set(patient_ids[:n_test])
        val_set  = set(patient_ids[n_test:n_test + n_val])
        train_set = set(patient_ids[n_test + n_val:])

        split_map = {"train": train_set, "val": val_set, "test": test_set}
        keep = split_map.get(split, train_set)
        return [s for s in all_samples if s["patient_id"] in keep]

    def _load_clip(self, path: Path) -> Optional[torch.Tensor]:
        """Load a video clip. Returns (T, 3, H, W) float32 or None on failure."""
        try:
            import decord  # type: ignore
            vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
            total = len(vr)
            if total == 0:
                raise ValueError("empty video")
            idxs = np.linspace(0, total - 1, self.n_frames, dtype=int)
            frames_np = vr.get_batch(idxs).asnumpy()   # (T, H, W, C) uint8
        except Exception:
            try:
                import cv2  # type: ignore
                cap = cv2.VideoCapture(str(path))
                frames_list = []
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
                idxs = np.linspace(0, total - 1, self.n_frames, dtype=int)
                for fi in idxs:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                    ok, fr = cap.read()
                    if ok:
                        frames_list.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                cap.release()
                if not frames_list:
                    return None
                while len(frames_list) < self.n_frames:
                    frames_list.append(frames_list[-1])
                frames_np = np.stack(frames_list[:self.n_frames])
            except Exception:
                return None

        import cv2  # type: ignore
        sz = self.img_size
        resized = np.stack([cv2.resize(f, (sz, sz)) for f in frames_np])
        clip = torch.from_numpy(resized).float() / 255.0   # (T, H, W, 3)
        clip = clip.permute(0, 3, 1, 2)                    # (T, 3, H, W)
        return clip

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        clip   = self._load_clip(sample["path"])
        if clip is None:
            clip = torch.zeros(self.n_frames, 3, self.img_size, self.img_size)

        lbl = sample["labels"]
        label = torch.tensor(
            [lbl["tb"], lbl["pneumonia"], lbl["covid"]], dtype=torch.float32
        )
        return {
            "clip":       clip,
            "label":      label,
            "patient_id": sample["patient_id"],
        }


# ── Finetune experiment ───────────────────────────────────────────────────────

class LUSPatientFinetune(FinetuneExperiment):
    """
    Patient-level TB/Pneumonia/COVID prediction from LUS video clips.

    Uses the video branch (frozen teacher, clip_cls token).
    Three-output sigmoid head; patient-level AUC by averaging clip predictions.
    """

    EXPERIMENT_NAME = "lus_patient_classification"
    DATASET_ID      = "Benin-LUS+RSA-LUS"
    TASK            = "multilabel_classification"
    BENCHMARK_CLS   = None   # custom evaluate() below

    def __init__(
        self,
        data_root_benin: str,
        data_root_rsa:   str,
        output_dir:      str,
        cfg:             FinetuneConfig,
        n_frames:        int = N_FRAMES,
        img_size:        int = CLIP_SIZE,
    ):
        # We store dual roots; pass benin as primary data_root for base class
        super().__init__(data_root=data_root_benin, output_dir=output_dir, cfg=cfg)
        self.data_root_rsa = data_root_rsa
        self.n_frames      = n_frames
        self.img_size      = img_size

    # ── Abstract interface ────────────────────────────────────────────────────

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        """Linear multi-label head: vid_embed_dim → 3 logits (tb/pneumonia/covid)."""
        return nn.Linear(embed_dim, len(TASK_KEYS))

    def setup(self, img_branch, device: str = "cuda", vid_branch=None):
        """Override: use video branch embed_dim."""
        assert vid_branch is not None, "LUSPatientFinetune requires vid_branch"
        self.img_branch = img_branch
        self.vid_branch = vid_branch
        self.device     = device

        if self.cfg.freeze_backbone:
            for p in vid_branch.parameters():
                p.requires_grad_(False)
            vid_branch.eval()

        embed_dim = vid_branch.embed_dim
        self.head = self.build_head(embed_dim, self.cfg).to(device)
        log.info(f"[LUS] Multi-label head: {self.head}")

    def build_dataloader(self, split: str) -> DataLoader:
        dataset = LUSPatientDataset(
            root_benin = str(self.data_root),
            root_rsa   = str(self.data_root_rsa),
            split      = split,
            n_frames   = self.n_frames,
            img_size   = self.img_size,
        )
        return DataLoader(
            dataset,
            batch_size  = self.cfg.batch_size,
            shuffle     = (split == "train"),
            num_workers = self.cfg.num_workers,
            pin_memory  = True,
        )

    def compute_loss(self, batch, feats, head_output) -> torch.Tensor:
        label = batch["label"].to(self.device)    # (B, 3)
        return F.binary_cross_entropy_with_logits(head_output, label)

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        self.head.train()
        self.vid_branch.teacher.eval()
        total_loss = 0.0
        n = 0

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
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.vid_branch.teacher.eval()

        patient_preds: Dict[str, List[np.ndarray]] = defaultdict(list)
        patient_labels: Dict[str, np.ndarray] = {}
        total_loss = 0.0
        n = 0

        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True)
                     if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            vid_out = self.vid_branch.forward_teacher(batch["clip"])
            logits  = self.head(vid_out["clip_cls"])
            loss    = self.compute_loss(batch, {}, logits)
            total_loss += loss.item()
            n += 1

            probs = torch.sigmoid(logits).cpu().float().numpy()
            for i, pid in enumerate(batch["patient_id"]):
                patient_preds[pid].append(probs[i])
                patient_labels[pid] = batch["label"][i].cpu().numpy()

        # Patient-level aggregation (mean over clips per patient)
        all_pred  = np.stack([np.mean(patient_preds[p], 0)
                              for p in sorted(patient_preds)])
        all_label = np.stack([patient_labels[p]
                              for p in sorted(patient_preds)])

        aucs = {}
        for ti, task in enumerate(TASK_KEYS):
            y_true = all_label[:, ti]
            y_pred = all_pred[:, ti]
            if len(np.unique(y_true)) < 2:
                aucs[f"val_auc_{task}"] = float("nan")
            else:
                aucs[f"val_auc_{task}"] = round(float(auc_roc(y_pred, y_true)), 4)

        valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
        aucs["val_auc_macro"] = round(float(np.mean(valid_aucs)), 4) if valid_aucs else float("nan")
        aucs["val_loss"]      = round(total_loss / max(n, 1), 4)
        return aucs

    def evaluate(self, split: str = "test") -> dict:
        """
        Run patient-level evaluation.  Returns per-task AUC + macro AUC.
        """
        loader  = self.build_dataloader(split)
        metrics = self.compute_val_metrics(loader)
        metrics["experiment"] = self.EXPERIMENT_NAME
        metrics["split"]      = split
        self._save_results(metrics)
        return metrics

    @classmethod
    def from_config(
        cls,
        raw: dict,
        output_dir: str,
    ) -> "LUSPatientFinetune":
        cfg = FinetuneConfig.from_dict(raw.get("finetune", raw))
        return cls(
            data_root_benin = raw.get("dataset_root_benin", ""),
            data_root_rsa   = raw.get("dataset_root_rsa",   ""),
            output_dir      = output_dir,
            cfg             = cfg,
            n_frames        = raw.get("finetune", raw).get("n_frames", N_FRAMES),
            img_size        = raw.get("finetune", raw).get("img_size",  CLIP_SIZE),
        )
