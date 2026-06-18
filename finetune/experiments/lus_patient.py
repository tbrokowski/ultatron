"""
finetune/experiments/lus_patient.py  ·  LUS patient-level TB finetune (Gated MIL)
===================================================================================

Task:    Binary classification: TB vs no-TB (patient level).
Dataset: Benin-LUS + RSA-LUS (pooled, patient-level train/val/test split).
Branch:  Video branch (V-JEPA2 teacher, frozen) — clip_cls token.
Head:    Gated Attention MIL pool → MLPClsHead (1 logit).
Metric:  Patient-level AUC-ROC for TB (val_auc_tb).

Multiple-instance learning note
--------------------------------
Each patient is a *bag* of video clips (different lung sites/depths).
Training: all clips for a patient are encoded by the frozen backbone, then
          aggregated by GatedAttentionMILPool into a patient embedding z,
          then classified by MLPClsHead.  Loss is binary BCE.
Evaluation: the MIL head directly produces patient-level predictions.
"""
from __future__ import annotations

import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from finetune.base import FinetuneExperiment, FinetuneConfig
from eval.metrics import auc_roc
from models.heads.mil_head import MIL_HIDDEN_DIM, PatientMILClsHead

log = logging.getLogger(__name__)

CLIP_ENCODE_BS  = 8
N_FRAMES        = 8
CLIP_SIZE       = 224


# ── Dataset ──────────────────────────────────────────────────────────────────

class LUSPatientBagDataset(Dataset):
    """
    One sample = one patient = a bag of video clips.

    Returns:
        clips     : list of (n_frames, 3, H, W) float32 tensors [0, 1]
        label     : scalar float32 (TB: 0 or 1)
        patient_id: str
    """

    def __init__(
        self,
        root_benin: str,
        root_rsa:   str,
        split:      str   = "train",
        n_frames:   int   = N_FRAMES,
        img_size:   int   = CLIP_SIZE,
        seed:       int   = 42,
        val_frac:   float = 0.15,
        test_frac:  float = 0.15,
    ):
        self.n_frames = n_frames
        self.img_size = img_size
        self.patients = self._collect(root_benin, root_rsa, split, seed,
                                      val_frac, test_frac)
        n_clips = sum(len(p["paths"]) for p in self.patients)
        log.info(f"[LUS] {split}: {len(self.patients)} patients, {n_clips} clips total")

    @staticmethod
    def _read_labels(labels_csv: Path) -> Dict[str, int]:
        """Return {patient_id: tb_label} (0 or 1)."""
        labels: Dict[str, int] = {}
        with labels_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("record_id", "").strip()
                if not pid:
                    continue
                try:
                    tb = int(float(row.get("TB Label", 0) or 0))
                    labels[pid] = int(tb > 0)
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
        all_patients: Dict[str, dict] = {}

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
                global_pid = f"{source}_{pid}"
                all_patients[global_pid] = {
                    "patient_id": global_pid,
                    "paths":      clips,
                    "label":      labels[pid],
                }

        patient_ids = sorted(all_patients.keys())
        rng = random.Random(seed)
        rng.shuffle(patient_ids)
        n_test = max(1, int(len(patient_ids) * test_frac))
        n_val  = max(1, int(len(patient_ids) * val_frac))
        test_set  = set(patient_ids[:n_test])
        val_set   = set(patient_ids[n_test:n_test + n_val])
        train_set = set(patient_ids[n_test + n_val:])

        split_map = {"train": train_set, "val": val_set, "test": test_set}
        keep = split_map.get(split, train_set)
        return [all_patients[pid] for pid in patient_ids if pid in keep]

    def _load_clip(self, path: Path) -> Optional[torch.Tensor]:
        try:
            import decord  # type: ignore
            vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
            total = len(vr)
            if total == 0:
                raise ValueError("empty video")
            idxs = np.linspace(0, total - 1, self.n_frames, dtype=int)
            frames_np = vr.get_batch(idxs).asnumpy()
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
        clip = torch.from_numpy(resized).float() / 255.0
        clip = clip.permute(0, 3, 1, 2)
        return clip

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        clips = []
        for path in patient["paths"]:
            clip = self._load_clip(path)
            if clip is None:
                clip = torch.zeros(self.n_frames, 3, self.img_size, self.img_size)
            clips.append(clip)

        if not clips:
            clips = [torch.zeros(self.n_frames, 3, self.img_size, self.img_size)]

        return {
            "clips":      clips,
            "label":      torch.tensor(float(patient["label"]), dtype=torch.float32),
            "patient_id": patient["patient_id"],
        }


def mil_collate_fn(batch: List[dict]) -> dict:
    clips_list  = [torch.stack(item["clips"], dim=0) for item in batch]
    labels      = torch.stack([item["label"] for item in batch])   # (B,)
    patient_ids = [item["patient_id"] for item in batch]
    return {"clips": clips_list, "labels": labels, "patient_ids": patient_ids}


# ── Finetune experiment ───────────────────────────────────────────────────────

class LUSPatientFinetune(FinetuneExperiment):
    """
    Patient-level TB prediction from LUS video clips via Gated Attention MIL + MLP.
    """

    EXPERIMENT_NAME = "lus_patient_tb_mil"
    DATASET_ID      = "Benin-LUS+RSA-LUS"
    TASK            = "binary_classification"
    BENCHMARK_CLS   = None

    def __init__(
        self,
        data_root_benin: str,
        data_root_rsa:   str,
        output_dir:      str,
        cfg:             FinetuneConfig,
        n_frames:        int = N_FRAMES,
        img_size:        int = CLIP_SIZE,
        mil_hidden_dim:  int = MIL_HIDDEN_DIM,
        clip_encode_bs:  int = CLIP_ENCODE_BS,
        checkpoint_dir:  str | None = None,
    ):
        super().__init__(
            data_root=data_root_benin,
            output_dir=output_dir,
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
        )
        self.data_root_rsa  = data_root_rsa
        self.n_frames       = n_frames
        self.img_size       = img_size
        self.mil_hidden_dim = mil_hidden_dim
        self.clip_encode_bs = clip_encode_bs

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        return PatientMILClsHead(
            embed_dim=embed_dim,
            hidden_dim=self.mil_hidden_dim,
            n_classes=1,
        )

    def setup(self, img_branch=None, device: str = "cuda", vid_branch=None, encoder=None):
        from finetune.backbones.ultatron_encoder import UltatronBranchEncoder
        if encoder is not None:
            self.encoder    = encoder
            self.img_branch = encoder
            self.vid_branch = encoder
        else:
            assert vid_branch is not None, "LUSPatientFinetune requires vid_branch"
            self.encoder    = UltatronBranchEncoder(img_branch, vid_branch)
            self.img_branch = img_branch
            self.vid_branch = vid_branch
        self.device = device

        if self.cfg.freeze_backbone and vid_branch is not None:
            for p in vid_branch.parameters():
                p.requires_grad_(False)
            vid_branch.eval()

        embed_dim = self.encoder.video_embed_dim
        try:
            backbone_dtype = next(
                p for mod in self.encoder._nn_modules() for p in mod.parameters()
            ).dtype
        except StopIteration:
            backbone_dtype = torch.bfloat16
        self.head = self.build_head(embed_dim, self.cfg).to(device=device, dtype=backbone_dtype)
        log.info(f"[LUS] PatientMILClsHead: embed_dim={embed_dim}, "
                 f"mil_hidden={self.mil_hidden_dim} (dtype={backbone_dtype})")

    def build_dataloader(self, split: str) -> DataLoader:
        dataset = LUSPatientBagDataset(
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
            collate_fn  = mil_collate_fn,
        )

    def compute_loss(self, batch, feats, head_output) -> torch.Tensor:
        labels = batch["labels"].to(self.device)
        return self.head.loss(head_output, labels)

    @torch.no_grad()
    def _encode_clips(self, clips: torch.Tensor) -> torch.Tensor:
        N = clips.shape[0]
        embeddings = []
        for start in range(0, N, self.clip_encode_bs):
            sub = clips[start:start + self.clip_encode_bs].to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                out = self.encoder.encode_video(sub)
            embeddings.append(out["clip_cls"].float())
        return torch.cat(embeddings, dim=0)

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        self.head.train()
        self.encoder.eval()
        total_loss = 0.0
        n = 0

        for batch in loader:
            labels = batch["labels"].to(self.device)
            logits_list = []

            head_dtype = next(self.head.parameters()).dtype
            for clips in batch["clips"]:
                H = self._encode_clips(clips).to(dtype=head_dtype)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_available()):
                    logit = self.head(H)
                logits_list.append(logit)

            logits = torch.stack(logits_list)
            loss = self.head.loss(logits, labels)
            self._backward_step_with_scaler(
                loss, optimiser, scaler, self.head.parameters()
            )
            optimiser.zero_grad(set_to_none=True)

            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.encoder.eval()

        all_probs:  List[float] = []
        all_labels: List[float] = []
        total_loss = 0.0
        n = 0

        for batch in val_loader:
            labels = batch["labels"].to(self.device)
            logits_list = []

            head_dtype = next(self.head.parameters()).dtype
            for clips in batch["clips"]:
                H = self._encode_clips(clips).to(dtype=head_dtype)
                logit = self.head(H)
                logits_list.append(logit)

            logits = torch.stack(logits_list)
            loss = self.head.loss(logits, labels)
            total_loss += loss.item()
            n += 1

            probs = torch.sigmoid(logits).cpu().float().numpy().reshape(-1)
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().float().numpy().reshape(-1).tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_probs)

        metrics: dict = {"val_loss": round(total_loss / max(n, 1), 4)}
        if len(np.unique(y_true)) < 2:
            metrics["val_auc_tb"] = float("nan")
        else:
            metrics["val_auc_tb"] = round(float(auc_roc(y_true, y_pred)), 4)
        return metrics

    def evaluate(self, split: str = "test") -> dict:
        loader  = self.build_dataloader(split)
        metrics = self.compute_val_metrics(loader)
        metrics["experiment"] = self.EXPERIMENT_NAME
        metrics["split"]      = split
        self._save_results(metrics)
        return metrics

    @classmethod
    def from_config(cls, raw: dict, output_dir: str) -> "LUSPatientFinetune":
        ft = raw.get("finetune", raw)
        cfg = FinetuneConfig.from_dict(ft)
        return cls(
            data_root_benin = raw.get("dataset_root_benin", ""),
            data_root_rsa   = raw.get("dataset_root_rsa",   ""),
            output_dir      = output_dir,
            cfg             = cfg,
            n_frames        = ft.get("n_frames",      N_FRAMES),
            img_size        = ft.get("img_size",       CLIP_SIZE),
            mil_hidden_dim  = ft.get("mil_hidden_dim", MIL_HIDDEN_DIM),
            clip_encode_bs  = ft.get("clip_encode_bs", CLIP_ENCODE_BS),
        )
