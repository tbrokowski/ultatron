"""
finetune/experiments/lus_patient.py  ·  LUS patient-level TB finetune (Gated MIL)
==================================================================================

Task:    Binary classification: TB (single head, sigmoid + BCE).
Dataset: Benin-LUS + RSA-LUS (pooled, patient-level train/val/test split).
Branch:  Video branch (V-JEPA2 teacher, frozen) — clip_cls token.
Head:    Gated Attention MIL (ABMIL, Ilse et al. 2018):
           each clip embedding → gated attention weights → weighted sum → Linear(D, 1).
Metric:  Patient-level AUC-ROC for TB.

Multiple-instance learning note
--------------------------------
Each patient is a *bag* of video clips (different lung sites/depths).
Training: all clips for a patient are encoded by the frozen backbone, then
          aggregated by the trainable GatedAttentionMIL head into a single
          patient-level logit.  Loss is BCE vs the patient's TB label.
Evaluation: the MIL head directly produces a patient-level prediction — no
            post-hoc mean pooling required.
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

TASK            = "tb"
MIL_HIDDEN_DIM  = 256
CLIP_ENCODE_BS  = 8     # sub-batch size when encoding clips for one patient
N_FRAMES        = 8
CLIP_SIZE       = 224


# ── Gated Attention MIL ───────────────────────────────────────────────────────

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention Multiple Instance Learning (Ilse et al., 2018).

    For a bag of N clip embeddings H ∈ R^{N×D}:

        e_i = tanh(W_V · h_i)          (attention branch)
        g_i = sigmoid(W_U · h_i)       (gate branch)
        a_i = softmax(w^T (e_i ⊙ g_i)) (attention weights, sums to 1)
        z   = Σ a_i · h_i              (aggregated patient representation)
        logit = W_c · z                (scalar TB logit)
    """

    def __init__(self, embed_dim: int, hidden_dim: int = MIL_HIDDEN_DIM):
        super().__init__()
        self.attn_V     = nn.Linear(embed_dim, hidden_dim)
        self.attn_U     = nn.Linear(embed_dim, hidden_dim)
        self.attn_w     = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            H: (N, D) clip embeddings for one patient
        Returns:
            scalar logit (pre-sigmoid)
        """
        A = self.attn_w(torch.tanh(self.attn_V(H)) * torch.sigmoid(self.attn_U(H)))  # (N, 1)
        A = torch.softmax(A, dim=0)                                                    # (N, 1)
        z = (A * H).sum(dim=0)                                                         # (D,)
        return self.classifier(z).squeeze(-1)                                          # scalar

    def attention_weights(self, H: torch.Tensor) -> torch.Tensor:
        """Return attention weights (N,) for interpretability."""
        A = self.attn_w(torch.tanh(self.attn_V(H)) * torch.sigmoid(self.attn_U(H)))
        return torch.softmax(A, dim=0).squeeze(-1)


# ── Dataset ──────────────────────────────────────────────────────────────────

class LUSPatientBagDataset(Dataset):
    """
    One sample = one patient = a bag of video clips.

    Layout:
        {root}/cleaned/videos/{PatientID}_{Site}_{Depth}_{Count}.mp4
        {root}/cleaned/labels_multidiagnosis.csv   (Benin)  or
        {root}/cleaned/rsa_pathology_labels.csv    (RSA)

    Returns:
        clips     : list of (n_frames, 3, H, W) float32 tensors [0, 1]
        label     : scalar float32  (TB: 0 or 1)
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

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _read_labels(labels_csv: Path) -> Dict[str, int]:
        """Return {patient_id: tb_label (0|1)}."""
        labels: Dict[str, int] = {}
        with labels_csv.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("record_id", "").strip()
                if not pid:
                    continue
                try:
                    labels[pid] = int(float(row.get("TB Label", 0) or 0))
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

        # Patient-level split
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
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        clips = []
        for path in patient["paths"]:
            clip = self._load_clip(path)
            if clip is None:
                clip = torch.zeros(self.n_frames, 3, self.img_size, self.img_size)
            clips.append(clip)

        # Guarantee at least one clip
        if not clips:
            clips = [torch.zeros(self.n_frames, 3, self.img_size, self.img_size)]

        return {
            "clips":      clips,                                          # list[Tensor(T,3,H,W)]
            "label":      torch.tensor(patient["label"], dtype=torch.float32),  # scalar
            "patient_id": patient["patient_id"],
        }


def mil_collate_fn(batch: List[dict]) -> dict:
    """
    Collate a list of patient bags into a batch.

    Because each patient has a different number of clips we keep clips as a
    Python list (one stacked tensor per patient) rather than padding.

    Returns:
        clips      : list[Tensor(N_i, T, 3, H, W)]  — one tensor per patient
        labels     : Tensor(B,)
        patient_ids: list[str]
    """
    clips_list   = [torch.stack(item["clips"], dim=0) for item in batch]
    labels       = torch.stack([item["label"] for item in batch])
    patient_ids  = [item["patient_id"] for item in batch]
    return {"clips": clips_list, "labels": labels, "patient_ids": patient_ids}


# ── Finetune experiment ───────────────────────────────────────────────────────

class LUSPatientFinetune(FinetuneExperiment):
    """
    Patient-level TB prediction from LUS video clips via Gated Attention MIL.

    Uses the video branch (frozen V-JEPA2 teacher, clip_cls token).
    Each patient's clips are independently encoded, then aggregated by the
    GatedAttentionMIL head into a single patient-level TB logit.
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
    ):
        super().__init__(data_root=data_root_benin, output_dir=output_dir, cfg=cfg)
        self.data_root_rsa  = data_root_rsa
        self.n_frames       = n_frames
        self.img_size       = img_size
        self.mil_hidden_dim = mil_hidden_dim
        self.clip_encode_bs = clip_encode_bs

    # ── Abstract interface ────────────────────────────────────────────────────

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        """Gated Attention MIL head."""
        return GatedAttentionMIL(embed_dim=embed_dim, hidden_dim=self.mil_hidden_dim)

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

        embed_dim      = vid_branch.embed_dim
        backbone_dtype = next(vid_branch.parameters()).dtype
        self.head = self.build_head(embed_dim, self.cfg).to(device=device, dtype=backbone_dtype)
        log.info(f"[LUS] GatedAttentionMIL head: embed_dim={embed_dim}, "
                 f"hidden_dim={self.mil_hidden_dim} (dtype={backbone_dtype})")

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
        labels = batch["labels"].to(self.device)   # (B,)
        return F.binary_cross_entropy_with_logits(head_output, labels)

    # ── Clip encoding helper ──────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_clips(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Encode a bag of clips through the frozen backbone.

        Args:
            clips: (N, T, 3, H, W)  — all clips for one patient
        Returns:
            H: (N, D)  — clip_cls embeddings
        """
        N = clips.shape[0]
        embeddings = []
        for start in range(0, N, self.clip_encode_bs):
            sub = clips[start:start + self.clip_encode_bs].to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                out = self.vid_branch.forward_teacher(sub)
            embeddings.append(out["clip_cls"].float())
        return torch.cat(embeddings, dim=0)   # (N, D)

    # ── Training loop ─────────────────────────────────────────────────────────

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        self.head.train()
        self.vid_branch.teacher.eval()
        total_loss = 0.0
        n = 0

        for batch in loader:
            labels = batch["labels"].to(self.device)   # (B,)
            logits_list = []

            for i, clips in enumerate(batch["clips"]):
                # clips: (N_i, T, 3, H, W)
                H = self._encode_clips(clips)           # (N_i, D)
                with torch.autocast("cuda", dtype=torch.bfloat16,
                                    enabled=torch.cuda.is_available()):
                    logit = self.head(H)                # scalar
                logits_list.append(logit)

            logits = torch.stack(logits_list)           # (B,)

            loss = F.binary_cross_entropy_with_logits(logits, labels)
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

        all_probs:  List[float] = []
        all_labels: List[float] = []
        total_loss = 0.0
        n = 0

        for batch in val_loader:
            labels = batch["labels"].to(self.device)   # (B,)
            logits_list = []

            for clips in batch["clips"]:
                H = self._encode_clips(clips)
                logit = self.head(H)
                logits_list.append(logit)

            logits = torch.stack(logits_list)          # (B,)
            loss   = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()
            n += 1

            probs = torch.sigmoid(logits).cpu().float().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().float().numpy().tolist())

        y_true = np.array(all_labels)
        y_pred = np.array(all_probs)

        if len(np.unique(y_true)) < 2:
            val_auc = float("nan")
        else:
            val_auc = round(float(auc_roc(y_true, y_pred)), 4)

        return {
            "val_auc_tb": val_auc,
            "val_loss":   round(total_loss / max(n, 1), 4),
        }

    def evaluate(self, split: str = "test") -> dict:
        """Run patient-level evaluation. Returns TB AUC and loss."""
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
