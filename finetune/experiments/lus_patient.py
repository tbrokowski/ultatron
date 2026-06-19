"""
finetune/experiments/lus_patient.py  ·  LUS patient-level TB finetune (Gated MIL)
===================================================================================

Task:    Binary classification: TB vs no-TB (patient level).
Dataset: Benin-LUS (default); optional RSA-LUS via ``include_rsa``.
         Benin uses the adapter's deterministic 80/10/10 patient splits.
Branch:  Video branch (V-JEPA2 teacher, frozen) — clip_cls token.
Head:    Gated Attention MIL pool → MLPClsHead (1 logit).
Metric:  Patient-level AUC-ROC for TB (val_auc_tb).

Performance
-----------
Frozen-backbone video encoding dominates runtime (~0.25 s/clip on large backbones).
Training subsamples up to ``max_clips_per_patient`` clips per epoch and batches
clip encoding across the full patient batch to keep the GPU busy.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from data.adapters.lung.benin_lus import BeninLUSAdapter
from finetune.base import FinetuneExperiment, FinetuneConfig
from eval.metrics import auc_roc
from models.heads.mil_head import MIL_HIDDEN_DIM, PatientMILClsHead

log = logging.getLogger(__name__)

CLIP_ENCODE_BS = 16
N_FRAMES = 8
CLIP_SIZE = 224
DEFAULT_MAX_CLIPS = 12


def _load_clip(path: Path, n_frames: int, img_size: int) -> Optional[torch.Tensor]:
    """Load (T, 3, H, W) float32 clip in [0, 1], or None on failure."""
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
        total = len(vr)
        if total == 0:
            raise ValueError("empty video")
        idxs = np.linspace(0, total - 1, n_frames, dtype=int)
        frames_np = vr.get_batch(idxs).asnumpy()
    except Exception:
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(str(path))
            frames_list = []
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            idxs = np.linspace(0, total - 1, n_frames, dtype=int)
            for fi in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, fr = cap.read()
                if ok:
                    frames_list.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            cap.release()
            if not frames_list:
                return None
            while len(frames_list) < n_frames:
                frames_list.append(frames_list[-1])
            frames_np = np.stack(frames_list[:n_frames])
        except Exception:
            return None

    import cv2  # type: ignore
    resized = np.stack([cv2.resize(f, (img_size, img_size)) for f in frames_np])
    clip = torch.from_numpy(resized).float() / 255.0
    return clip.permute(0, 3, 1, 2)


def collect_benin_patient_bags(root: str, split: str) -> List[dict]:
    """Build patient bags from Benin-LUS using the adapter's 80/10/10 splits."""
    adapter = BeninLUSAdapter(root)
    bags: Dict[str, dict] = {}

    for entry in adapter.iter_entries():
        if entry.split != split or entry.modality_type != "video":
            continue
        pid = entry.study_id
        pl = (entry.source_meta or {}).get("patient_labels") or {}
        path = Path(entry.image_paths[0])
        if pid not in bags:
            bags[pid] = {
                "patient_id": pid,
                "paths": [],
                "label": int(pl.get("tb", 0)),
            }
        bags[pid]["paths"].append(path)

    patients = [bags[pid] for pid in sorted(bags)]
    for patient in patients:
        patient["paths"] = sorted(patient["paths"])
    return patients


def collect_rsa_patient_bags(root: str, split: str) -> List[dict]:
    """Build patient bags from RSA-LUS using the adapter's 80/10/10 splits."""
    from data.adapters.lung.rsa_lus import RSALUSAdapter

    adapter = RSALUSAdapter(root)
    bags: Dict[str, dict] = {}

    for entry in adapter.iter_entries():
        if entry.split != split or entry.modality_type != "video":
            continue
        pid = entry.study_id
        pl = (entry.source_meta or {}).get("patient_labels") or {}
        path = Path(entry.image_paths[0])
        global_pid = f"rsa_{pid}"
        if global_pid not in bags:
            bags[global_pid] = {
                "patient_id": global_pid,
                "paths": [],
                "label": int(pl.get("tb", 0)),
            }
        bags[global_pid]["paths"].append(path)

    patients = [bags[pid] for pid in sorted(bags)]
    for patient in patients:
        patient["paths"] = sorted(patient["paths"])
    return patients


class LUSPatientBagDataset(Dataset):
    """
    One sample = one patient = a bag of video clips.

    Returns:
        clips     : (N, T, 3, H, W) float32 tensor [0, 1]
        label     : scalar float32 (TB: 0 or 1)
        patient_id: str
    """

    def __init__(
        self,
        root_benin: str,
        split: str = "train",
        root_rsa: str = "",
        include_rsa: bool = False,
        n_frames: int = N_FRAMES,
        img_size: int = CLIP_SIZE,
        seed: int = 42,
        max_clips_per_patient: Optional[int] = DEFAULT_MAX_CLIPS,
    ):
        self.split = split
        self.n_frames = n_frames
        self.img_size = img_size
        self.seed = seed
        self.max_clips = max_clips_per_patient if split == "train" else None
        self._epoch = 0

        self.patients = collect_benin_patient_bags(root_benin, split)
        if include_rsa and root_rsa:
            self.patients.extend(collect_rsa_patient_bags(root_rsa, split))

        n_clips = sum(len(p["paths"]) for p in self.patients)
        sources = "Benin-LUS" + ("+RSA-LUS" if include_rsa and root_rsa else "")
        log.info(f"[LUS] {split}: {len(self.patients)} patients, {n_clips} clips ({sources})")

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def _select_paths(self, paths: List[Path], idx: int) -> List[Path]:
        if self.max_clips is None or len(paths) <= self.max_clips:
            return paths
        rng = random.Random(self.seed + self._epoch * 10_000 + idx)
        return rng.sample(paths, self.max_clips)

    def __len__(self) -> int:
        return len(self.patients)

    def __getitem__(self, idx: int) -> dict:
        patient = self.patients[idx]
        paths = self._select_paths(patient["paths"], idx)
        clips: List[torch.Tensor] = []
        for path in paths:
            clip = _load_clip(path, self.n_frames, self.img_size)
            if clip is None:
                clip = torch.zeros(self.n_frames, 3, self.img_size, self.img_size)
            clips.append(clip)

        if not clips:
            clips = [torch.zeros(self.n_frames, 3, self.img_size, self.img_size)]

        return {
            "clips": torch.stack(clips, dim=0),
            "label": torch.tensor(float(patient["label"]), dtype=torch.float32),
            "patient_id": patient["patient_id"],
        }


def mil_collate_fn(batch: List[dict]) -> dict:
    clips_list = [item["clips"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    patient_ids = [item["patient_id"] for item in batch]
    return {"clips": clips_list, "labels": labels, "patient_ids": patient_ids}


def _parse_max_clips(raw_value) -> Optional[int]:
    if raw_value is None:
        return DEFAULT_MAX_CLIPS
    value = int(raw_value)
    return value if value > 0 else None


class LUSPatientFinetune(FinetuneExperiment):
    """Patient-level TB prediction from LUS video clips via Gated Attention MIL + MLP."""

    EXPERIMENT_NAME = "lus_patient_tb_mil"
    DATASET_ID = "Benin-LUS"
    TASK = "binary_classification"
    BENCHMARK_CLS = None

    def __init__(
        self,
        data_root_benin: str,
        output_dir: str,
        cfg: FinetuneConfig,
        data_root_rsa: str = "",
        include_rsa: bool = False,
        n_frames: int = N_FRAMES,
        img_size: int = CLIP_SIZE,
        mil_hidden_dim: int = MIL_HIDDEN_DIM,
        clip_encode_bs: int = CLIP_ENCODE_BS,
        max_clips_per_patient: Optional[int] = DEFAULT_MAX_CLIPS,
        checkpoint_dir: str | None = None,
    ):
        super().__init__(
            data_root=data_root_benin,
            output_dir=output_dir,
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
        )
        self.data_root_rsa = data_root_rsa
        self.include_rsa = include_rsa
        self.n_frames = n_frames
        self.img_size = img_size
        self.mil_hidden_dim = mil_hidden_dim
        self.clip_encode_bs = clip_encode_bs
        self.max_clips_per_patient = max_clips_per_patient
        if include_rsa and data_root_rsa:
            self.DATASET_ID = "Benin-LUS+RSA-LUS"

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        return PatientMILClsHead(
            embed_dim=embed_dim,
            hidden_dim=self.mil_hidden_dim,
            n_classes=1,
        )

    def setup(self, img_branch=None, device: str = "cuda", vid_branch=None, encoder=None):
        from finetune.backbones.ultatron_encoder import UltatronBranchEncoder
        if encoder is not None:
            self.encoder = encoder
            self.img_branch = encoder
            self.vid_branch = encoder
        else:
            assert vid_branch is not None, "LUSPatientFinetune requires vid_branch"
            self.encoder = UltatronBranchEncoder(img_branch, vid_branch)
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
        log.info(
            f"[LUS] PatientMILClsHead: embed_dim={embed_dim}, "
            f"mil_hidden={self.mil_hidden_dim}, max_clips={self.max_clips_per_patient} "
            f"(dtype={backbone_dtype})"
        )

    def build_dataloader(self, split: str) -> DataLoader:
        dataset = LUSPatientBagDataset(
            root_benin=str(self.data_root),
            root_rsa=str(self.data_root_rsa),
            include_rsa=self.include_rsa,
            split=split,
            n_frames=self.n_frames,
            img_size=self.img_size,
            max_clips_per_patient=self.max_clips_per_patient,
        )
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=(split == "train"),
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            collate_fn=mil_collate_fn,
            persistent_workers=self.cfg.num_workers > 0,
        )

    def run(self) -> dict:
        """Training loop with per-epoch clip resampling on the train split."""
        assert self.head is not None, "Call setup() before run()"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train_loader = self.build_dataloader("train")
        val_loader = self.build_dataloader("val")
        train_ds = train_loader.dataset

        from torch.cuda.amp import GradScaler

        all_params = list(self._iter_trainable_params())
        optimiser = torch.optim.AdamW(
            all_params,
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        use_amp_scaler = next(self.head.parameters()).dtype != torch.bfloat16
        scaler = GradScaler(enabled=torch.cuda.is_available() and use_amp_scaler)

        log.info(
            f"[{self.EXPERIMENT_NAME}] Training for up to "
            f"{self.cfg.max_epochs} epochs on {self.DATASET_ID}"
        )

        for epoch in range(self.cfg.max_epochs):
            self._current_epoch = epoch
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch)
            epoch_loss = self._train_epoch(train_loader, optimiser, scaler)
            val_metrics = self.compute_val_metrics(val_loader)
            val_metrics["epoch"] = epoch
            val_metrics["train_loss"] = epoch_loss
            self._train_log.append(val_metrics)

            if epoch % self.cfg.log_every == 0:
                log.info(
                    f"  Epoch {epoch:3d}  train_loss={epoch_loss:.4f}  "
                    + "  ".join(
                        f"{k}={v:.4f}"
                        for k, v in val_metrics.items()
                        if isinstance(v, float) and k != "train_loss"
                    )
                )

            monitor_val = val_metrics.get(self.cfg.monitor, epoch_loss)
            improved = (
                monitor_val < self._best_metric
                if self.cfg.monitor_mode == "min"
                else monitor_val > self._best_metric
            )
            if improved:
                self._best_metric = monitor_val
                self._patience_ctr = 0
                if self.cfg.checkpoint_best:
                    self._save_head("best_head.pt")
            else:
                self._patience_ctr += 1
                if self._patience_ctr >= self.cfg.patience:
                    log.info(f"  Early stopping at epoch {epoch} (patience={self.cfg.patience})")
                    break

        best_path = self._head_checkpoint_path("best_head.pt")
        if best_path.exists():
            self.load_head(str(best_path))

        self._save_log()
        return {"train_log": self._train_log, "best_metric": self._best_metric}

    def compute_loss(self, batch, feats, head_output) -> torch.Tensor:
        labels = batch["labels"].to(self.device)
        return self.head.loss(head_output, labels)

    @torch.no_grad()
    def _encode_clips(self, clips: torch.Tensor) -> torch.Tensor:
        embeddings = []
        for start in range(0, clips.shape[0], self.clip_encode_bs):
            sub = clips[start:start + self.clip_encode_bs].to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                out = self.encoder.encode_video(sub)
            embeddings.append(out["clip_cls"].float())
        return torch.cat(embeddings, dim=0)

    def _encode_patient_batch(
        self,
        clips_list: List[torch.Tensor],
        head_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flatten all clips in the batch, encode once, pad for MIL."""
        flat = torch.cat(clips_list, dim=0)
        flat_emb = self._encode_clips(flat).to(dtype=head_dtype)

        bags: List[torch.Tensor] = []
        offset = 0
        for clips in clips_list:
            n = clips.shape[0]
            bags.append(flat_emb[offset:offset + n])
            offset += n

        max_n = max(bag.shape[0] for bag in bags)
        embed_dim = bags[0].shape[-1]
        H = torch.zeros(len(bags), max_n, embed_dim, device=self.device, dtype=head_dtype)
        mask = torch.zeros(len(bags), max_n, dtype=torch.bool, device=self.device)
        for i, bag in enumerate(bags):
            n = bag.shape[0]
            H[i, :n] = bag
            mask[i, :n] = True
        return H, mask

    def _forward_mil_batch(self, clips_list: List[torch.Tensor], *, train: bool) -> torch.Tensor:
        head_dtype = next(self.head.parameters()).dtype
        H, mask = self._encode_patient_batch(clips_list, head_dtype)
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
                enabled=torch.cuda.is_available() and train,
            ):
                logits = self.head(H, mask=mask)
        if logits.dim() == 2 and logits.shape[-1] == 1:
            logits = logits.squeeze(-1)
        return logits

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        self.head.train()
        self.encoder.eval()
        total_loss = 0.0
        n = 0

        for batch in loader:
            labels = batch["labels"].to(self.device)
            logits = self._forward_mil_batch(batch["clips"], train=True)
            loss = self.head.loss(logits, labels)
            self._backward_step_with_scaler(loss, optimiser, scaler, self.head.parameters())
            optimiser.zero_grad(set_to_none=True)
            total_loss += loss.item()
            n += 1

        return total_loss / max(n, 1)

    @torch.no_grad()
    def compute_val_metrics(self, val_loader: DataLoader) -> dict:
        self.head.eval()
        self.encoder.eval()

        all_probs: List[float] = []
        all_labels: List[float] = []
        total_loss = 0.0
        n = 0

        for batch in val_loader:
            labels = batch["labels"].to(self.device)
            logits = self._forward_mil_batch(batch["clips"], train=False)
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
        loader = self.build_dataloader(split)
        metrics = self.compute_val_metrics(loader)
        metrics["experiment"] = self.EXPERIMENT_NAME
        metrics["split"] = split
        self._save_results(metrics)
        return metrics

    @classmethod
    def from_config(cls, raw: dict, output_dir: str) -> "LUSPatientFinetune":
        ft = raw.get("finetune", raw)
        cfg = FinetuneConfig.from_dict(ft)
        return cls(
            data_root_benin=raw.get("dataset_root_benin", ""),
            data_root_rsa=raw.get("dataset_root_rsa", ""),
            include_rsa=bool(raw.get("include_rsa", False)),
            output_dir=output_dir,
            cfg=cfg,
            n_frames=ft.get("n_frames", N_FRAMES),
            img_size=ft.get("img_size", CLIP_SIZE),
            mil_hidden_dim=ft.get("mil_hidden_dim", MIL_HIDDEN_DIM),
            clip_encode_bs=ft.get("clip_encode_bs", CLIP_ENCODE_BS),
            max_clips_per_patient=_parse_max_clips(ft.get("max_clips_per_patient")),
        )


def lus_patient_kwargs_from_raw(raw_cfg: dict, ft_raw: dict | None = None) -> dict:
    """Shared kwargs builder for scripts/finetune.py call sites."""
    ft = ft_raw if ft_raw is not None else raw_cfg.get("finetune", raw_cfg)
    return {
        "data_root_benin": raw_cfg.get("dataset_root_benin", ""),
        "data_root_rsa": raw_cfg.get("dataset_root_rsa", ""),
        "include_rsa": bool(raw_cfg.get("include_rsa", False)),
        "n_frames": ft.get("n_frames", N_FRAMES),
        "img_size": ft.get("img_size", CLIP_SIZE),
        "mil_hidden_dim": ft.get("mil_hidden_dim", MIL_HIDDEN_DIM),
        "clip_encode_bs": ft.get("clip_encode_bs", CLIP_ENCODE_BS),
        "max_clips_per_patient": _parse_max_clips(ft.get("max_clips_per_patient")),
    }
