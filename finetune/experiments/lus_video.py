"""
finetune/experiments/lus_video.py  ·  LUS video/image-level multilabel finetune
=================================================================================

Task:    Per-clip multilabel classification: A-line, B-line, confluent B-line,
         pleural effusion, large consolidation, small consolidation, pneumothorax
         (7-dim sigmoid vector, one per site/clip).
Dataset: Benin-LUS + RSA-LUS (pooled, patient-level train/val/test split).
         Loads both video clips (cleaned/videos/) and still images (cleaned/images/).
Branch:  Video branch (V-JEPA2, frozen) — clip_cls token.
         Video clips are encoded as full T-frame clips.
         Still images are tiled to T identical frames (static clip) so both
         modalities flow through the same video encoder and a single head.
Head:    7 independent binary MLP heads (one per finding) on clip_cls.
Metric:  Per-finding AUC-ROC + macro AUC across the 7 canonical findings.

Label source
------------
  * processed_files.csv  → which files exist, their Patient ID / Site / type
  * labels_multidiagnosis.csv (Benin) / rsa_pathology_labels.csv (RSA)
    → site-prefixed one-hot columns, e.g.  APXD_A-line, APXD_B-lines, ...

Patient-level split (70/15/15) is applied so that no patient appears in
multiple splits, consistent with the patient-level MIL experiment.
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
from models.heads.multifinding_head import build_multifinding_head

log = logging.getLogger(__name__)

# Canonical 7-finding vocabulary (mirrors data/adapters/lung/benin_lus.py)
LUS_CANONICAL_FINDINGS: Tuple[str, ...] = (
    "a_line",
    "b_line",
    "confluent_b_line",
    "pleural_effusion",
    "large_consolidation",
    "small_consolidation",
    "pneumothorax",
)

_SUFFIX_TO_INDEX: Dict[str, int] = {
    "A-line":                       0,
    "B-lines":                      1,
    "Confluent B-lines":            2,
    "Pleural effusion":             3,
    "large Consolidations":         4,
    "small Consolidations or Nodules": 5,
    "Pattern A' (pneumothorax)":    6,
}

N_FINDINGS   = len(LUS_CANONICAL_FINDINGS)
IMG_SIZE     = 224
N_FRAMES     = 8     # frames sampled per video clip (matches lus_patient.py)
VAL_FRAC     = 0.15
TEST_FRAC    = 0.15
ENCODE_BS    = 16    # clip encoding sub-batch size


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_video_labels(patient_row: dict, site: str) -> List[float]:
    """Build a length-7 multilabel vector for (patient, site)."""
    vec = [0.0] * N_FINDINGS
    site_prefix = site.strip()
    for suffix, idx in _SUFFIX_TO_INDEX.items():
        col = f"{site_prefix}_{suffix}"
        if col in patient_row:
            try:
                val = int(patient_row[col])
            except ValueError:
                val = 0
            if val == 1:
                vec[idx] = 1.0
    return vec


def _load_labels_by_patient(labels_csv: Path) -> Dict[str, dict]:
    """Return {record_id: row_dict} for the label CSV."""
    result: Dict[str, dict] = {}
    with labels_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("record_id", "").strip()
            if pid:
                result[pid] = row
    return result


def _load_clip_from_video(path: Path, n_frames: int, img_size: int) -> Optional[torch.Tensor]:
    """
    Load n_frames uniformly-sampled frames from a video clip.
    Returns (T, 3, H, W) float32 in [0, 1], or None on failure.
    Mirrors the logic in LUSPatientBagDataset._load_clip.
    """
    try:
        import decord  # type: ignore
        vr = decord.VideoReader(str(path), ctx=decord.cpu(0))
        total = len(vr)
        if total == 0:
            raise ValueError("empty video")
        idxs = np.linspace(0, total - 1, n_frames, dtype=int)
        frames_np = vr.get_batch(idxs).asnumpy()   # (T, H, W, 3) uint8
    except Exception:
        try:
            import cv2  # type: ignore
            cap = cv2.VideoCapture(str(path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            idxs = np.linspace(0, total - 1, n_frames, dtype=int)
            frames_list = []
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
    clip = torch.from_numpy(resized).float() / 255.0   # (T, H, W, 3)
    return clip.permute(0, 3, 1, 2)                    # (T, 3, H, W)


def _load_clip_from_image(path: Path, n_frames: int, img_size: int) -> Optional[torch.Tensor]:
    """
    Load a still image and tile it n_frames times to form a static clip.
    Returns (T, 3, H, W) float32 in [0, 1], or None on failure.
    Allows still images to be processed by the same video branch as video clips.
    """
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(path))
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
    except Exception:
        return None

    frame = torch.from_numpy(img).float() / 255.0   # (H, W, 3)
    frame = frame.permute(2, 0, 1)                  # (3, H, W)
    return frame.unsqueeze(0).expand(n_frames, -1, -1, -1).contiguous()  # (T, 3, H, W)


# ── Dataset ───────────────────────────────────────────────────────────────────

class LUSVideoDataset(Dataset):
    """
    One sample = one media file (video clip OR still image) for a given
    patient + lung site.

    Layout:
        {root}/cleaned/videos/{PatientID}_{Site}_{Depth}_{Count}.mp4
        {root}/cleaned/images/{PatientID}_{Site}_{Depth}_{Count}.png
        {root}/cleaned/processed_files.csv
        {root}/cleaned/labels_multidiagnosis.csv   (Benin)
        {root}/cleaned/rsa_pathology_labels.csv    (RSA)

    Returns:
        clip      : Tensor(T, 3, H, W) float32 [0, 1]
                    Videos: T uniformly-sampled frames via the video branch.
                    Images: single frame tiled T times (static clip).
        label     : Tensor(7,) float32  — per-finding multilabel vector
        patient_id: str
        site      : str
    """

    def __init__(
        self,
        root_benin: str,
        root_rsa:   str,
        split:      str   = "train",
        n_frames:   int   = N_FRAMES,
        img_size:   int   = IMG_SIZE,
        seed:       int   = 42,
        val_frac:   float = VAL_FRAC,
        test_frac:  float = TEST_FRAC,
    ):
        self.n_frames = n_frames
        self.img_size = img_size
        self.samples  = self._collect(root_benin, root_rsa, split, seed,
                                      val_frac, test_frac)
        log.info(f"[LUSVideo] {split}: {len(self.samples)} samples")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _collect(
        self,
        root_benin: str,
        root_rsa:   str,
        split:      str,
        seed:       int,
        val_frac:   float,
        test_frac:  float,
    ) -> List[dict]:
        # Accumulate all samples across both datasets, grouped by patient
        # for a patient-level split.
        samples_by_patient: Dict[str, List[dict]] = defaultdict(list)

        for source, root_str, labels_name, type_col in [
            ("benin", root_benin, "labels_multidiagnosis.csv", "type"),
            ("rsa",   root_rsa,   "rsa_pathology_labels.csv",  "Type"),
        ]:
            root        = Path(root_str) / "cleaned"
            labels_csv  = root / labels_name
            proc_csv    = root / "processed_files.csv"
            videos_dir  = root / "videos"
            images_dir  = root / "images"

            if not labels_csv.exists():
                log.warning(f"[LUSVideo] Labels not found: {labels_csv}. Skipping {source}.")
                continue
            if not proc_csv.exists():
                log.warning(f"[LUSVideo] processed_files.csv not found: {proc_csv}. Skipping {source}.")
                continue

            labels_by_patient = _load_labels_by_patient(labels_csv)

            with proc_csv.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    patient_id = row.get("Patient ID", "").strip()
                    if not patient_id:
                        continue
                    new_name = row.get("New File Name", "").strip()
                    if not new_name:
                        continue
                    site     = row.get("Site", "").strip()
                    row_type = row.get(type_col, "video").strip().lower()

                    if row_type == "video":
                        media_path = videos_dir / new_name
                    else:
                        media_path = images_dir / new_name

                    if not media_path.exists():
                        continue

                    label_row = labels_by_patient.get(patient_id)
                    if label_row is None:
                        continue

                    video_labels = _build_video_labels(label_row, site)
                    global_pid   = f"{source}_{patient_id}"

                    samples_by_patient[global_pid].append({
                        "path":       media_path,
                        "modality":   row_type,
                        "label":      video_labels,
                        "patient_id": global_pid,
                        "site":       site,
                    })

        # Patient-level split
        patient_ids = sorted(samples_by_patient.keys())
        rng = random.Random(seed)
        rng.shuffle(patient_ids)
        n_test = max(1, int(len(patient_ids) * test_frac))
        n_val  = max(1, int(len(patient_ids) * val_frac))
        test_set  = set(patient_ids[:n_test])
        val_set   = set(patient_ids[n_test:n_test + n_val])
        train_set = set(patient_ids[n_test + n_val:])

        split_map = {"train": train_set, "val": val_set, "test": test_set}
        keep = split_map.get(split, train_set)

        samples: List[dict] = []
        for pid in patient_ids:
            if pid in keep:
                samples.extend(samples_by_patient[pid])
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        path: Path    = s["path"]
        modality: str = s["modality"]

        if modality == "video":
            clip = _load_clip_from_video(path, self.n_frames, self.img_size)
        else:
            clip = _load_clip_from_image(path, self.n_frames, self.img_size)

        if clip is None:
            clip = torch.zeros(self.n_frames, 3, self.img_size, self.img_size)

        return {
            "clip":       clip,                                              # (T, 3, H, W)
            "label":      torch.tensor(s["label"], dtype=torch.float32),    # (7,)
            "patient_id": s["patient_id"],
            "site":       s["site"],
        }


# ── Finetune experiment ───────────────────────────────────────────────────────

class LUSVideoFinetune(FinetuneExperiment):
    """
    Per-clip video-level multilabel classification via the frozen video branch
    (V-JEPA2) + independent binary MLP heads (one per finding).

    Each sample — whether a .mp4 video clip or a still image — is encoded by
    the frozen video branch into a clip_cls embedding, then classified by
    MultiFindingBinaryHead with per-finding sigmoid BCE loss.

    Still images are tiled to n_frames identical frames so they flow through
    the same video encoder as genuine video clips.
    """

    EXPERIMENT_NAME = "lus_video_multilabel"
    DATASET_ID      = "Benin-LUS+RSA-LUS"
    TASK            = "multilabel_classification"
    BENCHMARK_CLS   = None

    def __init__(
        self,
        data_root_benin: str,
        data_root_rsa:   str,
        output_dir:      str,
        cfg:             FinetuneConfig,
        n_frames:        int = N_FRAMES,
        img_size:        int = IMG_SIZE,
        encode_bs:       int = ENCODE_BS,
        checkpoint_dir:  str | None = None,
    ):
        super().__init__(
            data_root=data_root_benin,
            output_dir=output_dir,
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
        )
        self.data_root_rsa = data_root_rsa
        self.n_frames      = n_frames
        self.img_size      = img_size
        self.encode_bs     = encode_bs

    # ── Abstract interface ────────────────────────────────────────────────────

    def build_head(self, embed_dim: int, cfg: FinetuneConfig) -> nn.Module:
        """Independent binary heads: video_embed_dim → 7 finding logits."""
        head_type = cfg.head_type if cfg.head_type in ("linear", "mlp") else "mlp"
        return build_multifinding_head(
            embed_dim=embed_dim,
            finding_names=LUS_CANONICAL_FINDINGS,
            head_type=head_type,
        )

    def setup(self, img_branch=None, device: str = "cuda", vid_branch=None, encoder=None):
        """Override: use the video branch (V-JEPA2) for clip-level encoding."""
        from finetune.backbones.ultatron_encoder import UltatronBranchEncoder
        if encoder is not None:
            self.encoder    = encoder
            self.vid_branch = encoder
        else:
            assert vid_branch is not None, "LUSVideoFinetune requires vid_branch"
            self.encoder    = UltatronBranchEncoder(img_branch, vid_branch)
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
        log.info(f"[LUSVideo] {self.head} on video branch: embed_dim={embed_dim}, "
                 f"n_findings={N_FINDINGS}, n_frames={self.n_frames} (dtype={backbone_dtype})")

    def build_dataloader(self, split: str) -> DataLoader:
        dataset = LUSVideoDataset(
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
        labels = batch["label"].to(self.device)   # (B, 7)
        return F.binary_cross_entropy_with_logits(head_output, labels)

    # ── Clip encoding helper ──────────────────────────────────────────────────

    @torch.no_grad()
    def _encode_clips(self, clips: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of clips through the frozen video backbone.

        Args:
            clips: (B, T, 3, H, W)
        Returns:
            embeddings: (B, D_vid)
        """
        B = clips.shape[0]
        embeddings = []
        for start in range(0, B, self.encode_bs):
            sub = clips[start:start + self.encode_bs].to(self.device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                out = self.encoder.encode_video(sub)
            embeddings.append(out["clip_cls"].float())
        return torch.cat(embeddings, dim=0)   # (B, D_vid)

    # ── Training loop ─────────────────────────────────────────────────────────

    def _train_epoch(self, loader, optimiser, scaler) -> float:
        self.head.train()
        self.encoder.eval()
        total_loss = 0.0
        n = 0

        for batch in loader:
            clips  = batch["clip"]                   # (B, T, 3, H, W)
            labels = batch["label"].to(self.device)  # (B, 7)

            feats  = self._encode_clips(clips)        # (B, D_vid)
            with torch.autocast("cuda", dtype=torch.bfloat16,
                                enabled=torch.cuda.is_available()):
                logits = self.head(feats)             # (B, 7)

            loss = F.binary_cross_entropy_with_logits(logits, labels)
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

        all_probs:  List[List[float]] = []
        all_labels: List[List[float]] = []
        total_loss = 0.0
        n = 0

        for batch in val_loader:
            clips  = batch["clip"]
            labels = batch["label"].to(self.device)  # (B, 7)

            feats  = self._encode_clips(clips)        # (B, D_vid)
            logits = self.head(feats)                 # (B, 7)

            loss = F.binary_cross_entropy_with_logits(logits, labels)
            total_loss += loss.item()
            n += 1

            probs = torch.sigmoid(logits).cpu().float().numpy()   # (B, 7)
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().float().numpy().tolist())

        y_true = np.array(all_labels)   # (N, 7)
        y_pred = np.array(all_probs)    # (N, 7)

        metrics: dict = {"val_loss": round(total_loss / max(n, 1), 4)}
        per_finding_aucs = []
        for i, finding in enumerate(LUS_CANONICAL_FINDINGS):
            col_true = y_true[:, i]
            col_pred = y_pred[:, i]
            if len(np.unique(col_true)) < 2:
                finding_auc = float("nan")
            else:
                finding_auc = round(float(auc_roc(col_true, col_pred)), 4)
            metrics[f"val_auc_{finding}"] = finding_auc
            per_finding_aucs.append(finding_auc)

        valid_aucs = [v for v in per_finding_aucs if not np.isnan(v)]
        metrics["val_auc_macro"] = round(float(np.mean(valid_aucs)), 4) if valid_aucs else float("nan")
        return metrics

    def evaluate(self, split: str = "test") -> dict:
        """Run clip-level evaluation. Returns per-finding + macro AUC and loss."""
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
    ) -> "LUSVideoFinetune":
        ft = raw.get("finetune", raw)
        cfg = FinetuneConfig.from_dict(ft)
        return cls(
            data_root_benin = raw.get("dataset_root_benin", ""),
            data_root_rsa   = raw.get("dataset_root_rsa",   ""),
            output_dir      = output_dir,
            cfg             = cfg,
            n_frames        = ft.get("n_frames",  N_FRAMES),
            img_size        = ft.get("img_size",  IMG_SIZE),
            encode_bs       = ft.get("encode_bs", ENCODE_BS),
        )
