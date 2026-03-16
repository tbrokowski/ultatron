"""
medsam_collator.py — Collators for SAM3, CLIP, and Aligned Dual-Stream batches
===============================================================================

Three additions to the collation layer:

1. SAMPromptCollator / SAMPromptBatch
   Bundles is_promptable samples with boxes, point prompts, text prompts,
   and ground-truth masks for MedSAM-3's iterative agent loop.

2. CrossModalCollator
   Packages image tensors + tokenised text for CLIP-style image-text
   contrastive training (cross-modal stream).

3. AlignedDualStreamBatch (replaces DualStreamBatch in Phase 3)
   Extends the dual-stream batch with alignment_pairs linking image
   crops to video tubes that share the same study_id and frame range.
   This is required for the cross-branch patch-to-tube distillation:
     L_cross = Σ_{(i,j)∈A} w_ij · D(z_i^img_teacher, z_j^vid_student)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
# 1. SAM Prompt Collator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SAMPromptBatch:
    """
    One batch for the MedSAM-3 agent loop.

    images        : (B, 1, H, W)  — full-resolution greyscale frames
                    (SAM encoder processes at its native 1024×1024 internally)
    boxes         : list of (N_i, 4) tensors in xyxy format, variable N_i per sample
    point_coords  : list of (N_i, K, 2) tensors  [K points per instance]
    point_labels  : list of (N_i, K)   tensors  [1=fg, 0=bg]
    text_prompts  : list of str — one text prompt per sample (may be empty string)
    gt_masks      : list of (N_i, 1, H, W) tensors — ground truth per instance
    n_instances   : LongTensor (B,) — N_i per sample
    anatomy_families : list[str]
    sample_ids    : list[str]
    dataset_ids   : list[str]
    is_iterative  : BoolTensor (B,) — run full agent refinement loop
    """
    images:           Tensor
    boxes:            List[Tensor]
    point_coords:     List[Tensor]
    point_labels:     List[Tensor]
    text_prompts:     List[str]
    gt_masks:         List[Tensor]
    n_instances:      Tensor
    anatomy_families: List[str]
    sample_ids:       List[str]
    dataset_ids:      List[str]
    is_iterative:     Tensor

    def to(self, device) -> "SAMPromptBatch":
        def _m(x):
            if isinstance(x, Tensor): return x.to(device)
            if isinstance(x, list):
                return [_m(v) for v in x]
            return x
        return SAMPromptBatch(
            images           = self.images.to(device),
            boxes            = [b.to(device) for b in self.boxes],
            point_coords     = [p.to(device) for p in self.point_coords],
            point_labels     = [p.to(device) for p in self.point_labels],
            text_prompts     = self.text_prompts,
            gt_masks         = [m.to(device) for m in self.gt_masks],
            n_instances      = self.n_instances.to(device),
            anatomy_families = self.anatomy_families,
            sample_ids       = self.sample_ids,
            dataset_ids      = self.dataset_ids,
            is_iterative     = self.is_iterative.to(device),
        )


class SAMPromptCollator:
    """
    Collates items from a dataset that returns per-sample dicts in the form:

        {
          "image":          Tensor (1, H, W)
          "boxes":          Tensor (N, 4)          — xyxy, from Instance.box
          "point_coords":   Tensor (N, K, 2)       — optional, may be zeros
          "point_labels":   Tensor (N, K)
          "text_prompt":    str                     — LabelSpec.text_label
          "gt_masks":       Tensor (N, 1, H, W)    — from Instance.mask_path
          "anatomy_family": str
          "sample_id":      str
          "dataset_id":     str
          "is_iterative":   bool
        }

    The dataset should filter to is_promptable=True samples before using
    this collator. Samples with no instances are skipped (filtered upstream).

    Usage
    -----
        dataset = DownstreamDataset(promptable_entries, task_config, ...)
        loader = DataLoader(dataset, collate_fn=SAMPromptCollator(), ...)

    Produces SAMPromptBatch objects ready for the MedSAM-3 interface.
    """

    def __call__(self, samples: List[Dict]) -> SAMPromptBatch:
        images = torch.stack([s["image"] for s in samples])  # B×1×H×W

        boxes        = [s.get("boxes",       torch.zeros(0, 4))     for s in samples]
        point_coords = [s.get("point_coords", torch.zeros(0, 1, 2)) for s in samples]
        point_labels = [s.get("point_labels", torch.zeros(0, 1))    for s in samples]
        gt_masks     = [s.get("gt_masks",     torch.zeros(0, 1, *images.shape[-2:])) for s in samples]

        n_instances = torch.tensor([len(b) for b in boxes], dtype=torch.long)

        text_prompts     = [s.get("text_prompt", "")          for s in samples]
        anatomy_families = [s.get("anatomy_family", "unknown") for s in samples]
        sample_ids       = [s.get("sample_id", "")             for s in samples]
        dataset_ids      = [s.get("dataset_id", "")            for s in samples]
        is_iterative     = torch.tensor([s.get("is_iterative", False) for s in samples])

        return SAMPromptBatch(
            images           = images,
            boxes            = boxes,
            point_coords     = point_coords,
            point_labels     = point_labels,
            text_prompts     = text_prompts,
            gt_masks         = gt_masks,
            n_instances      = n_instances,
            anatomy_families = anatomy_families,
            sample_ids       = sample_ids,
            dataset_ids      = dataset_ids,
            is_iterative     = is_iterative,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CrossModal (CLIP) Collator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CrossModalBatch:
    """
    One batch for CLIP-style image-text contrastive training.

    images          : (B, 1, H, W)
    input_ids       : (B, L)        — tokenised text (BioMedCLIP / PubMedBERT)
    attention_mask  : (B, L)
    text_raw        : list[str]     — raw text before tokenisation (for logging)
    anatomy_families: list[str]
    sample_ids      : list[str]
    dataset_ids     : list[str]
    """
    images:           Tensor
    input_ids:        Tensor
    attention_mask:   Tensor
    text_raw:         List[str]
    anatomy_families: List[str]
    sample_ids:       List[str]
    dataset_ids:      List[str]

    def to(self, device) -> "CrossModalBatch":
        return CrossModalBatch(
            images           = self.images.to(device),
            input_ids        = self.input_ids.to(device),
            attention_mask   = self.attention_mask.to(device),
            text_raw         = self.text_raw,
            anatomy_families = self.anatomy_families,
            sample_ids       = self.sample_ids,
            dataset_ids      = self.dataset_ids,
        )


class CrossModalCollator:
    """
    Collates items from CrossModalDataset:

        { "image": Tensor (1,H,W), "text": str, "sample_id": str,
          "dataset_id": str, "anatomy_family": str }

    Tokenises text on-the-fly using a Hugging Face tokeniser.

    Parameters
    ----------
    tokenizer_name : str
        HuggingFace model name or local path.
        Default: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        (BioMedCLIP-compatible).
    max_length : int
        Maximum token sequence length.
    """

    def __init__(
        self,
        tokenizer_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        max_length: int = 256,
    ):
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self._tokenizer = None   # lazy init (fork-safe for DataLoader workers)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            except ImportError:
                raise RuntimeError(
                    "transformers library required for CrossModalCollator: "
                    "pip install transformers"
                )
        return self._tokenizer

    def __call__(self, samples: List[Dict]) -> CrossModalBatch:
        images   = torch.stack([s["image"] for s in samples])   # B×1×H×W
        texts    = [s.get("text", "") for s in samples]

        tokenizer = self._get_tokenizer()
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return CrossModalBatch(
            images           = images,
            input_ids        = encoded["input_ids"],
            attention_mask   = encoded["attention_mask"],
            text_raw         = texts,
            anatomy_families = [s.get("anatomy_family", "unknown") for s in samples],
            sample_ids       = [s.get("sample_id", "")             for s in samples],
            dataset_ids      = [s.get("dataset_id", "")            for s in samples],
        )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Aligned Dual-Stream Batch  (Gap 1 — cross-branch frame alignment)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AlignmentPair:
    """
    One alignment triplet for L_cross computation.

    img_batch_idx : int   — index into AlignedDualStreamBatch.image_batch samples
    vid_batch_idx : int   — index into AlignedDualStreamBatch.video_batch samples
    frame_offset  : int   — temporal position t in the video clip (0-indexed)
    weight        : float — w_ij in L_cross; 1.0 if frames overlap, 0.5 for adjacent
    """
    img_batch_idx: int
    vid_batch_idx: int
    frame_offset:  int
    weight:        float = 1.0


@dataclass
class AlignedDualStreamBatch:
    """
    Extends DualStreamBatch with cross-branch frame alignment information.

    alignment_pairs : List[AlignmentPair]
        Identifies which image batch entry (i) corresponds to which video batch
        entry (j) at temporal position t.  Used by the cross-distillation module:

            for pair in batch.alignment_pairs:
                img_patch_tokens = img_teacher_out[pair.img_batch_idx]  # (N, D)
                vid_tube_tokens  = vid_student_out[pair.vid_batch_idx,
                                                   pair.frame_offset]   # (ph*pw, D)
                loss += pair.weight * D(img_patch_tokens, vid_tube_tokens)

    If no cross-branch pairs exist (disjoint studies in both batches), this list
    is empty and L_cross is skipped for this step. This is expected at the start
    of training when the curriculum draws diverse batches.
    """
    image_batch:      Dict[str, Any]
    video_batch:      Dict[str, Any]
    alignment_pairs:  List[AlignmentPair] = field(default_factory=list)

    @property
    def has_cross_branch_pairs(self) -> bool:
        return len(self.alignment_pairs) > 0

    @property
    def device(self):
        return self.image_batch["global_crops"].device

    def to(self, device) -> "AlignedDualStreamBatch":
        def _move(d):
            return {k: v.to(device) if isinstance(v, Tensor) else v
                    for k, v in d.items()}
        return AlignedDualStreamBatch(
            image_batch     = _move(self.image_batch),
            video_batch     = _move(self.video_batch),
            alignment_pairs = self.alignment_pairs,
        )


def build_alignment_pairs(
    image_batch: Dict[str, Any],
    video_batch: Dict[str, Any],
) -> List[AlignmentPair]:
    """
    Compute cross-branch alignment pairs from study_id and frame_index fields.

    The image batch must carry:
        "study_ids"         : list[str]    — one per sample
        "source_frame_idxs" : list[int]    — which frame in the original video

    The video batch must carry:
        "study_ids"              : list[str]     — one per clip
        "source_frame_indices"   : Tensor (B, T) — original frame numbers

    Two samples are paired when:
        image_study_id == video_study_id
        AND image_source_frame_idx falls within [vid_frame_start, vid_frame_end]
    """
    pairs: List[AlignmentPair] = []

    img_study_ids  = image_batch.get("study_ids", [])
    img_frame_idxs = image_batch.get("source_frame_idxs", [])
    vid_study_ids  = video_batch.get("study_ids", [])
    vid_frame_indices = video_batch.get("source_frame_indices")  # (B, T) or None

    if not img_study_ids or not vid_study_ids:
        return pairs

    # Build vid lookup: study_id → list[(vid_batch_idx, frame_positions)]
    vid_lookup: Dict[str, List[Tuple[int, List[int]]]] = {}
    for j, vsid in enumerate(vid_study_ids):
        if vsid not in vid_lookup:
            vid_lookup[vsid] = []
        if vid_frame_indices is not None:
            frames = vid_frame_indices[j].tolist()
        else:
            frames = []
        vid_lookup[vsid].append((j, frames))

    for i, (isid, iframe) in enumerate(zip(img_study_ids, img_frame_idxs)):
        if isid not in vid_lookup:
            continue
        for j, vframes in vid_lookup[isid]:
            if not vframes:
                # No frame index info — add a weak pair at offset 0
                pairs.append(AlignmentPair(i, j, frame_offset=0, weight=0.5))
                continue
            # Find which temporal slot in the video clip is closest to iframe
            vframes_arr = vframes
            if iframe < 0:
                continue  # image from static dataset, no frame index
            # Check overlap
            if iframe in vframes_arr:
                t = vframes_arr.index(iframe)
                pairs.append(AlignmentPair(i, j, frame_offset=t, weight=1.0))
            else:
                # Find nearest frame
                diffs = [abs(f - iframe) for f in vframes_arr]
                t = int(min(range(len(diffs)), key=lambda k: diffs[k]))
                dist = diffs[t]
                # Weight decays with distance (within 5 frames = valid pair)
                if dist <= 5:
                    w = max(0.1, 1.0 - dist * 0.15)
                    pairs.append(AlignmentPair(i, j, frame_offset=t, weight=w))

    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# 4. Aligned combined loader helper
# ─────────────────────────────────────────────────────────────────────────────

def make_aligned_dual_stream(
    image_batch: Dict[str, Any],
    video_batch: Dict[str, Any],
) -> AlignedDualStreamBatch:
    """
    Wraps a raw (image_batch, video_batch) pair into an AlignedDualStreamBatch
    by computing cross-branch alignment pairs.

    Drop-in replacement for the DualStreamBatch constructor in
    USFoundationDataModule.combined_loader():

        for img_b, vid_b in zip(img_loader, vid_loader):
            yield make_aligned_dual_stream(img_b, vid_b)
    """
    pairs = build_alignment_pairs(image_batch, video_batch)
    return AlignedDualStreamBatch(
        image_batch     = image_batch,
        video_batch     = video_batch,
        alignment_pairs = pairs,
    )
