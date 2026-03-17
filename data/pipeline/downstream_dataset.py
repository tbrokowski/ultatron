"""
downstream_dataset.py  ·  Universal downstream dataset for all training modes
==============================================================================

This module provides three dataset classes that all share the same
manifest + loading infrastructure but produce different label targets
depending on the training mode:

DownstreamDataset
    The workhorse.  For a given sample, loads the image/clip and builds
    a list[LabelTarget] covering ALL heads applicable to that sample
    based on the manifest entry's annotation tier and the active HeadRegistry.

CLIPPretrainDataset
    Adds a natural-language text alongside the image.  Used in pretraining
    phase when cross-modal CLIP loss is enabled.  Falls back to the
    anatomy + label_ontology for samples without free text.

PatientLevelDataset
    Groups manifest entries by study_id and presents all frames for a
    patient as a single item.  Used for patient-level classification
    (e.g., EF regression, overall LUS severity, PCOS yes/no).

All three classes are anatomy-agnostic: the same code handles cardiac
segmentation, LUS binary classification, and fetal measurement.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.schema.manifest import USManifestEntry, Instance, load_manifest
from data.pipeline.dataset import USFoundationDataset, load_image, load_mask, load_video_frames
from data.pipeline.transforms import (
    ImageSSLTransform, ImageSSLTransformConfig,
    VideoSSLTransform, VideoSSLTransformConfig,
    to_canonical_tensor,
)
from data.labels.label_interface import (
    LabelTarget, HeadSpec, HeadType, LossType,
    HEAD_REGISTRY,
    build_seg_target, build_cls_target, build_regression_target,
    build_clip_target, resolve_clip_text,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _head_ids_for_entry(entry: USManifestEntry) -> List[str]:
    """
    Return all head_ids from the global registry that are applicable
    to this entry, based on anatomy_family, available annotations, and
    dataset_id membership.
    """
    anatomy = entry.anatomy_family
    dataset_id = entry.dataset_id
    applicable = []

    for spec in HEAD_REGISTRY.all_enabled():
        # Filter by anatomy
        if spec.anatomy_family not in (anatomy, "multi"):
            continue
        # Filter by dataset membership (if spec restricts to specific datasets)
        if spec.dataset_ids and dataset_id not in spec.dataset_ids:
            continue
        # Filter by annotation availability
        if spec.is_segmentation and not entry.has_mask:
            continue
        if spec.is_classification and not entry.instances:
            continue
        if spec.is_regression:
            # Check if any instance has measurement_mm
            has_measurement = any(
                inst.measurement_mm is not None for inst in entry.instances
            )
            has_study_label = entry.source_meta.get("ef") is not None
            if not (has_measurement or has_study_label):
                continue
        applicable.append(spec.head_id)

    return applicable


# ─────────────────────────────────────────────────────────────────────────────
# 1. DownstreamDataset — universal, all head types
# ─────────────────────────────────────────────────────────────────────────────

class DownstreamDataset(USFoundationDataset):
    """
    Returns:
    {
      "image":          Tensor (1, H, W)
      "sample_id":      str
      "dataset_id":     str
      "anatomy":        str
      "modality_type":  str
      "label_targets":  List[LabelTarget]   ← the universal label interface
      "is_promptable":  bool
      "prompt_boxes":   List[[x1,y1,x2,y2]] or []
      "prompt_points":  List[[x,y,label]]   or []
      # Legacy keys for backward compat:
      "seg_mask":       Tensor(1,H,W) or None
      "cls_label":      int  (-1 = absent)
      "task_type":      str
    }
    """

    def __init__(
        self,
        entries: List[USManifestEntry],
        cfg: ImageSSLTransformConfig = None,
        root_remap: Optional[Dict[str, str]] = None,
        training_mode: str = "supervised",   # "supervised" | "clip" | "ssl"
        active_head_ids: Optional[List[str]] = None,  # None = all applicable
        return_aug: bool = False,            # if True, apply SSL augmentation
    ):
        super().__init__(entries, root_remap)
        self.cfg = cfg or ImageSSLTransformConfig()
        self.training_mode = training_mode
        self.active_head_ids = set(active_head_ids) if active_head_ids else None
        self.return_aug = return_aug
        if return_aug:
            self.transform = ImageSSLTransform(self.cfg)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.entries[idx]

        # ── Load image ───────────────────────────────────────────────────────
        if e.modality_type in ("video", "pseudo_video"):
            frames = self._load_clip(e, max_frames=64)
            frame_idx = torch.randint(len(frames), (1,)).item()
            img = frames[frame_idx]
        else:
            img = self._load_frame(e, 0)

        if self.return_aug:
            views = self.transform(img)
            image_tensor = views["global"][0]  # (C, H, W)
        else:
            from data.pipeline.transforms import to_canonical_tensor
            # Latest to_canonical_tensor signature does not take a size argument;
            # rely on native resolution here for smoke tests / simple usage.
            image_tensor = to_canonical_tensor(img)

        # ── Build label targets ───────────────────────────────────────────────
        targets: List[LabelTarget] = []
        applicable_heads = _head_ids_for_entry(e)

        for head_id in applicable_heads:
            if self.active_head_ids and head_id not in self.active_head_ids:
                continue
            spec = HEAD_REGISTRY.get(head_id)
            if spec is None:
                continue
            target = self._build_target(e, spec)
            if target is not None:
                targets.append(target)

        # CLIP mode: add text target for all labelled samples
        if self.training_mode == "clip":
            clip_target = self._build_clip_target(e)
            if clip_target:
                targets.append(clip_target)

        # ── Legacy keys (backward compat) ────────────────────────────────────
        seg_mask  = None
        cls_label = -1
        for inst in e.instances:
            if inst.mask_path and seg_mask is None:
                mp = self._remap_path(inst.mask_path)
                if Path(mp).exists():
                    mask_np = load_mask(mp)
                    seg_mask = torch.from_numpy(mask_np).float().unsqueeze(0)
            if inst.classification_label is not None and cls_label == -1:
                cls_label = inst.classification_label

        # ── Prompt info ───────────────────────────────────────────────────────
        prompt_boxes  = []
        prompt_points = []
        if e.is_promptable:
            for inst in e.instances:
                if inst.bbox_xyxy:
                    prompt_boxes.append(inst.bbox_xyxy)
                if inst.keypoints:
                    prompt_points.extend(inst.keypoints)

        return {
            "image":          image_tensor,
            "sample_id":      e.sample_id,
            "dataset_id":     e.dataset_id,
            "anatomy":        e.anatomy_family,
            "modality_type":  e.modality_type,
            "label_targets":  targets,
            "is_promptable":  e.is_promptable,
            "prompt_boxes":   prompt_boxes,
            "prompt_points":  prompt_points,
            # Legacy
            "seg_mask":       seg_mask,
            "cls_label":      cls_label,
            "task_type":      e.task_type,
        }

    def _build_target(
        self,
        e: USManifestEntry,
        spec: HeadSpec,
    ) -> Optional[LabelTarget]:
        """Route to the correct target builder based on head type."""

        if spec.is_segmentation:
            return self._build_seg_target_from_entry(e, spec)

        elif spec.is_classification:
            return self._build_cls_target_from_entry(e, spec)

        elif spec.is_regression:
            return self._build_regression_target_from_entry(e, spec)

        elif spec.is_clip:
            return None   # handled separately in _build_clip_target

        return None

    def _build_seg_target_from_entry(
        self, e: USManifestEntry, spec: HeadSpec
    ) -> Optional[LabelTarget]:
        """Load the first matching mask for this head."""
        for inst in e.instances:
            if inst.mask_path is None:
                continue
            mp = self._remap_path(inst.mask_path)
            if not Path(mp).exists():
                continue
            mask_np = load_mask(mp)
            mask_t = torch.from_numpy(mask_np).float().unsqueeze(0)
            return LabelTarget(
                head_id=spec.head_id,
                head_type=spec.head_type,
                loss_type=spec.loss_type,
                anatomy=e.anatomy_family,
                value=mask_t,
                instance_id=inst.instance_id,
                is_valid=True,
            )
        return None

    def _build_cls_target_from_entry(
        self, e: USManifestEntry, spec: HeadSpec
    ) -> Optional[LabelTarget]:
        """Extract classification label from instance or source_meta."""
        # Benin/RSA LUS: video-level multilabel vector packed in source_meta
        if spec.head_id == "lus_video_multilabel":
            vec = e.source_meta.get("video_labels")
            if vec is None:
                return None
            try:
                t = torch.tensor(vec, dtype=torch.float32)
            except Exception:
                return None
            return LabelTarget(
                head_id=spec.head_id,
                head_type=spec.head_type,
                loss_type=spec.loss_type,
                anatomy=e.anatomy_family,
                value=t,
                is_valid=True,
            )
        # Try instance-level classification labels
        for inst in e.instances:
            if inst.classification_label is not None:
                label_int = inst.classification_label
                # For binary: map to 0/1 using anatomy label space
                if spec.head_type == HeadType.BINARY_CLS:
                    # Positive class is anything that is not "normal" or "benign"
                    is_pos = any(
                        k in inst.label_ontology
                        for k in ("malignant", "b_line", "consolidation",
                                  "covid", "pcos", "cancer")
                    )
                    label_int = int(is_pos)
                return LabelTarget(
                    head_id=spec.head_id,
                    head_type=spec.head_type,
                    loss_type=spec.loss_type,
                    anatomy=e.anatomy_family,
                    value=torch.tensor(label_int, dtype=torch.long),
                    class_name=inst.label_raw,
                    instance_id=inst.instance_id,
                    is_valid=True,
                )
        # Try study-level labels stored in source_meta
        if spec.label_key in e.source_meta:
            raw_val = e.source_meta[spec.label_key]
            try:
                label_int = int(raw_val)
                return LabelTarget(
                    head_id=spec.head_id,
                    head_type=spec.head_type,
                    loss_type=spec.loss_type,
                    anatomy=e.anatomy_family,
                    value=torch.tensor(label_int, dtype=torch.long),
                    is_valid=True,
                )
            except (TypeError, ValueError):
                pass
        return None

    def _build_regression_target_from_entry(
        self, e: USManifestEntry, spec: HeadSpec
    ) -> Optional[LabelTarget]:
        """Extract regression value from instance or source_meta."""
        # Measurement from instance
        for inst in e.instances:
            if inst.measurement_mm is not None:
                return build_regression_target(inst.measurement_mm, spec.head_id, e.anatomy_family)

        # EF or other study-level regression from source_meta
        ef_raw = e.source_meta.get("ef")
        if ef_raw and spec.label_key == "ef_value":
            try:
                return build_regression_target(float(ef_raw), spec.head_id, e.anatomy_family)
            except (TypeError, ValueError):
                pass
        # RSA LUS per-site severity (1-7, -1 = not measured)
        if spec.head_id == "lus_site_severity":
            sev_raw = e.source_meta.get("severity")
            try:
                sev = float(sev_raw)
            except (TypeError, ValueError):
                return None
            if sev < 0:
                return None
            return build_regression_target(sev, spec.head_id, e.anatomy_family)
        return None

    def _build_clip_target(self, e: USManifestEntry) -> Optional[LabelTarget]:
        """Build a CLIP text target from available label information."""
        # Prefer explicit text from source_meta
        text = e.source_meta.get("report_text") or e.source_meta.get("label_text")
        if not text:
            # Derive from label_ontology
            ontology = e.instances[0].label_ontology if e.instances else None
            text = resolve_clip_text(
                {"anatomy_family": e.anatomy_family, "task_type": e.task_type},
                ontology
            )
        if not text:
            return None

        head_id = f"{e.anatomy_family}_clip_pretrain"
        if not HEAD_REGISTRY.get(head_id):
            head_id = "us_clip_universal"
        if not HEAD_REGISTRY.get(head_id):
            return None

        return build_clip_target(text, head_id, e.anatomy_family)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLIPPretrainDataset
# ─────────────────────────────────────────────────────────────────────────────

class CLIPPretrainDataset(DownstreamDataset):
    """
    Variant of DownstreamDataset that ensures every item has a CLIP text target.
    Samples without any label fall back to anatomy-level template text.

    Used when training_mode="clip" to generate (image, text) pairs for
    weakly-supervised CLIP pretraining at scale.

    Returns all standard DownstreamDataset keys plus:
        "clip_text":    str   (natural language description)
        "clip_text_tokens": None  (tokenization deferred to collator)
    """

    def __init__(
        self,
        entries: List[USManifestEntry],
        cfg: ImageSSLTransformConfig = None,
        root_remap: Optional[Dict[str, str]] = None,
        min_text_quality: int = 0,  # 0=all, 1=requires ontology label, 2=requires report
    ):
        # Filter entries by available text quality
        filtered = entries
        if min_text_quality >= 2:
            filtered = [
                e for e in entries
                if e.source_meta.get("report_text")
            ]
        elif min_text_quality >= 1:
            filtered = [
                e for e in entries
                if e.instances or e.label_raw
            ]

        super().__init__(
            filtered, cfg=cfg, root_remap=root_remap,
            training_mode="clip", return_aug=False,
        )
        # Enable universal CLIP head
        HEAD_REGISTRY.enable("us_clip_universal")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        out = super().__getitem__(idx)
        e = self.entries[idx]

        # Ensure there is always at least a fallback text
        clip_text = None
        for target in out["label_targets"]:
            if target.head_type == HeadType.CLIP_PRETRAIN and target.text:
                clip_text = target.text
                break

        if clip_text is None:
            # Fallback: anatomy-level description
            anatomy = e.anatomy_family.replace("_", " ")
            clip_text = f"An ultrasound image of {anatomy}"

        out["clip_text"] = clip_text
        out["clip_text_tokens"] = None   # will be filled by CLIPCollator
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. PatientLevelDataset
# ─────────────────────────────────────────────────────────────────────────────

class PatientLevelDataset(Dataset):
    """
    Groups manifest entries by study_id and presents all frames for a
    patient as a single dataset item.

    Used for:
    - Patient-level classification (PCOS: all ovarian frames → 1 label)
    - EF regression from echo clips
    - LUS severity score across multiple zones

    Returns:
    {
      "frames":         Tensor (T, 1, H, W)    — padded to max_frames
      "frame_mask":     BoolTensor (T,)         — True = valid frame
      "study_id":       str
      "dataset_id":     str
      "anatomy":        str
      "label_targets":  List[LabelTarget]       — study-level labels
      "n_frames":       int
    }
    """

    def __init__(
        self,
        entries: List[USManifestEntry],
        cfg: ImageSSLTransformConfig = None,
        root_remap: Optional[Dict[str, str]] = None,
        max_frames: int = 32,
        active_head_ids: Optional[List[str]] = None,
        video_cfg: Optional[VideoSSLTransformConfig] = None,
    ):
        self.cfg = cfg or ImageSSLTransformConfig()
        self.video_cfg = video_cfg
        self._video_transform = VideoSSLTransform(video_cfg) if video_cfg is not None else None
        self.root_remap = root_remap or {}
        self.max_frames = max_frames
        self.active_head_ids = set(active_head_ids) if active_head_ids else None

        # Group by study_id
        from collections import defaultdict
        groups: Dict[str, List[USManifestEntry]] = defaultdict(list)
        for e in entries:
            key = f"{e.dataset_id}__{e.study_id or e.sample_id}"
            groups[key].append(e)

        # Convert to list of (study_key, entries_list) pairs
        self.studies: List[Tuple[str, List[USManifestEntry]]] = list(groups.items())
        self._base_ds = DownstreamDataset.__new__(DownstreamDataset)
        self._base_ds.root_remap = self.root_remap

    def __len__(self) -> int:
        return len(self.studies)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        study_key, entries = self.studies[idx]
        # Sort by frame index if available
        entries = sorted(entries, key=lambda e: e.frame_indices[0] if e.frame_indices else 0)

        frames: List[Tensor] = []
        tube_masks_list: List[Tensor] = []

        for e in entries:
            if len(frames) >= self.max_frames:
                break
            try:
                if e.modality_type in ("video", "pseudo_video") and self._video_transform is not None:
                    raw_frames = self._base_ds._load_clip(
                        e, max_frames=self.video_cfg.max_n_frames
                    )
                    if not raw_frames:
                        continue
                    out = self._video_transform(raw_frames)
                    vis: Tensor  = out["visible"]    # (T, C, H, W)
                    tmask: Tensor = out["tube_mask"]  # (T, ph, pw)
                    remaining = self.max_frames - len(frames)
                    n_add = min(vis.shape[0], remaining)
                    for fi in range(n_add):
                        frames.append(vis[fi])
                    tube_masks_list.append(tmask[:n_add])
                elif e.modality_type in ("video", "pseudo_video"):
                    clips = self._base_ds._load_clip(e, max_frames=4)
                    if not clips:
                        continue
                    arr = clips[len(clips) // 2]
                    frames.append(to_canonical_tensor(arr))
                else:
                    arr = self._base_ds._load_frame(e, 0)
                    frames.append(to_canonical_tensor(arr))
            except Exception as ex:
                log.warning(f"Frame load error in patient dataset {study_key}: {ex}")

        n_valid = len(frames)
        if n_valid == 0:
            frames = [torch.zeros(3, 224, 224)]

        C = frames[0].shape[0]

        # Pad all real frames to the study-max (H, W) preserving native resolution
        max_h = max(f.shape[-2] for f in frames)
        max_w = max(f.shape[-1] for f in frames)
        padded: List[Tensor] = []
        for f in frames:
            if f.shape[-2] == max_h and f.shape[-1] == max_w:
                padded.append(f)
            else:
                p = torch.zeros(C, max_h, max_w, dtype=f.dtype)
                p[:, :f.shape[-2], :f.shape[-1]] = f
                padded.append(p)

        while len(padded) < self.max_frames:
            padded.append(torch.zeros(C, max_h, max_w))

        frame_stack = torch.stack(padded, dim=0)    # (max_frames, C, H, W)
        frame_mask = torch.zeros(self.max_frames, dtype=torch.bool)
        frame_mask[:n_valid] = True

        # Combine per-clip tube masks into a single (max_frames, ph, pw) tensor
        tube_mask_out: Optional[Tensor] = None
        if tube_masks_list:
            try:
                combined = torch.cat(tube_masks_list, dim=0)  # (n_masked, ph, pw)
                _, ph, pw = combined.shape
                if combined.shape[0] < self.max_frames:
                    pad = torch.zeros(
                        self.max_frames - combined.shape[0], ph, pw, dtype=torch.bool
                    )
                    combined = torch.cat([combined, pad], dim=0)
                tube_mask_out = combined[:self.max_frames]
            except Exception as ex:
                log.warning(f"tube_mask concat failed for {study_key}: {ex}")

        # Study-level label targets from first valid entry with labels
        targets: List[LabelTarget] = []
        dataset_id = entries[0].dataset_id
        anatomy = entries[0].anatomy_family

        for e in entries:
            for spec in HEAD_REGISTRY.for_anatomy(anatomy):
                if spec.head_type not in (HeadType.PATIENT_CLS,
                                           HeadType.PATIENT_REGRESSION):
                    continue
                if self.active_head_ids and spec.head_id not in self.active_head_ids:
                    continue
                # EF regression from echo
                if spec.head_type == HeadType.PATIENT_REGRESSION:
                    ef_raw = e.source_meta.get("ef")
                    if ef_raw:
                        try:
                            t = build_regression_target(float(ef_raw), spec.head_id, anatomy)
                            if t:
                                targets.append(t)
                                break
                        except (TypeError, ValueError):
                            pass
                # Benin/RSA LUS patient-level classification
                if spec.head_type == HeadType.PATIENT_CLS:
                    pl = e.source_meta.get("patient_labels") or {}
                    key_map = {
                        "lus_patient_tb": "tb",
                        "lus_patient_pneumonia": "pneumonia",
                        "lus_patient_covid": "covid",
                    }
                    field = key_map.get(spec.head_id)
                    if field is None or field not in pl:
                        continue
                    val = int(pl[field])
                    t = LabelTarget(
                        head_id=spec.head_id,
                        head_type=spec.head_type,
                        loss_type=spec.loss_type,
                        anatomy=anatomy,
                        value=torch.tensor(float(val), dtype=torch.float32),
                        is_valid=True,
                    )
                    targets.append(t)
                    break

        return {
            "frames":        frame_stack,
            "frame_mask":    frame_mask,
            "tube_mask":     tube_mask_out,   # (max_frames, ph, pw) bool or None
            "study_id":      study_key,
            "dataset_id":    dataset_id,
            "anatomy":       anatomy,
            "label_targets": targets,
            "n_frames":      n_valid,
        }

    def _remap_path(self, p: str) -> str:
        for old, new in self.root_remap.items():
            if p.startswith(old):
                return p.replace(old, new, 1)
        return p


# ─────────────────────────────────────────────────────────────────────────────
# 4. Builder functions (convenience API)
# ─────────────────────────────────────────────────────────────────────────────

def build_downstream_loader(
    manifest_path: str,
    task: str = "segmentation",         # "segmentation" | "classification" | "clip" | "patient"
    anatomy: Optional[str] = None,
    dataset_ids: Optional[List[str]] = None,
    split: str = "train",
    batch_size: int = 16,
    num_workers: int = 4,
    root_remap: Optional[Dict[str, str]] = None,
    cfg: Optional[ImageSSLTransformConfig] = None,
    max_patient_frames: int = 32,
    supervised_only: bool = True,
    active_head_ids: Optional[List[str]] = None,
    **loader_kwargs,
):
    """
    Build a DataLoader for a specific downstream task.

    Examples
    --------
    # Segmentation fine-tuning on cardiac data
    loader = build_downstream_loader(manifest, task="segmentation", anatomy="cardiac")

    # LUS binary classification (a-lines)
    loader = build_downstream_loader(manifest, task="classification", anatomy="lung",
                                      active_head_ids=["lus_aline_binary"])

    # CLIP pretraining (all anatomies)
    loader = build_downstream_loader(manifest, task="clip")

    # Patient-level EF regression
    loader = build_downstream_loader(manifest, task="patient", anatomy="cardiac",
                                      active_head_ids=["cardiac_ef_regression"])
    """
    from torch.utils.data import DataLoader
    from collators import DownstreamCollator

    entries = load_manifest(
        manifest_path,
        split=split,
        anatomy_families=[anatomy] if anatomy else None,
    )
    if supervised_only and task != "clip":
        entries = [e for e in entries if e.task_type != "ssl_only"]
    if dataset_ids:
        entries = [e for e in entries if e.dataset_id in dataset_ids]

    log.info(f"build_downstream_loader: task={task}, anatomy={anatomy}, "
             f"n_entries={len(entries)}")

    if task == "clip":
        dataset = CLIPPretrainDataset(entries, cfg=cfg, root_remap=root_remap)
    elif task == "patient":
        dataset = PatientLevelDataset(
            entries, cfg=cfg, root_remap=root_remap,
            max_frames=max_patient_frames,
            active_head_ids=active_head_ids,
        )
    else:
        training_mode = "supervised"
        dataset = DownstreamDataset(
            entries, cfg=cfg, root_remap=root_remap,
            training_mode=training_mode,
            active_head_ids=active_head_ids,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=DownstreamCollator(),
        shuffle=(split == "train"),
        drop_last=(split == "train"),
        **loader_kwargs,
    )
