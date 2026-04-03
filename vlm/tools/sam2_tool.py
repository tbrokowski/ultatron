"""
vlm/tools/sam2_tool.py  ·  SAM2Tool
=====================================

Wraps the frozen Segment Anything Model 2 (SAM2) for use as an agentic tool
inside the Qwen2.5-VL iMCoT loop.

Input  : a bbox [x1, y1, x2, y2] or point [x, y, label] parsed from the
         model's <tool_call> JSON output.
Output : a binary mask (H, W) numpy array.

The mask is then passed to SAMTokBridge (samtok.py) which converts it into
the 2-token observation that is appended to the trajectory.

References
----------
  SAM-R1 (arxiv 2505.22596)   — SAM as reward provider in RL loop
  GenSeg-R1 (arxiv 2602.09701) — SAM2-in-the-loop rewards for referring seg
  MedSAM-Agent (arxiv 2602.03320) — multi-turn agentic SAM for medical images
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

log = logging.getLogger(__name__)


class SAM2Tool:
    """
    Frozen SAM2 wrapper.  Accepts bbox or point prompts and returns binary masks.

    Parameters
    ----------
    model_cfg   : SAM2 config name (e.g. "sam2_hiera_large") or path to .yaml
    checkpoint  : path to SAM2 checkpoint (.pt)
    device      : "cuda" or "cpu"
    hf_cache_dir: optional HuggingFace cache directory override

    If neither model_cfg nor checkpoint is provided, falls back to trying the
    HuggingFace Hub (facebook/sam2-hiera-large).
    """

    def __init__(
        self,
        model_cfg:    Optional[str] = None,
        checkpoint:   Optional[str] = None,
        device:       str = "cuda",
        hf_cache_dir: Optional[str] = None,
    ):
        self.device      = device
        self._predictor  = None
        self._sam2_model = None

        self._build(model_cfg, checkpoint, hf_cache_dir)

    def _build(
        self,
        model_cfg:    Optional[str],
        checkpoint:   Optional[str],
        hf_cache_dir: Optional[str],
    ):
        """Load SAM2 predictor (frozen)."""
        try:
            self._build_from_sam2_package(model_cfg, checkpoint)
        except ImportError:
            log.warning(
                "sam2 package not found. Trying HuggingFace transformers SAM2..."
            )
            try:
                self._build_from_hf(hf_cache_dir)
            except Exception as e:
                log.error(
                    f"SAM2 could not be loaded: {e}. "
                    "Install sam2: pip install 'git+https://github.com/facebookresearch/sam2.git'"
                )
                self._predictor = None

    def _build_from_sam2_package(self, model_cfg: Optional[str], checkpoint: Optional[str]):
        """Use the official facebook/sam2 package."""
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if model_cfg is None:
            model_cfg = "sam2_hiera_large"
        if checkpoint is None:
            raise ImportError("checkpoint path is required for sam2 package build")

        sam2 = build_sam2(model_cfg, checkpoint, device=self.device)
        sam2.eval()
        for p in sam2.parameters():
            p.requires_grad_(False)
        self._sam2_model = sam2
        self._predictor  = SAM2ImagePredictor(sam2)
        log.info(f"SAM2Tool: loaded from package (cfg={model_cfg!r}, ckpt={checkpoint!r})")

    def _build_from_hf(self, hf_cache_dir: Optional[str]):
        """Fall back to HuggingFace transformers SAM2."""
        from transformers import Sam2Processor, Sam2Model

        model_id = "facebook/sam2-hiera-large"
        kwargs: Dict[str, Any] = {}
        if hf_cache_dir:
            kwargs["cache_dir"] = hf_cache_dir

        self._hf_processor = Sam2Processor.from_pretrained(model_id, **kwargs)
        self._hf_model     = Sam2Model.from_pretrained(model_id, **kwargs).to(self.device)
        self._hf_model.eval()
        for p in self._hf_model.parameters():
            p.requires_grad_(False)
        self._predictor = "hf"
        log.info("SAM2Tool: loaded from HuggingFace transformers")

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_bbox(
        self,
        image: Any,                   # PIL Image or np.ndarray (H, W, 3) uint8
        bbox:  List[float],           # [x1, y1, x2, y2] in pixel coords
    ) -> Tuple[np.ndarray, float]:
        """
        Run SAM2 with a bounding-box prompt.

        Returns
        -------
        mask  : np.ndarray bool (H, W) — best mask
        score : float — SAM2 confidence score
        """
        image_np = self._to_numpy(image)

        if self._predictor == "hf":
            return self._hf_predict_bbox(image_np, bbox)

        if self._predictor is None:
            log.warning("SAM2 not available — returning empty mask")
            h, w = image_np.shape[:2]
            return np.zeros((h, w), dtype=bool), 0.0

        self._predictor.set_image(image_np)
        box_arr = np.array(bbox, dtype=np.float32)[None]  # (1, 4)
        masks, scores, _ = self._predictor.predict(
            box=box_arr,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool), float(scores[best_idx])

    @torch.no_grad()
    def predict_point(
        self,
        image:  Any,
        points: List[List[float]],    # [[x, y], ...]
        labels: List[int],            # 1 = foreground, 0 = background
    ) -> Tuple[np.ndarray, float]:
        """Run SAM2 with a point prompt."""
        image_np = self._to_numpy(image)

        if self._predictor == "hf":
            return self._hf_predict_point(image_np, points, labels)
        if self._predictor is None:
            h, w = image_np.shape[:2]
            return np.zeros((h, w), dtype=bool), 0.0

        self._predictor.set_image(image_np)
        pts  = np.array(points, dtype=np.float32)
        lbs  = np.array(labels, dtype=np.int32)
        masks, scores, _ = self._predictor.predict(
            point_coords    = pts,
            point_labels    = lbs,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        return masks[best_idx].astype(bool), float(scores[best_idx])

    def _hf_predict_bbox(
        self, image_np: np.ndarray, bbox: List[float]
    ) -> Tuple[np.ndarray, float]:
        import torch
        inputs = self._hf_processor(
            images=image_np,
            input_boxes=[[[bbox]]],
            return_tensors="pt",
        ).to(self.device)
        outputs = self._hf_model(**inputs)
        masks = self._hf_processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )
        mask = masks[0][0][0].cpu().numpy().astype(bool)
        score = float(outputs.iou_scores[0][0][0].item())
        return mask, score

    def _hf_predict_point(
        self, image_np: np.ndarray, points: List[List[float]], labels: List[int]
    ) -> Tuple[np.ndarray, float]:
        inputs = self._hf_processor(
            images=image_np,
            input_points=[points],
            input_labels=[labels],
            return_tensors="pt",
        ).to(self.device)
        outputs = self._hf_model(**inputs)
        masks = self._hf_processor.post_process_masks(
            outputs.pred_masks, inputs["original_sizes"], inputs["reshaped_input_sizes"]
        )
        mask  = masks[0][0][0].cpu().numpy().astype(bool)
        score = float(outputs.iou_scores[0][0][0].item())
        return mask, score

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_numpy(image: Any) -> np.ndarray:
        """Convert PIL Image or ndarray to uint8 RGB numpy array."""
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)
            return image
        # PIL Image
        return np.array(image.convert("RGB"))
