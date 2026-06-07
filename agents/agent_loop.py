"""
agents/agent_loop.py  ·  Agentic SAM Segmentation Loop
========================================================

Implements the multi-step agent loop referenced by viz/sam_prompts.py:

  1. Backbone attention  → derive spatial prompt points from CLS attention map.
  2. SAM invocation      → predict initial segmentation mask.
  3. Iterative refinement → examine mask boundary, add corrective points,
                            re-prompt SAM until quality converges or max_iters hit.
  4. (Optional) VLM reasoning → ask the Ultatron VLM to critique the mask and
                                  decide whether additional tool calls are needed.

This module is a pure inference utility — it does not train anything.
It is used by:
  - eval/metrics.py (AutoResearch agent loop evaluation)
  - viz/sam_prompts.py (visualisation of prompt placement and mask refinement)
  - scripts/run_agent.py (batch agent evaluation on downstream tasks)

Data flow
---------
    image
      │
      ├── backbone.forward_teacher() ──► attention_map (ph, pw)
      │         │
      │         └── attention_to_prompt_points() ──► prompt_points (N, 2)
      │
      └── SAM2Tool.predict_point(prompt_points) ──► mask_0, score_0
                │
                └── [for iter 1..max_iters]
                        boundary_corrective_points(mask_k, gt_mask?) ──► extra_points
                        SAM2Tool.predict_point(all_points) ──► mask_{k+1}, score_{k+1}
                        early_stop if Dice converged

Usage
-----
    from agents.agent_loop import AgentLoop, AgentLoopConfig

    loop = AgentLoop(sam2_tool=SAM2Tool(...), backbone=img_branch)
    result = loop.run(image, gt_mask=gt)

    # Visualise
    from viz.sam_prompts import plot_iterative_refinement
    plot_iterative_refinement(image, result.masks, gt_mask=gt, dice_scores=result.dice)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class AgentLoopConfig:
    """Hyperparameters for the agentic SAM refinement loop."""

    max_iters: int = 4
    """Maximum number of SAM re-prompting iterations after the initial call."""

    n_fg_prompts: int = 3
    """Foreground prompt points derived from top-attention patches per call."""

    n_bg_prompts: int = 1
    """Background corrective points added from mask boundary errors each iter."""

    attention_threshold: float = 0.6
    """Percentile of attention map to threshold for foreground prompt selection."""

    convergence_dice: float = 0.02
    """Stop early when Dice improvement between iterations falls below this."""

    min_score: float = 0.5
    """Minimum SAM confidence score; below this we add more corrective points."""

    use_vlm_critique: bool = False
    """If True and a VLM student is provided, query it for mask quality feedback."""


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class AgentLoopResult:
    """Full history of one agentic segmentation run."""

    masks: List[np.ndarray]
    """Binary masks (H, W) bool, one per iteration (including iter 0)."""

    scores: List[float]
    """SAM confidence scores per iteration."""

    dice: List[float]
    """Dice vs ground truth per iteration (empty list if gt_mask not provided)."""

    prompt_points: List[np.ndarray]
    """(N_pts, 2) prompt-point arrays (row, col) added at each iteration."""

    prompt_labels: List[np.ndarray]
    """(N_pts,) label arrays corresponding to prompt_points."""

    attention_map: Optional[np.ndarray] = None
    """(ph, pw) CLS attention map used to initialise prompts."""

    converged: bool = False
    """True if the loop stopped early due to Dice convergence."""

    @property
    def best_mask(self) -> np.ndarray:
        if not self.masks:
            raise RuntimeError("No masks produced by agent loop.")
        if self.dice:
            return self.masks[int(np.argmax(self.dice))]
        return self.masks[-1]

    @property
    def best_dice(self) -> float:
        return float(np.max(self.dice)) if self.dice else float("nan")


# ── Core agent ────────────────────────────────────────────────────────────────

class AgentLoop:
    """
    Multi-step agentic segmentation loop.

    Parameters
    ----------
    sam2_tool  : SAM2Tool  — frozen SAM2 wrapper (from vlm/tools/sam2_tool.py)
    backbone   : ImageBranch or None
                 If provided, CLS attention maps are extracted from the backbone
                 teacher for prompt initialisation.  If None, prompt points must
                 be supplied manually via run(prompt_points=...).
    cfg        : AgentLoopConfig
    device     : str
    vlm_student: StudentModel or None — for optional VLM critique
    """

    def __init__(
        self,
        sam2_tool:   Any,
        backbone:    Optional[Any] = None,
        cfg:         AgentLoopConfig = AgentLoopConfig(),
        device:      str = "cuda",
        vlm_student: Optional[Any] = None,
    ):
        self.sam2       = sam2_tool
        self.backbone   = backbone
        self.cfg        = cfg
        self.device     = device
        self.vlm        = vlm_student

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(
        self,
        image:         Any,              # PIL Image or np.ndarray (H, W, 3) uint8
        gt_mask:       Optional[np.ndarray] = None,  # (H, W) bool ground truth
        prompt_points: Optional[np.ndarray] = None,  # override backbone attention
        image_tensor:  Optional[Tensor] = None,       # pre-converted (3, H, W) float
    ) -> AgentLoopResult:
        """
        Run the full agentic loop.

        Parameters
        ----------
        image         : input image (PIL or numpy)
        gt_mask       : ground-truth mask for Dice computation (eval only)
        prompt_points : (N, 2) initial (row, col) prompt points; if None, derived
                        from backbone attention map
        image_tensor  : pre-converted float tensor for backbone forward

        Returns
        -------
        AgentLoopResult with full iteration history
        """
        image_np = _to_numpy(image)

        # ── Step 1: get initial prompt points ─────────────────────────────────
        attn_map = None
        if prompt_points is None:
            attn_map, prompt_points, prompt_labels = self._attention_prompts(
                image_np, image_tensor
            )
        else:
            prompt_labels = np.ones(len(prompt_points), dtype=np.int32)

        # ── Step 2: initial SAM call ──────────────────────────────────────────
        all_points = list(prompt_points)   # (row, col) — convert to (x, y) for SAM
        all_labels = list(prompt_labels)

        mask, score = self._sam_predict(image_np, all_points, all_labels)

        result = AgentLoopResult(
            masks         = [mask],
            scores        = [score],
            dice          = [_dice(mask, gt_mask)] if gt_mask is not None else [],
            prompt_points = [prompt_points.copy() if isinstance(prompt_points, np.ndarray) else np.array(prompt_points)],
            prompt_labels = [np.array(all_labels[:len(prompt_points)])],
            attention_map = attn_map,
        )

        # ── Step 3: iterative refinement ─────────────────────────────────────
        for _iter in range(self.cfg.max_iters):
            # Optional VLM critique — ask VLM whether the mask looks correct
            if self.cfg.use_vlm_critique and self.vlm is not None:
                should_continue = self._vlm_critique(image_np, mask)
                if not should_continue:
                    log.debug("VLM critique: mask accepted — stopping loop.")
                    break

            # Derive corrective points from mask boundary
            corr_pts, corr_lbs = self._corrective_points(mask, gt_mask)
            if len(corr_pts) == 0:
                log.debug("No corrective points generated — stopping loop.")
                break

            all_points = all_points + list(corr_pts)
            all_labels = all_labels + list(corr_lbs)

            new_mask, new_score = self._sam_predict(image_np, all_points, all_labels)

            new_dice = _dice(new_mask, gt_mask) if gt_mask is not None else None

            result.masks.append(new_mask)
            result.scores.append(new_score)
            result.prompt_points.append(np.array(corr_pts))
            result.prompt_labels.append(np.array(corr_lbs))
            if new_dice is not None:
                result.dice.append(new_dice)
                prev_dice = result.dice[-2]
                if abs(new_dice - prev_dice) < self.cfg.convergence_dice:
                    result.converged = True
                    log.debug(
                        f"Converged at iter {_iter + 1} (Dice Δ={abs(new_dice - prev_dice):.4f})"
                    )
                    break

            mask = new_mask

        return result

    # ── Attention → prompt points ─────────────────────────────────────────────

    def _attention_prompts(
        self,
        image_np:     np.ndarray,
        image_tensor: Optional[Tensor],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract CLS attention from backbone teacher → pick top-attention patches
        as foreground prompts.

        Returns
        -------
        attn_map      : (ph, pw) numpy float
        prompt_points : (N, 2) float — (row, col) in image pixels
        prompt_labels : (N,) int32 — all 1 (foreground)
        """
        if self.backbone is None or image_tensor is None:
            # Fall back to image-centre prompt if no backbone is available
            H, W = image_np.shape[:2]
            pts  = np.array([[H / 2, W / 2]], dtype=np.float32)
            lbs  = np.array([1], dtype=np.int32)
            return None, pts, lbs

        try:
            with torch.no_grad():
                x = image_tensor.unsqueeze(0).to(self.device)
                t_out = self.backbone.forward_teacher(x)
                attn  = self.backbone.teacher.get_last_attention()  # (1, H, N, N)

            # CLS-row attention → (ph, pw)
            attn_cls = attn[0, :, 0, 1:].mean(0)  # (N_patches,)
            ph = pw = int(len(attn_cls) ** 0.5)
            attn_map = attn_cls.reshape(ph, pw).cpu().numpy()

        except (AttributeError, NotImplementedError, RuntimeError) as e:
            log.warning(f"AgentLoop: backbone attention failed ({e}), using centre prompt.")
            H, W = image_np.shape[:2]
            pts  = np.array([[H / 2, W / 2]], dtype=np.float32)
            lbs  = np.array([1], dtype=np.int32)
            return None, pts, lbs

        # Threshold at attention_threshold percentile and pick top-N patch centres
        H, W = image_np.shape[:2]
        patch_h = H / ph
        patch_w = W / pw

        flat   = attn_map.flatten()
        thresh = float(np.percentile(flat, self.cfg.attention_threshold * 100))
        cands  = np.argwhere(attn_map >= thresh)  # (K, 2) — (row_patch, col_patch)

        if len(cands) == 0:
            cands = np.argwhere(attn_map >= attn_map.min())

        # Sort by attention descending, take top-N
        scores = attn_map[cands[:, 0], cands[:, 1]]
        order  = np.argsort(scores)[::-1][: self.cfg.n_fg_prompts]
        cands  = cands[order]

        # Convert patch indices → image pixel (row, col) — patch centre
        prompt_points = np.stack([
            (cands[:, 0] + 0.5) * patch_h,
            (cands[:, 1] + 0.5) * patch_w,
        ], axis=1).astype(np.float32)

        prompt_labels = np.ones(len(prompt_points), dtype=np.int32)

        return attn_map, prompt_points, prompt_labels

    # ── Corrective points from mask boundary ──────────────────────────────────

    def _corrective_points(
        self,
        current_mask: np.ndarray,           # (H, W) bool — current prediction
        gt_mask:      Optional[np.ndarray], # (H, W) bool — GT if available
    ) -> Tuple[List, List]:
        """
        Derive corrective prompt points from the current mask.

        With GT: FP boundary regions → background points; FN regions → foreground.
        Without GT: sample points along the predicted mask boundary (uncertain regions)
                    as additional foreground prompts.
        """
        if gt_mask is not None:
            return self._gt_corrective_points(current_mask, gt_mask)
        return self._boundary_corrective_points(current_mask)

    def _gt_corrective_points(
        self,
        pred: np.ndarray,
        gt:   np.ndarray,
    ) -> Tuple[List, List]:
        """Add FN centroid as foreground and FP centroid as background."""
        pts, lbs = [], []

        fn_mask = gt & ~pred
        if fn_mask.any():
            rows, cols = np.where(fn_mask)
            pts.append([float(rows.mean()), float(cols.mean())])
            lbs.append(1)

        fp_mask = pred & ~gt
        n_bg    = self.cfg.n_bg_prompts
        if fp_mask.any() and n_bg > 0:
            rows, cols = np.where(fp_mask)
            pts.append([float(rows.mean()), float(cols.mean())])
            lbs.append(0)

        return pts, lbs

    def _boundary_corrective_points(self, mask: np.ndarray) -> Tuple[List, List]:
        """
        Without GT: sample a few foreground points from low-confidence boundary
        regions to encourage the model to expand the mask.
        """
        try:
            from scipy.ndimage import distance_transform_edt, binary_erosion
            eroded  = binary_erosion(mask, iterations=3)
            boundary = mask & ~eroded
            rows, cols = np.where(boundary)
            if len(rows) == 0:
                return [], []
            n = min(self.cfg.n_bg_prompts, len(rows))
            idx  = np.random.choice(len(rows), n, replace=False)
            pts  = [[float(rows[i]), float(cols[i])] for i in idx]
            lbs  = [1] * n
            return pts, lbs
        except ImportError:
            return [], []

    # ── SAM dispatch ──────────────────────────────────────────────────────────

    def _sam_predict(
        self,
        image_np:   np.ndarray,
        points_rc:  List,        # list of [row, col]
        labels:     List,        # list of int
    ) -> Tuple[np.ndarray, float]:
        """Call SAM2 with (row, col) → converted to (x, y) = (col, row)."""
        points_xy = [[c, r] for r, c in points_rc]
        return self.sam2.predict_point(image_np, points_xy, labels)

    # ── VLM critique (optional) ───────────────────────────────────────────────

    def _vlm_critique(self, image_np: np.ndarray, mask: np.ndarray) -> bool:
        """
        Ask the VLM student to assess whether the current mask is acceptable.
        Returns True to continue iterating, False to accept and stop.

        Generates a brief structured prompt and checks the model's confidence token.
        """
        if self.vlm is None:
            return True
        try:
            from PIL import Image as _PIL
            pil  = _PIL.fromarray(image_np)
            resp = self.vlm.generate_with_tools(
                prompt=(
                    "Review the segmentation mask for this ultrasound image. "
                    "Is additional refinement needed? Answer YES or NO."
                ),
                image=pil,
                max_new_tokens=8,
            )
            text = resp.get("text", "").upper()
            return "YES" in text
        except Exception as e:
            log.debug(f"VLM critique failed ({e}), continuing loop.")
            return True


# ── Convenience factory ───────────────────────────────────────────────────────

def build_agent_loop(
    sam2_checkpoint: Optional[str] = None,
    sam2_model_cfg:  Optional[str] = None,
    backbone:        Optional[Any] = None,
    cfg:             Optional[AgentLoopConfig] = None,
    device:          str = "cuda",
    vlm_student:     Optional[Any] = None,
) -> AgentLoop:
    """
    Convenience factory — builds SAM2Tool and wraps in AgentLoop.

    Parameters
    ----------
    sam2_checkpoint : path to SAM2 .pt checkpoint
    sam2_model_cfg  : SAM2 config name (e.g. "sam2_hiera_large")
    backbone        : frozen ImageBranch for attention-based prompt initialisation
    cfg             : AgentLoopConfig (defaults used if None)
    device          : "cuda" or "cpu"
    vlm_student     : optional StudentModel for VLM critique

    Returns
    -------
    AgentLoop ready for inference
    """
    from vlm.tools.sam2_tool import SAM2Tool

    sam2 = SAM2Tool(
        model_cfg  = sam2_model_cfg,
        checkpoint = sam2_checkpoint,
        device     = device,
    )
    return AgentLoop(
        sam2_tool   = sam2,
        backbone    = backbone,
        cfg         = cfg or AgentLoopConfig(),
        device      = device,
        vlm_student = vlm_student,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_numpy(image: Any) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).clip(0, 255).astype(np.uint8)
        return image
    return np.array(image.convert("RGB"))


def _dice(pred: np.ndarray, gt: Optional[np.ndarray]) -> float:
    if gt is None:
        return float("nan")
    p = pred.astype(bool).flatten()
    g = (gt > 0).flatten()
    inter = (p & g).sum()
    denom = p.sum() + g.sum()
    return float(2 * inter / (denom + 1e-8))
