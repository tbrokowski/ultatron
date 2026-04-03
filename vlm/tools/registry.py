"""
vlm/tools/registry.py  ·  ToolRegistry
========================================

Parses tool-call JSON emitted by the Qwen2.5-VL student and dispatches to the
appropriate tool (currently SAM2).  Returns observation tokens and metadata.

Tool-call format emitted by the model
--------------------------------------
  <tool_call>{"tool": "sam2", "bbox": [x1, y1, x2, y2]}</tool_call>
  <tool_call>{"tool": "sam2", "point": [x, y], "label": 1}</tool_call>
  <tool_call>{"tool": "sam2", "points": [[x1,y1],[x2,y2]], "labels": [1,0]}</tool_call>

Returns
-------
  obs_token_strs : List[str]   — SAMTok tokens to append to trajectory
  info           : dict        — {tool, input, mask_area_fraction, sam_score, mode}
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry that maps tool names to handlers.

    Parameters
    ----------
    sam2_tool     : SAM2Tool instance (frozen)
    samtok_bridge : SAMTokBridge instance
    tokenizer     : HuggingFace tokenizer (from Qwen processor)
    processor     : Qwen processor (needed for overlay image path)
    """

    def __init__(
        self,
        sam2_tool:     Any,
        samtok_bridge: Any,
        tokenizer:     Any,
        processor:     Any,
    ):
        self._sam2     = sam2_tool
        self._samtok   = samtok_bridge
        self._tokenizer = tokenizer
        self._processor = processor
        self._handlers  = {"sam2": self._handle_sam2}

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def dispatch(
        self,
        call:  Dict[str, Any],
        image: Optional[Any] = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Execute a tool call.

        Parameters
        ----------
        call  : parsed JSON dict from <tool_call>…</tool_call>
        image : original image context (PIL Image)

        Returns
        -------
        obs_tokens : List[str] — token strings to append (loss_mask=0)
        info       : metadata dict
        """
        tool_name = call.get("tool", "")
        handler   = self._handlers.get(tool_name)
        if handler is None:
            log.warning(f"ToolRegistry: unknown tool '{tool_name}' — returning empty obs.")
            return [], {"error": f"unknown_tool:{tool_name}"}
        return handler(call, image)

    def _handle_sam2(
        self,
        call:  Dict[str, Any],
        image: Optional[Any],
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Execute SAM2 with bbox or point prompt."""
        if image is None:
            log.warning("SAM2 called with no image context — skipping.")
            return [], {"error": "no_image"}

        info: Dict[str, Any] = {"tool": "sam2", "input": call}

        try:
            if "bbox" in call:
                bbox = call["bbox"]
                mask, score = self._sam2.predict_bbox(image, bbox)
            elif "point" in call:
                pts    = [[call["point"][0], call["point"][1]]]
                labels = [call.get("label", 1)]
                mask, score = self._sam2.predict_point(image, pts, labels)
            elif "points" in call:
                mask, score = self._sam2.predict_point(
                    image, call["points"], call.get("labels", [1] * len(call["points"]))
                )
            else:
                log.warning("SAM2 call missing 'bbox' or 'point' — returning empty mask.")
                import numpy as _np
                h = getattr(image, "height", 224)
                w = getattr(image, "width",  224)
                mask, score = _np.zeros((h, w), dtype=bool), 0.0

        except Exception as e:
            log.error(f"SAM2 inference failed: {e}")
            import numpy as _np
            h = getattr(image, "height", 224)
            w = getattr(image, "width",  224)
            mask, score = _np.zeros((h, w), dtype=bool), 0.0
            info["error"] = str(e)

        # Encode mask → obs tokens
        obs_tokens, enc_info = self._samtok.encode(
            mask, image, self._tokenizer, self._processor, sam_score=score
        )
        info.update(enc_info)
        info["raw_mask"] = mask  # kept for reward computation (gt Dice)
        return obs_tokens, info

    # ── System prompt helper ─────────────────────────────────────────────────

    @staticmethod
    def system_prompt_fragment() -> str:
        """
        Return the tool-call section to prepend to the system prompt so the
        model knows the SAM2 tool signature.  Mirrors DeepEyes Appendix A.1.
        """
        return """# Tools
You have access to a segmentation tool to assist with ultrasound image analysis.

<tools>
{
  "type": "function",
  "function": {
    "name": "sam2",
    "description": "Segment a region of interest in the ultrasound image using SAM2. Provide either a bounding box or a point prompt.",
    "parameters": {
      "type": "object",
      "properties": {
        "bbox": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 4,
          "maxItems": 4,
          "description": "Bounding box [x1, y1, x2, y2] in pixel coordinates."
        },
        "point": {
          "type": "array",
          "items": {"type": "number"},
          "minItems": 2,
          "maxItems": 2,
          "description": "Single point [x, y] in pixel coordinates."
        },
        "label": {
          "type": "integer",
          "description": "Point label: 1 for foreground, 0 for background. Default: 1."
        }
      }
    }
  }
}
</tools>

# How to call the tool
Emit a JSON block wrapped in <tool_call> tags:
<tool_call>{"tool": "sam2", "bbox": [x1, y1, x2, y2]}</tool_call>
or
<tool_call>{"tool": "sam2", "point": [x, y], "label": 1}</tool_call>

The segmentation result will appear as <observation>[SEG_TOKENS]</observation>.
"""
