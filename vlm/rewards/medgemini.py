"""
vlm/rewards/medgemini.py  ·  MedGeminiReward
=============================================

Uses Med-Gemini (or a local medical VLM judge) as the teacher reward model.

Two operating modes
-------------------
Mode A — API (Gemini 1.5 Pro / 2.0 Flash via Google Generative AI):
    Requires GOOGLE_API_KEY env var or explicit api_key parameter.
    The judge receives: original image + clinical question + model answer +
    weak ground-truth label, and returns a score 0–1.

Mode B — Local fallback:
    Uses a locally-deployed medical VLM (e.g. LLaVA-Med, BioViL-T) loaded
    from a local path.  Falls back to rule-based scoring if no local model.

Mode C — Rule-based (always available):
    regression (EF)     : R = max(0, 1 - |pred - gt| / 20.0)
    classification      : binary exact-match or semantic match
    segmentation        : delegated to SegmentationReward
    Activated whenever the LLM judge is not available or task is numeric.

The mode is selected automatically based on config; API is tried first,
then local, then rule-based.

References
----------
  Med-Gemini (arxiv 2405.03162)   — multimodal medical AI judge
  MedGRPO (arxiv 2512.06581)      — medical LLM judge on 5 clinical dimensions
  MedGR2  (arxiv 2508.20549)      — generative reward learning for medical RL
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

from vlm.rewards.base import RewardFunction, RewardOutput

log = logging.getLogger(__name__)

# ── Prompt templates per task type ──────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are a board-certified radiologist specializing in ultrasound interpretation.
Your task is to evaluate an AI model's answer to a clinical question about an
ultrasound image.  You will be given:
  - The clinical question
  - The ground-truth label (weak label, may be imprecise)
  - The AI model's answer

Rate the clinical accuracy of the AI's answer on a scale from 0.0 to 1.0:
  1.0  — clinically correct and well-reasoned
  0.7  — mostly correct with minor inaccuracies
  0.5  — partially correct
  0.3  — relevant but substantially incorrect
  0.0  — completely wrong or hallucinated

Respond with ONLY a JSON object: {"score": <float 0.0-1.0>, "rationale": "<brief>"}
"""

_JUDGE_USER_TEMPLATE = """\
Clinical question: {question}

Ground-truth label: {ground_truth}

AI model answer: {model_answer}

Rate the clinical accuracy.
"""

# Normalisation constant for EF (±20 EF units = 0 reward per the plan spec)
_EF_NORM = 20.0


class MedGeminiReward(RewardFunction):
    """
    Teacher reward from Med-Gemini or fallback judge.

    Parameters
    ----------
    weight        : contribution weight in composite reward
    mode          : "auto" | "api" | "local" | "rule"
                    "auto" tries api → local → rule in order
    api_key       : Google Generative AI API key (overrides GOOGLE_API_KEY env var)
    api_model     : Gemini model ID (default: gemini-2.0-flash)
    local_model   : path or HF ID for local medical VLM judge
    local_dtype   : dtype for local model
    hf_cache_dir  : HF cache dir override
    """

    def __init__(
        self,
        weight:       float         = 1.0,
        mode:         str           = "auto",
        api_key:      Optional[str] = None,
        api_model:    str           = "gemini-2.0-flash",
        local_model:  Optional[str] = None,
        local_dtype:  str           = "bfloat16",
        hf_cache_dir: Optional[str] = None,
    ):
        super().__init__(weight)
        self.mode         = mode
        self.api_model    = api_model
        self.hf_cache_dir = hf_cache_dir

        self._api_client    = None
        self._local_model   = None
        self._local_proc    = None
        self._active_mode   = "rule"

        if mode in ("api", "auto"):
            self._init_api(api_key)
        if self._active_mode == "rule" and mode in ("local", "auto") and local_model:
            self._init_local(local_model, local_dtype)

    # ── Initialisation ───────────────────────────────────────────────────────

    def _init_api(self, api_key: Optional[str]):
        """Initialise Google Generative AI client."""
        key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            log.info("MedGeminiReward: no GOOGLE_API_KEY found; API mode unavailable.")
            return
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            self._api_client  = genai.GenerativeModel(self.api_model)
            self._active_mode = "api"
            log.info(f"MedGeminiReward: API mode active (model={self.api_model!r})")
        except ImportError:
            log.warning("google-generativeai not installed; API mode unavailable. "
                        "Install with: pip install google-generativeai")
        except Exception as e:
            log.warning(f"MedGeminiReward: API init failed ({e}); trying local/rule mode.")

    def _init_local(self, model_id: str, dtype: str):
        """Initialise a local medical VLM judge (e.g. LLaVA-Med)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            _dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                          "float32": torch.float32}
            kw: Dict[str, Any] = dict(
                torch_dtype=_dtype_map.get(dtype, torch.bfloat16),
                device_map="auto",
            )
            if self.hf_cache_dir:
                kw["cache_dir"] = self.hf_cache_dir
            self._local_proc  = AutoTokenizer.from_pretrained(model_id, **kw)
            self._local_model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
            self._local_model.eval()
            self._active_mode = "local"
            log.info(f"MedGeminiReward: local mode active (model={model_id!r})")
        except Exception as e:
            log.warning(f"MedGeminiReward: local model init failed ({e}); using rule mode.")

    # ── Compute ─────────────────────────────────────────────────────────────

    def compute(
        self,
        trajectory:   Dict[str, Any],
        ground_truth: Any,
        task_type:    str,
        dataset_id:   Optional[str] = None,
        image:        Optional[Any] = None,
    ) -> RewardOutput:
        """Compute accuracy reward for a single trajectory."""
        model_answer = trajectory.get("answer", "") or trajectory.get("text", "")

        # Numeric tasks: always rule-based (no LLM needed)
        if task_type in ("regression", "measurement"):
            return self._rule_regression(model_answer, ground_truth, task_type)

        if self._active_mode == "api":
            return self._api_judge(model_answer, ground_truth, task_type, image)
        if self._active_mode == "local":
            return self._local_judge(model_answer, ground_truth, task_type)
        return self._rule_classification(model_answer, ground_truth, task_type)

    # ── Rule-based rewards ───────────────────────────────────────────────────

    @staticmethod
    def _rule_regression(
        model_answer: str,
        ground_truth: Any,
        task_type:    str,
    ) -> RewardOutput:
        """R_acc = max(0, 1 - |pred - gt| / norm)."""
        try:
            gt_val = float(ground_truth)
        except (TypeError, ValueError):
            return RewardOutput(score=0.0, breakdown={"rule": "gt_not_numeric"})

        # Try to extract a float from the model answer
        pred_val = _extract_float(model_answer)
        if pred_val is None:
            return RewardOutput(score=0.0, breakdown={"rule": "pred_not_numeric"})

        norm  = _EF_NORM if task_type == "regression" else max(abs(gt_val) * 0.5, 1.0)
        score = max(0.0, 1.0 - abs(pred_val - gt_val) / norm)
        return RewardOutput(
            score=score,
            breakdown={"pred": pred_val, "gt": gt_val, "abs_err": abs(pred_val - gt_val)},
        )

    @staticmethod
    def _rule_classification(
        model_answer: str,
        ground_truth: Any,
        task_type:    str,
    ) -> RewardOutput:
        """Binary exact-match (case-insensitive, stripped)."""
        gt_str   = str(ground_truth).strip().lower()
        pred_str = str(model_answer).strip().lower()
        # Allow partial match (gt_str in pred_str) for natural language answers
        exact   = gt_str == pred_str
        partial = gt_str in pred_str or pred_str in gt_str
        score   = 1.0 if exact else (0.5 if partial else 0.0)
        return RewardOutput(
            score=score,
            breakdown={"exact_match": float(exact), "partial_match": float(partial)},
        )

    # ── API judge ────────────────────────────────────────────────────────────

    def _api_judge(
        self,
        model_answer: str,
        ground_truth: Any,
        task_type:    str,
        image:        Optional[Any],
    ) -> RewardOutput:
        """Call Gemini API to score the answer."""
        prompt = _JUDGE_USER_TEMPLATE.format(
            question=f"Clinical task: {task_type}",
            ground_truth=str(ground_truth),
            model_answer=model_answer,
        )
        try:
            contents: List[Any] = []
            if image is not None:
                # Pass image to Gemini for multimodal judging
                import PIL.Image
                if not isinstance(image, PIL.Image.Image):
                    PIL.Image.fromarray(image)
                contents.append(image)
            contents.append(prompt)

            response = self._api_client.generate_content(
                contents,
                generation_config={"temperature": 0.0, "max_output_tokens": 256},
                system_instruction=_JUDGE_SYSTEM,
            )
            text = response.text.strip()
            return _parse_judge_response(text)

        except Exception as e:
            log.warning(f"MedGeminiReward API call failed: {e}. Falling back to rule-based.")
            return self._rule_classification(model_answer, ground_truth, task_type)

    # ── Local judge ──────────────────────────────────────────────────────────

    def _local_judge(
        self,
        model_answer: str,
        ground_truth: Any,
        task_type:    str,
    ) -> RewardOutput:
        """Query a local medical VLM judge."""
        prompt = (
            f"{_JUDGE_SYSTEM}\n\n"
            + _JUDGE_USER_TEMPLATE.format(
                question=f"Clinical task: {task_type}",
                ground_truth=str(ground_truth),
                model_answer=model_answer,
            )
        )
        try:
            import torch
            inputs = self._local_proc(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            ).to(next(self._local_model.parameters()).device)
            with torch.no_grad():
                out = self._local_model.generate(**inputs, max_new_tokens=128, temperature=0.0)
            text = self._local_proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            return _parse_judge_response(text)
        except Exception as e:
            log.warning(f"Local judge failed: {e}. Rule-based fallback.")
            return self._rule_classification(model_answer, ground_truth, task_type)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_float(text: str) -> Optional[float]:
    """Extract the first float-like token from model answer text or JSON."""
    # Try JSON first
    try:
        parsed = json.loads(text)
        for key in ("value", "ef", "ejection_fraction", "measurement", "score"):
            if key in parsed:
                return float(parsed[key])
    except Exception:
        pass
    # Regex fallback
    m = re.search(r"[-+]?\d*\.?\d+", text)
    return float(m.group()) if m else None


def _parse_judge_response(text: str) -> RewardOutput:
    """Parse judge JSON response {"score": ..., "rationale": ...}."""
    try:
        parsed = json.loads(text)
        score  = float(parsed.get("score", 0.0))
        score  = max(0.0, min(1.0, score))
        return RewardOutput(
            score=score,
            breakdown={"judge_score": score},
            meta={"rationale": parsed.get("rationale", "")},
        )
    except Exception:
        # Try to extract a float from the text
        val = _extract_float(text)
        if val is not None:
            return RewardOutput(score=max(0.0, min(1.0, val)), breakdown={"judge_score": val})
        return RewardOutput(score=0.0, breakdown={"parse_error": 1.0}, meta={"raw": text})
