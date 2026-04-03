"""
vlm/  ·  Ultatron VLM GRPO Integration
========================================

Wires the frozen Ultatron backbone (DINOv3 + V-JEPA2) into Qwen2.5-VL 7B as a
domain-adapted visual encoder, trains it as an agentic student via GRPO with
SAM2 as an in-loop segmentation tool, and uses Med-Gemini as the teacher /
reward judge with weak clinical labels.

Architecture
------------
  Ultatron backbone (FROZEN) → UltatronProjector (MLP 1024→3584) → Qwen LLM
                                                                    ↕ iMCoT
  SAM2 tool (FROZEN) ← bbox/point prompt     mask obs tokens →  Qwen LLM
  SAMTok (FROZEN)   ← SAM2 mask                                       ↓
  Med-Gemini judge (FROZEN) ← final answer ──────────── R_acc → GRPO update

References
----------
  DeepEyes  (ICLR 2026, arxiv 2505.14362)  — agentic GRPO / iMCoT / conditional tool reward
  SAMTok    (arxiv 2601.16093)              — mask = 2 discrete tokens
  MedGRPO   (arxiv 2512.06581)              — medical video RL, cross-dataset normalization
  SAM-R1    (arxiv 2505.22596)              — SAM as reward provider
  Med-Gemini (arxiv 2405.03162)             — multimodal medical judge
"""
from vlm.projector import UltatronProjector
from vlm.student   import StudentModel

__all__ = ["UltatronProjector", "StudentModel"]
