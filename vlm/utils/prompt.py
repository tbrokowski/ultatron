"""
vlm/utils/prompt.py  ·  PromptBuilder
=======================================

Generates clinical prompt strings for each ultrasound task type and anatomy
family.  Prompts are designed to:

1. Elicit structured <think>…</think><answer>…</answer> output from Qwen.
2. Give the model permission to call the SAM2 segmentation tool when relevant.
3. Match the output format expected by FormatReward and MedGeminiReward.

Output format contract
----------------------
  All tasks:
    <think>step-by-step clinical reasoning</think>
    <answer>{"<key>": <value>}</answer>

  Regression (EF, measurement):
    <answer>{"value": 55.0}</answer>

  Classification (view, binary, multilabel):
    <answer>{"label": "A4C"}</answer>
    <answer>{"label": 1}</answer>

  Segmentation:
    <answer>{"label": "left_ventricle", "bbox": [x1, y1, x2, y2]}</answer>
    (model is expected to also call SAM2 to produce the actual mask)

  Open-ended / weak_label:
    <answer>{"findings": "...", "impression": "..."}</answer>

System prompt includes ToolRegistry.system_prompt_fragment() so the model
knows about the SAM2 tool.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class PromptBuilder:
    """
    Builds user-turn prompts for VLM GRPO training.

    Usage
    -----
      builder = PromptBuilder()
      prompt  = builder.build(task_type="regression", anatomy_family="cardiac", ...)
    """

    def build(
        self,
        task_type:      str,
        anatomy_family: str = "other",
        source_meta:    Dict[str, Any] = None,
        instances:      List[dict] = None,
        view_type:      Optional[str] = None,
        include_tool_instructions: bool = True,
    ) -> str:
        """
        Build a full system + user prompt for the given task.

        Parameters
        ----------
        task_type      : manifest task_type field
        anatomy_family : anatomy taxonomy string
        source_meta    : raw source metadata dict for context
        instances      : list of Instance dicts from the manifest
        view_type      : optional view hint (e.g. "A4C", "PLAX")
        include_tool_instructions : whether to prepend SAM2 tool instructions
        """
        source_meta = source_meta or {}
        instances   = instances   or []

        user_content = self._user_prompt(
            task_type, anatomy_family, source_meta, instances, view_type
        )

        if include_tool_instructions:
            from vlm.tools.registry import ToolRegistry
            tool_block = ToolRegistry.system_prompt_fragment()
            return tool_block + "\n\n" + user_content

        return user_content

    # ── User prompt per task ─────────────────────────────────────────────────

    def _user_prompt(
        self,
        task_type:      str,
        anatomy_family: str,
        source_meta:    Dict[str, Any],
        instances:      List[dict],
        view_type:      Optional[str],
    ) -> str:
        if task_type in ("regression", "measurement"):
            return self._regression_prompt(anatomy_family, source_meta, view_type)
        if task_type in ("classification", "sequence"):
            return self._classification_prompt(anatomy_family, source_meta, instances, view_type)
        if task_type == "segmentation":
            return self._segmentation_prompt(anatomy_family, instances)
        if task_type == "detection":
            return self._detection_prompt(anatomy_family, instances)
        # Default: open-ended / weak_label / ssl_only
        return self._open_ended_prompt(anatomy_family, source_meta)

    # ── Task-specific templates ───────────────────────────────────────────────

    @staticmethod
    def _regression_prompt(
        anatomy_family: str,
        source_meta:    Dict[str, Any],
        view_type:      Optional[str],
    ) -> str:
        view_hint = f" (view: {view_type})" if view_type else ""
        if anatomy_family == "cardiac":
            return (
                f"You are analyzing a cardiac ultrasound{view_hint}.\n\n"
                "Estimate the Left Ventricular Ejection Fraction (LVEF) as a percentage "
                "(0–100). Carefully examine systolic and diastolic phases.\n\n"
                "Think step-by-step about:\n"
                "  1. End-diastolic volume (EDV) estimation\n"
                "  2. End-systolic volume (ESV) estimation\n"
                "  3. LVEF = (EDV - ESV) / EDV × 100\n\n"
                "You may use the SAM2 tool to segment the left ventricle for a more "
                "precise volume estimate.\n\n"
                "Format your response as:\n"
                "<think>your reasoning here</think>\n"
                '<answer>{"value": <LVEF as float>}</answer>'
            )
        return (
            f"You are analyzing a {anatomy_family} ultrasound{view_hint}.\n\n"
            "Provide a quantitative measurement relevant to this image.\n\n"
            "<think>your reasoning here</think>\n"
            '<answer>{"value": <measurement as float>}</answer>'
        )

    @staticmethod
    def _classification_prompt(
        anatomy_family: str,
        source_meta:    Dict[str, Any],
        instances:      List[dict],
        view_type:      Optional[str],
    ) -> str:
        # Determine classification target from instances or anatomy
        labels = [inst.get("label_raw", "") for inst in instances if inst.get("label_raw")]
        label_hint = f"Possible labels include: {', '.join(labels)}." if labels else ""

        if anatomy_family == "cardiac":
            return (
                "You are analyzing a cardiac echocardiogram.\n\n"
                "Identify the echocardiographic view shown in this image.\n"
                "Common views: A4C (Apical 4-Chamber), PLAX (Parasternal Long-Axis), "
                "PSAX (Parasternal Short-Axis), A2C (Apical 2-Chamber), A3C (Apical 3-Chamber), "
                "Subcostal, Suprasternal.\n\n"
                f"{label_hint}\n\n"
                "<think>your reasoning here</think>\n"
                '<answer>{"label": "<view_name>"}</answer>'
            )
        if anatomy_family == "lung":
            return (
                "You are analyzing a lung ultrasound (LUS) image.\n\n"
                "Classify the pathological findings. Consider: B-lines, consolidation, "
                "pleural effusion, normal A-line pattern.\n"
                f"{label_hint}\n\n"
                "<think>your reasoning here</think>\n"
                '<answer>{"label": "<finding>"}</answer>'
            )
        if anatomy_family == "breast":
            return (
                "You are analyzing a breast ultrasound image.\n\n"
                "Classify the lesion if present: benign, malignant, or normal.\n\n"
                "<think>your reasoning here</think>\n"
                '<answer>{"label": "<benign|malignant|normal>"}</answer>'
            )
        # Generic
        return (
            f"You are analyzing a {anatomy_family} ultrasound.\n\n"
            f"Classify the primary finding in this image. {label_hint}\n\n"
            "<think>your reasoning here</think>\n"
            '<answer>{"label": "<class>"}</answer>'
        )

    @staticmethod
    def _segmentation_prompt(anatomy_family: str, instances: List[dict]) -> str:
        targets = [inst.get("label_raw", "region of interest") for inst in instances
                   if inst.get("mask_path") or inst.get("is_promptable")]
        target_str = targets[0] if targets else "the primary anatomical structure"

        return (
            f"You are analyzing a {anatomy_family} ultrasound.\n\n"
            f"Segment '{target_str}' in this image.\n\n"
            "Instructions:\n"
            "  1. Identify the structure's approximate location.\n"
            "  2. Call the SAM2 tool with a bounding box or point prompt.\n"
            "  3. Describe your segmentation in the answer.\n\n"
            "<think>your reasoning here</think>\n"
            '<answer>{"label": "<structure_name>", "bbox": [x1, y1, x2, y2]}</answer>'
        )

    @staticmethod
    def _detection_prompt(anatomy_family: str, instances: List[dict]) -> str:
        return (
            f"You are analyzing a {anatomy_family} ultrasound.\n\n"
            "Detect and localize the primary pathological finding.\n"
            "Provide a bounding box [x1, y1, x2, y2] in pixel coordinates.\n\n"
            "<think>your reasoning here</think>\n"
            '<answer>{"label": "<finding>", "bbox": [x1, y1, x2, y2]}</answer>'
        )

    @staticmethod
    def _open_ended_prompt(anatomy_family: str, source_meta: Dict[str, Any]) -> str:
        return (
            f"You are analyzing a {anatomy_family} ultrasound image.\n\n"
            "Provide a structured clinical interpretation:\n"
            "  - Findings: describe the key observations\n"
            "  - Impression: your overall clinical impression\n\n"
            "You may use the SAM2 tool to examine specific regions more closely.\n\n"
            "<think>your step-by-step analysis</think>\n"
            '<answer>{"findings": "<observations>", "impression": "<clinical impression>"}</answer>'
        )

    # ── System prompt for Qwen chat template ─────────────────────────────────

    @staticmethod
    def system_prompt() -> str:
        """System prompt prepended to all Qwen conversations."""
        return (
            "You are an expert medical imaging AI specializing in ultrasound interpretation. "
            "You provide accurate, evidence-based clinical assessments. "
            "Always think step-by-step before answering. "
            "Wrap your reasoning in <think>…</think> and your final answer in <answer>…</answer>. "
            "When precision is needed, use the SAM2 segmentation tool to identify structures."
        )
