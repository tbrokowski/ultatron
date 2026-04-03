"""
vlm/student.py  ·  StudentModel
=================================

Wraps Qwen2.5-VL-7B-Instruct with:
  1. LoRA adapters on the LLM (q_proj, v_proj, gate_proj, up_proj)
  2. Prepending Ultatron projected tokens before Qwen's own visual tokens
  3. Tool-call interception during generation (routes SAM2 calls out-of-model)
  4. Token-wise loss masking so GRPO gradient is applied only to model-generated
     tokens (not SAM2 observation tokens appended by the environment)

The Qwen2.5-VL ViT is kept frozen; only the LoRA parameters and the
UltatronProjector are updated during GRPO training.

A frozen reference copy (π_ref) of the initial LLM is maintained for KL
divergence computation.

Architecture (forward pass)
---------------------------
  pixel_values  ──► Qwen ViT  ──► qwen_vis_tokens (B, Nq, Dllm)
  patch_tokens  ──► UltatronProjector ──► ulta_vis_tokens (B, Nu, Dllm)
  tube_tokens   ──►  (vid head)        ──► ulta_vid_tokens (B, Tv, Dllm)

  concat: [ulta_vis | ulta_vid | qwen_vis]  →  prepended into Qwen LLM context

Tool-call interception
----------------------
During auto-regressive generation, if the model emits a JSON tool-call block
  <tool_call>{"tool": "sam2", "bbox": [x1,y1,x2,y2]}</tool_call>
the ToolRegistry is invoked, the returned observation tokens are appended to the
sequence with loss_mask=0, and generation continues.  This follows DeepEyes
(arxiv 2505.14362) Eq. 1: s_t = {X_{≤t}; I_{≤t}}.
"""
from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# LoRA target modules for Qwen2.5-VL LLM
_LORA_TARGETS = ["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]

# Special token strings
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END   = "</tool_call>"
OBS_START       = "<observation>"
OBS_END         = "</observation>"


@dataclass
class LoRAConfig:
    r:             int        = 64
    lora_alpha:    int        = 128
    lora_dropout:  float      = 0.05
    target_modules: List[str] = field(default_factory=lambda: list(_LORA_TARGETS))
    bias:          str        = "none"
    task_type:     str        = "CAUSAL_LM"


@dataclass
class StudentConfig:
    """Full configuration for StudentModel."""
    model_name:      str        = "Qwen/Qwen2.5-VL-7B-Instruct"
    hf_cache_dir:    Optional[str] = None
    lora:            LoRAConfig = field(default_factory=LoRAConfig)
    dtype:           str        = "bfloat16"
    device:          str        = "cuda"
    # Ultatron injection
    ultatron_dim:    int        = 1024
    qwen_hidden_dim: int        = 3584
    projector_mid:   int        = 2048
    # Generation
    max_new_tokens:  int        = 8192
    max_sam2_calls:  int        = 3
    temperature:     float      = 1.0
    do_sample:       bool       = True

    @classmethod
    def from_dict(cls, d: dict) -> "StudentConfig":
        lora_d = d.pop("lora", {})
        obj = cls(**{k: v for k, v in d.items() if hasattr(cls, k)})
        if lora_d:
            obj.lora = LoRAConfig(**{k: v for k, v in lora_d.items() if hasattr(LoRAConfig, k)})
        return obj

    @property
    def torch_dtype(self) -> torch.dtype:
        _m = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
        return _m[self.dtype]


class StudentModel(nn.Module):
    """
    Qwen2.5-VL 7B student with LoRA, Ultatron token injection, and tool routing.

    Parameters
    ----------
    cfg            : StudentConfig
    projector      : pre-built UltatronProjector (shares object with GRPO trainer)
    tool_registry  : vlm.tools.registry.ToolRegistry instance (optional at init)
    load_ref_model : if True, keep a frozen copy of the initial LLM for KL penalty
    """

    def __init__(
        self,
        cfg:            StudentConfig,
        projector:      "UltatronProjector",  # noqa: F821
        tool_registry:  Optional[Any] = None,
        load_ref_model: bool = True,
    ):
        super().__init__()
        self.cfg           = cfg
        self.projector     = projector
        self.tool_registry = tool_registry

        self._qwen_model, self._processor = self._load_qwen(cfg)
        self._apply_lora(cfg.lora)

        self.ref_model: Optional[nn.Module] = None
        if load_ref_model:
            self.ref_model = self._make_ref_model()

    # ── Internal builders ────────────────────────────────────────────────────

    @staticmethod
    def _load_qwen(cfg: StudentConfig):
        """Load Qwen2.5-VL model and processor."""
        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.47 with Qwen2-VL support is required. "
                "Install with: pip install 'transformers>=4.47'"
            ) from exc

        dtype = cfg.torch_dtype
        kwargs: Dict[str, Any] = dict(
            torch_dtype=dtype,
            device_map="auto",
        )
        if cfg.hf_cache_dir:
            kwargs["cache_dir"] = cfg.hf_cache_dir

        log.info(f"Loading Qwen2.5-VL from {cfg.model_name!r} (dtype={cfg.dtype})")
        model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_name, **kwargs)

        # Freeze the visual encoder — only LLM (LoRA) + projector are trained
        if hasattr(model, "visual"):
            for p in model.visual.parameters():
                p.requires_grad_(False)
            log.info("  Qwen ViT frozen.")

        processor = AutoProcessor.from_pretrained(
            cfg.model_name,
            **({"cache_dir": cfg.hf_cache_dir} if cfg.hf_cache_dir else {}),
        )

        # Register SAM2 observation special tokens
        new_toks = [TOOL_CALL_START, TOOL_CALL_END, OBS_START, OBS_END,
                    "<seg_0>", "<seg_1>"]
        n_added = processor.tokenizer.add_tokens(new_toks, special_tokens=True)
        if n_added > 0:
            model.resize_token_embeddings(len(processor.tokenizer))
            log.info(f"  Added {n_added} special tokens; embedding table resized.")

        return model, processor

    def _apply_lora(self, lora_cfg: LoRAConfig):
        """Wrap the Qwen LLM backbone in LoRA adapters."""
        try:
            from peft import LoraConfig as PeftLoraConfig, get_peft_model, TaskType
        except ImportError as exc:
            raise ImportError(
                "peft is required for LoRA. Install with: pip install peft"
            ) from exc

        peft_cfg = PeftLoraConfig(
            r               = lora_cfg.r,
            lora_alpha      = lora_cfg.lora_alpha,
            lora_dropout    = lora_cfg.lora_dropout,
            target_modules  = lora_cfg.target_modules,
            bias            = lora_cfg.bias,
            task_type       = TaskType.CAUSAL_LM,
        )
        self._qwen_model = get_peft_model(self._qwen_model, peft_cfg)
        n_trainable = sum(p.numel() for p in self._qwen_model.parameters() if p.requires_grad)
        n_total     = sum(p.numel() for p in self._qwen_model.parameters())
        log.info(f"  LoRA applied: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M params trainable "
                 f"({100*n_trainable/n_total:.2f}%)")

    def _make_ref_model(self) -> nn.Module:
        """Frozen copy of the base LLM for KL divergence penalty."""
        try:
            from peft import get_base_model
            base = get_base_model(self._qwen_model)
        except Exception:
            base = self._qwen_model
        ref = copy.deepcopy(base)
        for p in ref.parameters():
            p.requires_grad_(False)
        ref.eval()
        log.info("  Reference model (frozen π_ref) created for KL penalty.")
        return ref

    # ── Trainable parameters ─────────────────────────────────────────────────

    def trainable_parameters(self):
        """Yields (name, param) for LoRA + projector parameters only."""
        for name, param in self._qwen_model.named_parameters():
            if param.requires_grad:
                yield name, param
        for name, param in self.projector.named_parameters():
            yield f"projector.{name}", param

    # ── Forward (training) ───────────────────────────────────────────────────

    def forward(
        self,
        input_ids:        torch.Tensor,           # (B, L) token IDs (with obs tokens included)
        attention_mask:   torch.Tensor,           # (B, L)
        pixel_values:     Optional[torch.Tensor], # (B, 3, H, W) for Qwen ViT
        image_grid_thw:   Optional[torch.Tensor], # Qwen-specific image grid metadata
        patch_tokens:     Optional[torch.Tensor], # (B, N, 1024) Ultatron image tokens
        tube_tokens:      Optional[torch.Tensor], # (B, T, 1024) Ultatron video tokens
        labels:           Optional[torch.Tensor] = None,  # (B, L) — -100 for masked positions
        ultatron_position: str = "prefix",        # "prefix" | "interleave"
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward with Ultatron token injection.

        Ultatron tokens are prepended as a "prefix" into the inputs_embeds before
        Qwen's own visual tokens, following the concat strategy from the plan.

        Returns dict with at least:
          logits     : (B, L_full, vocab_size)
          loss       : scalar (if labels provided, else None)
          log_probs  : (B, L) per-token log-probabilities (for GRPO)
        """
        # 1. Project Ultatron tokens
        ultatron_embs = None
        if patch_tokens is not None or tube_tokens is not None:
            ultatron_embs = self.projector.project(patch_tokens, tube_tokens)  # (B, Nu, D)

        # 2. Get Qwen's token embeddings
        embed_layer = self._qwen_model.get_input_embeddings()
        token_embs  = embed_layer(input_ids)  # (B, L, D)

        # 3. Build visual prefix if Ultatron tokens exist
        if ultatron_embs is not None and ultatron_position == "prefix":
            # Cast to same dtype
            ultatron_embs = ultatron_embs.to(dtype=token_embs.dtype)
            # Prepend as prefix (before text tokens); attention mask extended with 1s
            B, Nu, D = ultatron_embs.shape
            prefix_mask = torch.ones(B, Nu, dtype=attention_mask.dtype, device=attention_mask.device)
            inputs_embeds   = torch.cat([ultatron_embs, token_embs], dim=1)
            attention_mask  = torch.cat([prefix_mask, attention_mask], dim=1)
            # Shift labels accordingly (prefix positions are not predicted)
            if labels is not None:
                prefix_labels = torch.full((B, Nu), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            inputs_embeds = token_embs

        # 4. Build Qwen visual inputs (if pixel_values provided)
        vision_kwargs: Dict[str, Any] = {}
        if pixel_values is not None:
            vision_kwargs["pixel_values"]   = pixel_values
        if image_grid_thw is not None:
            vision_kwargs["image_grid_thw"] = image_grid_thw

        # 5. Qwen LLM forward
        outputs = self._qwen_model(
            inputs_embeds  = inputs_embeds,
            attention_mask = attention_mask,
            labels         = labels,
            **vision_kwargs,
        )

        logits    = outputs.logits       # (B, L_full, V)
        loss      = outputs.loss
        log_probs = torch.log_softmax(logits, dim=-1)

        return {"logits": logits, "loss": loss, "log_probs": log_probs}

    # ── Reference model forward (no grad) ───────────────────────────────────

    @torch.no_grad()
    def ref_log_probs(
        self,
        input_ids:       torch.Tensor,
        attention_mask:  torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute per-token log probabilities under the frozen reference model π_ref.
        Used by the GRPO trainer for the KL penalty term.
        """
        if self.ref_model is None:
            raise RuntimeError("Reference model was not created (load_ref_model=False).")
        ref_out = self.ref_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        return torch.log_softmax(ref_out.logits, dim=-1)

    # ── Generation with tool interception ────────────────────────────────────

    @torch.no_grad()
    def generate_with_tools(
        self,
        prompt:           str,
        image:            Optional[Any] = None,           # PIL Image or path
        patch_tokens:     Optional[torch.Tensor] = None,
        tube_tokens:      Optional[torch.Tensor] = None,
        return_trajectory: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate a full iMCoT trajectory.

        The model generates token-by-token; when a complete <tool_call>…</tool_call>
        block is detected, the ToolRegistry executes SAM2 and the 2 SAMTok
        observation tokens are appended with loss_mask=0 before generation resumes.

        Returns
        -------
        dict with:
          text          : str  — full decoded trajectory
          token_ids     : List[int]
          loss_mask     : List[int]  — 1 = model-generated, 0 = observation
          tool_calls    : List[dict] — each {"tool": ..., "bbox": ..., "tokens": [...]}
          answer        : str  — content inside <answer>…</answer> tags
        """
        if self.tool_registry is None:
            raise RuntimeError(
                "tool_registry must be set on StudentModel before calling generate_with_tools."
            )

        device    = next(self._qwen_model.parameters()).device
        tokenizer = self._processor.tokenizer

        # Build prompt with image
        messages = self._build_messages(prompt, image)
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Tokenize
        inputs = self._processor(
            text=[text_prompt],
            images=[image] if image is not None else None,
            return_tensors="pt",
        ).to(device)

        # Prefill with Ultatron tokens
        ultatron_prefix = None
        if patch_tokens is not None or tube_tokens is not None:
            ultatron_prefix = self.projector.project(
                patch_tokens.to(device) if patch_tokens is not None else None,
                tube_tokens.to(device)  if tube_tokens  is not None else None,
            )  # (1, Nu, D)

        # Agentic generation loop
        all_token_ids: List[int] = []
        all_loss_mask: List[int] = []
        all_tool_calls: List[dict] = []
        full_text = ""
        sam2_call_count = 0

        # Stream tokens from the model, detect tool calls
        generated_ids = list(inputs["input_ids"][0].cpu().numpy())
        current_text  = ""
        in_tool_call  = False
        tool_buffer   = ""

        max_gen = self.cfg.max_new_tokens
        for _step in range(max_gen):
            # Forward to get next token logits
            inp_ids = torch.tensor([generated_ids], device=device)
            attn_mask = torch.ones_like(inp_ids)

            # Simplified generation — in practice use model.generate with custom stopping
            with torch.autocast(device_type="cuda", dtype=self.cfg.torch_dtype):
                out = self._qwen_model(
                    input_ids=inp_ids,
                    attention_mask=attn_mask,
                )
            next_token_logits = out.logits[0, -1, :]
            if self.cfg.do_sample and self.cfg.temperature > 0:
                probs = torch.softmax(next_token_logits / self.cfg.temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                next_token = next_token_logits.argmax().item()

            # Append to sequence
            generated_ids.append(next_token)
            all_token_ids.append(next_token)
            all_loss_mask.append(1)  # model-generated token

            tok_str = tokenizer.decode([next_token], skip_special_tokens=False)
            current_text += tok_str

            # Detect EOS
            if next_token == tokenizer.eos_token_id:
                break

            # Detect tool call block
            if TOOL_CALL_START in current_text and not in_tool_call:
                in_tool_call = True
                tool_buffer  = current_text[current_text.rfind(TOOL_CALL_START) + len(TOOL_CALL_START):]
                current_text = ""
                continue

            if in_tool_call:
                if TOOL_CALL_END in current_text:
                    # Extract tool call JSON
                    raw_json = tool_buffer[:tool_buffer.find(TOOL_CALL_END)]
                    in_tool_call = False
                    tool_buffer  = ""
                    current_text = ""

                    if sam2_call_count < self.cfg.max_sam2_calls:
                        obs_tokens, tool_info = self._execute_tool(raw_json, image)
                        sam2_call_count += 1
                        all_tool_calls.append(tool_info)

                        # Append obs tokens with loss_mask=0
                        obs_ids = [tokenizer.convert_tokens_to_ids(t) for t in obs_tokens]
                        generated_ids.extend(obs_ids)
                        all_token_ids.extend(obs_ids)
                        all_loss_mask.extend([0] * len(obs_ids))
                else:
                    tool_buffer += tok_str

            full_text = tokenizer.decode(all_token_ids, skip_special_tokens=False)

        # Extract final answer
        answer = self._extract_answer(full_text)

        return {
            "text":       full_text,
            "token_ids":  all_token_ids,
            "loss_mask":  all_loss_mask,
            "tool_calls": all_tool_calls,
            "answer":     answer,
        }

    def _execute_tool(
        self, raw_json: str, image: Optional[Any]
    ) -> Tuple[List[str], dict]:
        """Parse the tool call JSON and dispatch to the ToolRegistry."""
        try:
            call = json.loads(raw_json.strip())
        except json.JSONDecodeError:
            log.warning(f"Could not parse tool call JSON: {raw_json!r}")
            return [], {"error": "parse_failed", "raw": raw_json}

        obs_tokens, info = self.tool_registry.dispatch(call, image=image)
        return obs_tokens, info

    @staticmethod
    def _build_messages(prompt: str, image: Optional[Any]) -> List[dict]:
        """Build Qwen2.5-VL chat messages list."""
        content: List[dict] = []
        if image is not None:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract the content inside <answer>…</answer> tags."""
        m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        return m.group(1).strip() if m else text.strip()

    # ── Checkpoint helpers ───────────────────────────────────────────────────

    def save(self, output_dir: str):
        """Save LoRA adapters + projector weights."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        self._qwen_model.save_pretrained(str(out / "lora_adapters"))
        torch.save(self.projector.state_dict(), str(out / "projector.pt"))
        log.info(f"StudentModel saved to {out}")

    @classmethod
    def load(
        cls,
        output_dir:     str,
        cfg:            StudentConfig,
        tool_registry:  Optional[Any] = None,
        load_ref_model: bool = False,
    ) -> "StudentModel":
        """Resume from a saved StudentModel directory."""
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError("peft required") from exc

        from vlm.projector import UltatronProjector
        proj = UltatronProjector(
            img_dim  = cfg.ultatron_dim,
            vid_dim  = cfg.ultatron_dim,
            qwen_dim = cfg.qwen_hidden_dim,
            mid_dim  = cfg.projector_mid,
        ).to(device=cfg.device, dtype=cfg.torch_dtype)

        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.cfg           = cfg
        instance.projector     = proj
        instance.tool_registry = tool_registry

        out = Path(output_dir)
        base_qwen, proc = cls._load_qwen(cfg)
        instance._qwen_model = PeftModel.from_pretrained(base_qwen, str(out / "lora_adapters"))
        instance._processor  = proc

        proj_ckpt = out / "projector.pt"
        if proj_ckpt.exists():
            instance.projector.load_state_dict(
                torch.load(proj_ckpt, map_location="cpu")
            )
        instance.ref_model = instance._make_ref_model() if load_ref_model else None
        log.info(f"StudentModel loaded from {out}")
        return instance
