"""Text-model loading and LoRA helpers shared across the pipeline."""
from __future__ import annotations

from typing import Any, List

from transformers import AutoConfig, AutoModelForCausalLM


QWEN35_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}


def is_qwen35_checkpoint(model_name_or_path: str) -> bool:
    """Return whether a checkpoint uses the dense Qwen3.5 architecture."""
    try:
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        model_type = str(getattr(config, "model_type", ""))
        text_type = str(getattr(getattr(config, "text_config", None), "model_type", ""))
        return model_type in QWEN35_MODEL_TYPES or text_type in QWEN35_MODEL_TYPES
    except Exception:
        raw = str(model_name_or_path).lower()
        normalized = raw.replace("_", "").replace("-", "")
        return "qwen3.5" in raw or "qwen35" in normalized


def load_text_causal_lm(model_name_or_path: str, **kwargs: Any):
    """Load the text backbone and avoid retaining Qwen3.5's vision encoder."""
    if is_qwen35_checkpoint(model_name_or_path):
        try:
            from transformers import Qwen3_5ForCausalLM
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3.5 requires a recent Transformers build exporting "
                "Qwen3_5ForCausalLM. Install the version documented in README.md."
            ) from exc
        return Qwen3_5ForCausalLM.from_pretrained(model_name_or_path, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def discover_lora_target_modules(model: Any) -> List[str]:
    """Find text LoRA targets, including Qwen3.5 Gated DeltaNet projections."""
    candidates = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [name for name in candidates if name in present]
    if not targets:
        raise RuntimeError(
            "No supported LoRA target modules found; refusing to train an empty adapter."
        )
    return targets
