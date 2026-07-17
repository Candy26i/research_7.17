"""Chat-template normalization for non-thinking tool-calling training.

Why this exists
---------------
Qwen3.5 checkpoints ship the *think* variant of the chat template: with
``add_generation_prompt=True`` and no ``enable_thinking`` kwarg it pre-writes
an OPEN ``<think>\n`` tag at the end of the prompt. TRL's tool-call parser
(`parse_response` with the qwen3_5 response template/schema) then treats the
entire completion as ``reasoning_content`` until a ``</think>`` appears. Our
manager is SFT'd in non-thinking format (it never emits ``</think>``), so any
render site that does not receive ``enable_thinking=False`` silently swallows
every ``<tool_call>`` block into reasoning: tool call rate reads exactly 0 and
the reward collapses, with no error anywhere.

Passing ``chat_template_kwargs={"enable_thinking": False}`` is not enough on
its own: it only works if *every* render site inside the installed TRL /
transformers / vLLM forwards the kwarg. Flipping the template so its DEFAULT
is the closed think block removes the failure mode at the source.

We swap to TRL's canonical ``qwen3_5_nothink`` template (byte-identical to the
shipped think template except for the ``enable_thinking`` default) whenever it
is available, because TRL recognizes its own template strings when picking the
response schema and the prefix-preserving training template. Only when TRL's
constants are unavailable do we fall back to a minimal in-place patch of the
default branch.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

_PROBE_MESSAGES = [
    {"role": "system", "content": "probe"},
    {"role": "user", "content": "probe"},
]


def generation_prompt_opens_think(
    tok: Any,
    chat_template_kwargs: Optional[Dict[str, Any]] = None,
) -> bool:
    """True if the rendered generation prompt ends with an OPEN <think> tag."""
    rendered = tok.apply_chat_template(
        _PROBE_MESSAGES,
        add_generation_prompt=True,
        tokenize=False,
        **(chat_template_kwargs or {}),
    )
    return rendered.rstrip().endswith("<think>")


def ensure_nothink_chat_template(tok: Any, tag: str = "") -> str:
    """Make the tokenizer's chat template default to the CLOSED think block.

    Returns one of:
      - "already-nothink":   default generation prompt already closes <think>
      - "swapped-trl-nothink": replaced with TRL's canonical qwen3_5_nothink
        template (preferred: TRL recognizes it for response-schema and
        training-template selection)
      - "patched-default":   flipped the `enable_thinking` default in place
        (fallback when TRL's template constants are unavailable)
      - "unknown-think-template": the template opens <think> by default but
        could not be normalized — callers should fail fast before training.

    Explicit ``enable_thinking=True`` still enables thinking after either
    normalization, so this only changes the *default*.
    """
    prefix = f"[{tag}] " if tag else ""
    if tok.chat_template is None or not generation_prompt_opens_think(tok):
        return "already-nothink"

    # Preferred: swap to TRL's canonical nothink template so TRL's exact
    # string matching (add_response_schema / get_training_chat_template)
    # keeps working.
    try:
        from trl.chat_template_utils import (  # type: ignore
            qwen3_5_nothink_chat_template,
            qwen3_5_think_chat_template,
        )
        if tok.chat_template == qwen3_5_think_chat_template:
            tok.chat_template = qwen3_5_nothink_chat_template
            if not generation_prompt_opens_think(tok):
                print(f"{prefix}chat template: think -> nothink (TRL canonical)")
                return "swapped-trl-nothink"
    except ImportError:
        pass

    # Fallback: flip the default of the `enable_thinking` switch in place.
    # think templates use `enable_thinking is defined and enable_thinking is
    # false` to select the closed block; rewriting the condition so that an
    # UNDEFINED enable_thinking also selects it preserves explicit True/False
    # behavior while changing only the default.
    patched = tok.chat_template.replace(
        "enable_thinking is defined and enable_thinking is false",
        "enable_thinking is not defined or enable_thinking is false",
    )
    if patched != tok.chat_template:
        original = tok.chat_template
        tok.chat_template = patched
        if not generation_prompt_opens_think(tok):
            print(f"{prefix}chat template: patched enable_thinking default to nothink")
            return "patched-default"
        tok.chat_template = original

    print(
        f"{prefix}WARNING: chat template opens <think> by default and could "
        f"not be normalized; tool-call parsing will silently fail unless "
        f"every render site passes enable_thinking=False."
    )
    return "unknown-think-template"
