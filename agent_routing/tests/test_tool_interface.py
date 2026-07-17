"""Regression tests for the manager <-> tool interface.

These cover the exact breakages that made GRPO/eval tool call rate read 0:
  1. Qwen3.5 renders native tool calls as <function=...>/<parameter=...> XML,
     not Qwen3's <tool_call>{JSON}</tool_call>; eval must parse both.
  2. HF chat templates require tool_call `arguments` to be a mapping — the
     Qwen3.5 template raises on the OpenAI wire format (a JSON string).
  3. Think-variant chat templates pre-open a `<think>` tag in the generation
     prompt by default, which makes the response parser swallow the whole
     completion (tool calls included) into reasoning_content.
"""
import sys
import types

import pytest

sys.path.insert(0, ".")

from src.manager.evolve import _normalize_tool_args, _tool_call_message as evolve_tool_call_message
from src.manager.reward import _has_plaintext_tool_artifacts
from src.pipeline.stages import _extract_manager_tool_calls, _tool_call_message as eval_tool_call_message


QWEN35_XML = (
    "<think>\nsome private reasoning\n</think>\n"
    "DRAFT_ANSWER_C\n"
    "<tool_call>\n<function=reasoner_tool>\n"
    "<parameter=example_id>\n42\n</parameter>\n"
    "</function>\n</tool_call>"
)
QWEN3_JSON = (
    'DRAFT_ANSWER_A\n<tool_call>\n'
    '{"name": "extractor_tool", "arguments": {"example_id": 7}}\n'
    "</tool_call>"
)


def test_extract_tool_calls_qwen35_xml():
    content, calls = _extract_manager_tool_calls(QWEN35_XML)
    assert calls == [{"name": "reasoner_tool", "arguments": {"example_id": 42}}]
    assert content == "DRAFT_ANSWER_C"


def test_extract_tool_calls_qwen3_json():
    content, calls = _extract_manager_tool_calls(QWEN3_JSON)
    assert calls == [{"name": "extractor_tool", "arguments": {"example_id": 7}}]
    assert content == "DRAFT_ANSWER_A"


def test_extract_tool_calls_string_param_value():
    text = (
        "<tool_call>\n<function=verifier_tool>\n"
        "<parameter=current_draft>\nB\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    _, calls = _extract_manager_tool_calls(text)
    assert calls == [{"name": "verifier_tool", "arguments": {"current_draft": "B"}}]


def test_tool_call_messages_carry_dict_arguments():
    # HF chat templates iterate `arguments` as a mapping; a JSON string raises
    # "Can only get item pairs from a mapping" in the Qwen3.5 template.
    msg = eval_tool_call_message("verifier_tool", {"current_draft": "B"}, "id1")
    assert isinstance(msg["tool_calls"][0]["function"]["arguments"], dict)
    msg2 = evolve_tool_call_message(
        "verifier_tool", 42, "id2", "argument", extra_args={"current_draft": "B"}
    )
    args2 = msg2["tool_calls"][0]["function"]["arguments"]
    assert args2 == {"example_id": 42, "current_draft": "B"}


def test_normalize_tool_args_converts_legacy_string_arguments():
    legacy = [{
        "role": "assistant",
        "tool_calls": [{
            "type": "function",
            "function": {"name": "verifier_tool", "arguments": '{"current_draft": "B"}'},
        }],
    }]
    out = _normalize_tool_args(legacy)
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"current_draft": "B"}


def test_reward_flags_qwen35_xml_artifacts():
    assert _has_plaintext_tool_artifacts("blah <function=verifier_tool> blah")
    assert _has_plaintext_tool_artifacts("<parameter=current_draft>\nB")
    assert not _has_plaintext_tool_artifacts("ANSWER_B")


# ---- nothink template normalization (fallback patch path) ----

_MINI_THINK_TEMPLATE = (
    "{%- for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor %}"
    "{%- if add_generation_prompt %}<|im_start|>assistant\n"
    "{%- if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n"
    "{%- else %}<think>\n{%- endif %}{%- endif %}"
)


class _StubTok:
    chat_template = _MINI_THINK_TEMPLATE

    def apply_chat_template(self, messages, add_generation_prompt=False,
                            tokenize=False, **kwargs):
        import jinja2
        return jinja2.Template(self.chat_template).render(
            messages=messages, add_generation_prompt=add_generation_prompt, **kwargs
        )


def test_ensure_nothink_patches_think_default():
    jinja2 = pytest.importorskip("jinja2")  # noqa: F841
    from src.utils.chat_template import (
        ensure_nothink_chat_template,
        generation_prompt_opens_think,
    )
    tok = _StubTok()
    assert generation_prompt_opens_think(tok)
    action = ensure_nothink_chat_template(tok)
    assert action in ("swapped-trl-nothink", "patched-default")
    assert not generation_prompt_opens_think(tok)
    # explicit thinking mode still available
    assert generation_prompt_opens_think(tok, {"enable_thinking": True})
    # explicit False still works
    assert not generation_prompt_opens_think(tok, {"enable_thinking": False})
