"""Manager system prompt + final-answer parsing.

The manager is a deliberation orchestrator with three cognitive specialist tools:
  - extractor_tool: information extraction
  - reasoner_tool: structured reasoning
  - verifier_tool: domain audit and error detection

Adaptive Deliberation Control (ADC) policy:
  - 0 to 3 tool calls allowed. Each tool may be called at most once.
  - After each tool result, output DRAFT_ANSWER_<TOKEN> before deciding to continue.
  - Stop calling tools when further help is unlikely to improve the draft answer.
  - Final answer ends with exactly one line: ANSWER_<TOKEN>
  - Manager MUST NOT emit tool-call JSON or XML in plain text content.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


def _label_to_token(label: str) -> str:
    s = str(label).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s.upper()


def _token_to_label(token: str, choices: Dict[str, str]) -> str:
    """Map ANSWER_<TOKEN> back to canonical choice key."""
    t = token.upper().strip()
    for k in choices.keys():
        if _label_to_token(k) == t:
            return k
    return token


# Regex for final ANSWER_ on last line
ANSWER_LASTLINE_RE_FOR_KEYS = re.compile(
    r"^\s*(?:answer\s*[:=\-]?\s*)?ANSWER_([A-Za-z0-9_]+)\b[^\w]*$",
    re.IGNORECASE,
)

# Regex for DRAFT_ANSWER_ anywhere in text (intermediate candidate answers)
DRAFT_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*DRAFT_ANSWER_([A-Za-z0-9_]+)\b",
    re.IGNORECASE,
)


def build_manager_system_prompt(
    label_keys: List[str],
    task_description: str = "",
    exploration_hint: str = "",
) -> str:
    """Build the manager's system prompt with ADC deliberation policy.

    Args:
        label_keys: choice keys for the current task (e.g. ["A","B","C","D"]).
        task_description: optional one-liner describing the task domain.
        exploration_hint: START-style hint injected after deliberation policy
            during GRPO training to encourage multi-tool exploration.
            Leave empty for evaluation / deployment.
    """
    answer_lines = "\n".join(f"  ANSWER_{_label_to_token(k)}" for k in label_keys)
    draft_lines  = "\n".join(f"  DRAFT_ANSWER_{_label_to_token(k)}" for k in label_keys)
    desc = task_description or "You are a manager agent solving a multiple-choice question."
    hint_block = f"\n{exploration_hint.strip()}\n" if exploration_hint.strip() else ""
    return (
        desc + "\n\n"
        "You have THREE cognitive specialist tools:\n"
        "  - extractor_tool: extracts key signals and structures relevant facts from the question.\n"
        "  - reasoner_tool: produces a structured reasoning scaffold (sub-questions, per-choice analysis).\n"
        "  - verifier_tool: identifies domain principles and audits reasoning for errors. "
        "Pass your current draft answer key via the current_draft argument "
        "(e.g. current_draft=\"B\") so it audits that specific hypothesis.\n\n"
        "Deliberation policy:\n"
        "  - You may call 0 to 3 tools total. Each tool may be used at most once.\n"
        "  - After reading each tool result, state your current best answer on a new line:\n"
        + draft_lines + "\n"
        "  - Then decide: only call another tool if it might change your draft answer.\n"
        "  - Stop when additional tools are unlikely to improve your answer.\n"
        "  - Reserve all three tools for genuinely hard cases where each adds new signal.\n"
        + hint_block + "\n"
        "Output rules:\n"
        "  - Use the native tool-calling interface. Do NOT write tool calls as text, XML, or JSON.\n"
        "  - In a turn where you call a tool, output exactly one DRAFT_ANSWER_ line \n"
        "    plus the native tool call, but NOT the final ANSWER_.\n"
        "  - When you are ready to submit your final answer (no more tools), end with exactly:\n"
        + answer_lines + "\n"
        "  - Do not provide free-form reasoning, explanations, JSON, or XML in text.\n"
        "  - A final turn contains exactly one ANSWER_ line and nothing else.\n"
        "  - Do not output <think> tags.\n"
    )


def build_manager_tool_schemas(binding_mode: str) -> List[Dict[str, Any]]:
    """OpenAI-style JSON schemas for the three manager tools.

    Single source of truth shared by manager SFT (rendered into the prompt so
    training matches rollout), evaluation, and any server-side tool wiring.
    GRPO training passes Python callables to TRL, which derives equivalent
    schemas from their signatures/docstrings — keep both in sync.
    """
    required = ["example_id"] if binding_mode == "argument" else []
    properties: Dict[str, Any] = (
        {
            "example_id": {
                "type": "integer",
                "description": "The current example ID from the user message.",
            }
        }
        if binding_mode == "argument"
        else {}
    )
    verifier_properties = dict(properties)
    verifier_properties["current_draft"] = {
        "type": "string",
        "description": "Your current draft answer key (e.g. \"B\") to audit.",
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "extractor_tool",
                "description": "Extract decision-relevant factual signals from the question and context.",
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reasoner_tool",
                "description": "Produce a structured reasoning scaffold for the choices.",
                "parameters": {"type": "object", "properties": properties, "required": required},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "verifier_tool",
                "description": "Identify relevant domain principles and audit the reasoning for logical or computational errors. Pass your current draft answer via current_draft.",
                "parameters": {"type": "object", "properties": verifier_properties, "required": required},
            },
        },
    ]


def build_manager_user_message(
    example_id: int,
    question: str,
    context: str,
    choices: Dict[str, str],
    binding_mode: str = "environment",
) -> str:
    lines = [f"Example ID: {example_id}", "", f"Question:\n{question}", ""]
    if choices:
        choices_block = "Choices:\n" + "\n".join(f"  {k}. {v}" for k, v in choices.items())
        lines.append(choices_block)
        lines.append("")
    if context:
        lines.append(f"Context:\n{context}")
        lines.append("")
    if binding_mode == "argument":
        lines.append(
            "If you call a tool, pass the current Example ID as the example_id argument. "
            "For verifier_tool, also pass your current draft answer key as current_draft."
        )
    else:
        lines.append(
            "If you call a tool, the current example is already bound — no example_id is needed. "
            "For verifier_tool, pass your current draft answer key as current_draft."
        )
    lines.append("")
    lines.append("If you do not call any tool, answer directly.")
    return "\n".join(lines)


# def parse_final_answer(text: str, choice_keys: List[str]) -> Optional[str]:
#     """Parse the final ANSWER_<TOKEN> line and map to a canonical choice key."""
#     if not text:
#         return None
#     lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
#     if not lines:
#         return None
#     m = ANSWER_LASTLINE_RE_FOR_KEYS.match(lines[-1])
#     if not m:
#         return None
#     token = m.group(1).upper()
#     for k in choice_keys:
#         if _label_to_token(k) == token:
#             return k
#     return None
def parse_final_answer(text: str, choice_keys: List[str]) -> Optional[str]:
    """Parse the final ANSWER_<TOKEN> and map to a canonical choice key."""
    if not text:
        return None
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = ANSWER_LASTLINE_RE_FOR_KEYS.search(ln)
        if not m:
            continue
        token = m.group(1).upper()
        for k in choice_keys:
            if _label_to_token(k) == token:
                return k
    return None


def parse_draft_answer(text: str, choice_keys: List[str]) -> Optional[str]:
    """Parse the LAST DRAFT_ANSWER_<TOKEN> from an assistant turn.

    Returns the most recent draft choice key, or None if not present.
    Used by the ADC reward function to track intermediate answer transitions.
    """
    if not text:
        return None
    matches = DRAFT_ANSWER_RE.findall(str(text))
    if not matches:
        return None
    token = matches[-1].upper()
    for k in choice_keys:
        if _label_to_token(k) == token:
            return k
    return None


def count_unpaired_tool_turns(
    completion: Any,
    choice_keys: List[str],
) -> "tuple[int, int]":
    """Return (n_tool_turns, n_unpaired): assistant turns carrying tool_calls,
    and how many of them lack a DRAFT_ANSWER_ line in the same turn.

    Per-TURN pairing is the semantically correct enforcement of "declare a
    draft before each tool call": a global draft count can be satisfied
    post-hoc after seeing tool results (see ADC_RESIDUAL_HOLES.md §2), which
    corrupts the W->C correction statistics.
    """
    if not isinstance(completion, list):
        return 0, 0
    n_turns = 0
    n_unpaired = 0
    for msg in completion:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        if not msg.get("tool_calls"):
            continue
        content = msg.get("content")
        if isinstance(content, list):
            text = " ".join(
                blk.get("text", "") for blk in content
                if isinstance(blk, dict) and "text" in blk
            )
        else:
            text = str(content or "")
        n_turns += 1
        if parse_draft_answer(text, choice_keys) is None:
            n_unpaired += 1
    return n_turns, n_unpaired


def extract_answer_sequence(
    completion: Any,
    choice_keys: List[str],
) -> List[Optional[str]]:
    """Extract the full sequence of candidate answers from a completion.

    Scans every assistant turn for DRAFT_ANSWER_; only the LAST assistant
    turn contributes the final ANSWER_. Returns (draft_0, ..., final) in
    chronological order. Used by the ADC reward function.

    Bare ANSWER_ lines in intermediate turns are deliberately IGNORED:
    counting them would (a) let the policy satisfy the missing-draft penalty
    without ever using the DRAFT_ format, and (b) let it dilute an early
    wrong draft by echoing the (post-tool-result) answer as extra entries in
    the anytime average — a small but farmable reward leak.
    """
    if not isinstance(completion, list):
        text = str(completion) if completion else ""
        result = []
        draft = parse_draft_answer(text, choice_keys)
        if draft is not None:
            result.append(draft)
        final = parse_final_answer(text, choice_keys)
        if final is not None:
            result.append(final)
        return result

    texts: List[str] = []
    for msg in completion:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            text = " ".join(
                blk.get("text", "") for blk in content
                if isinstance(blk, dict) and "text" in blk
            )
        else:
            text = str(content or "")
        texts.append(text)

    sequence: List[Optional[str]] = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        draft = parse_draft_answer(text, choice_keys)
        if draft is not None:
            sequence.append(draft)
        if i == len(texts) - 1:
            final = parse_final_answer(text, choice_keys)
            if final is not None:
                sequence.append(final)
    return sequence
