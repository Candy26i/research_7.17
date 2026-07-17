"""Teacher client abstraction.

A TeacherClient takes a list of OpenAI-style chat messages and returns
a normalized TeacherResponse. Concrete implementations:

  - AnthropicTeacherClient (Claude)
  - OpenAITeacherClient    (GPT-4o, GPT-4-turbo, etc.)
  - DeepSeekTeacherClient  (DeepSeek V4 / chat / reasoner)

Switching is done via build_teacher_client(provider, model, ...).
Each generated SFT sample carries provider + model in its metadata so
downstream comparison studies can group by teacher identity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TeacherResponse:
    text: str
    provider: str
    model: str
    raw: Dict[str, Any] = field(default_factory=dict)
    cached: bool = False


class TeacherClient:
    provider: str = "abstract"

    def __init__(self, model: str, timeout: int = 120, max_retries: int = 3) -> None:
        self.model = model
        self.timeout = int(timeout)
        self.max_retries = int(max_retries)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> TeacherResponse:
        raise NotImplementedError


def build_teacher_client(
    provider: str,
    model: str,
    timeout: int = 120,
    max_retries: int = 3,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> TeacherClient:
    p = provider.strip().lower()
    if p == "anthropic" or p == "claude":
        from .anthropic_client import AnthropicTeacherClient
        return AnthropicTeacherClient(
            model=model, timeout=timeout, max_retries=max_retries, api_key=api_key
        )
    if p == "openai" or p == "gpt":
        from .openai_client import OpenAITeacherClient
        return OpenAITeacherClient(
            model=model, timeout=timeout, max_retries=max_retries,
            api_key=api_key, base_url=base_url,
        )
    if p == "deepseek":
        from .deepseek_client import DeepSeekTeacherClient
        return DeepSeekTeacherClient(
            model=model, timeout=timeout, max_retries=max_retries,
            api_key=api_key, base_url=base_url,
        )
    raise ValueError(f"Unknown teacher provider: {provider}")