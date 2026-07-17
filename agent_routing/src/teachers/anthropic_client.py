"""Anthropic (Claude) teacher client."""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from .base import TeacherClient, TeacherResponse


class AnthropicTeacherClient(TeacherClient):
    provider = "anthropic"

    def __init__(
        self,
        model: str,
        timeout: int = 120,
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout, max_retries=max_retries)
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise ImportError(
                "Install `anthropic` to use AnthropicTeacherClient."
            ) from e
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set.")
        self._client = Anthropic(api_key=key, timeout=self.timeout)

    @staticmethod
    def _split_system(messages: List[Dict[str, str]]):
        system_parts = []
        non_system = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(str(m.get("content", "")))
            else:
                non_system.append({
                    "role": m.get("role", "user"),
                    "content": str(m.get("content", "")),
                })
        return "\n\n".join(system_parts), non_system

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> TeacherResponse:
        system, msgs = self._split_system(messages)
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    system=system,
                    messages=msgs,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                text_parts = []
                for block in resp.content:
                    if getattr(block, "type", "") == "text":
                        text_parts.append(block.text)
                text = "".join(text_parts).strip()
                return TeacherResponse(
                    text=text, provider=self.provider, model=self.model,
                    raw={"id": getattr(resp, "id", "")},
                )
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                continue
        raise RuntimeError(f"Anthropic chat failed after retries: {last_err}")