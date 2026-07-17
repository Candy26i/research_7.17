"""DeepSeek teacher client.

DeepSeek API is OpenAI-compatible. Default base_url = https://api.deepseek.com.
Default model strings:
  - deepseek-chat        (V4 / general chat)
  - deepseek-reasoner    (R1 / reasoning)
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from .base import TeacherClient, TeacherResponse


DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"


class DeepSeekTeacherClient(TeacherClient):
    provider = "deepseek"

    def __init__(
        self,
        model: str,
        timeout: int = 120,
        max_retries: int = 3,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(model=model, timeout=timeout, max_retries=max_retries)
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "Install `openai` to use DeepSeekTeacherClient (uses OpenAI-compatible API)."
            ) from e
        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise RuntimeError("DEEPSEEK_API_KEY not set.")
        url = base_url or os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_DEFAULT_BASE_URL)
        self._client = OpenAI(api_key=key, base_url=url, timeout=self.timeout)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> TeacherResponse:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                text = (resp.choices[0].message.content or "").strip()
                return TeacherResponse(
                    text=text, provider=self.provider, model=self.model,
                    raw={"id": getattr(resp, "id", "")},
                )
            except Exception as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(min(2 ** attempt, 8))
                continue
        raise RuntimeError(f"DeepSeek chat failed after retries: {last_err}")