"""Ollama (local LLM server) provider.

Uses Ollama's HTTP API through the standard library, so no extra package is
needed.  Data is sent only to the configured ``base_url`` (``localhost`` by
default).
"""

from __future__ import annotations

from typing import Any

from highhxpack.exceptions import ProviderError
from highhxpack.providers.base import (
    DEFAULT_TIMEOUT_SECONDS,
    LLMProvider,
    post_json,
    validate_base_url,
)

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"


class OllamaProvider(LLMProvider):
    """Chat completion through a running Ollama server."""

    def __init__(
        self,
        model: str = DEFAULT_OLLAMA_MODEL,
        *,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.model = model
        self.base_url = validate_base_url(base_url, provider="Ollama")
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_output: bool = False,
        temperature: float = 0.0,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_output:
            payload["format"] = "json"
        data = post_json(
            f"{self.base_url}/api/chat",
            payload,
            timeout=self.timeout,
            provider="Ollama",
            unreachable_hint=(
                f"Start Ollama (`ollama serve`) and pull the model (`ollama pull {self.model}`)."
            ),
        )
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ProviderError("Ollama returned a response without message content.")
        return content
