"""OpenAI (or OpenAI-compatible) chat provider.

Requires the optional ``openai`` package: ``pip install "highhxpack[openai]"``.
Using this provider sends the text being processed to the configured API.
"""

from __future__ import annotations

import os
from typing import Any

from highhxpack.exceptions import ConfigurationError, ProviderError, ProviderNotAvailableError
from highhxpack.providers.base import DEFAULT_TIMEOUT_SECONDS, LLMProvider, validate_base_url
from highhxpack.utils.logging import redact

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
API_KEY_ENV_VARS = ("HIGHHXPACK_OPENAI_API_KEY", "OPENAI_API_KEY")


def load_openai_client(
    api_key: str | None, base_url: str | None, timeout: float
) -> Any:  # pragma: no cover - exercised only with the optional dependency installed
    try:
        import openai
    except ImportError as exc:
        raise ProviderNotAvailableError(
            "The OpenAI provider requires the 'openai' package.",
            hint='Install it with: pip install "highhxpack[openai]"',
        ) from exc
    key = api_key or next((os.environ[v] for v in API_KEY_ENV_VARS if os.environ.get(v)), None)
    if not key:
        raise ConfigurationError(
            "No OpenAI API key configured.",
            hint="Set the OPENAI_API_KEY environment variable or pass api_key=...",
        )
    if base_url:
        base_url = validate_base_url(base_url, provider="OpenAI")
    return openai.OpenAI(api_key=key, base_url=base_url, timeout=timeout)


class OpenAIProvider(LLMProvider):
    """Chat completion through the OpenAI API (or a compatible endpoint)."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        client: Any = None,
    ):
        self.model = model
        self.name = f"openai:{model}"
        # The client is created eagerly so configuration errors surface immediately.
        self._client = client or load_openai_client(api_key, base_url, timeout)

    def __repr__(self) -> str:  # never include the API key
        return f"OpenAIProvider(model={self.model!r})"

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
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if json_output:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
        except Exception as exc:  # the SDK raises many exception types
            raise ProviderError(
                "OpenAI request failed.",
                reason=redact(f"{type(exc).__name__}: {exc}"),
                hint="Check the API key, model name and network connectivity.",
            ) from exc
        if not isinstance(content, str):
            raise ProviderError("OpenAI returned an empty response.")
        return content
