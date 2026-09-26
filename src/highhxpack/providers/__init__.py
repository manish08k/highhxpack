"""Optional LLM providers.  None is required for core memory functionality."""

from __future__ import annotations

from highhxpack.exceptions import ConfigurationError
from highhxpack.providers.base import LLMProvider
from highhxpack.providers.custom import CallableProvider
from highhxpack.providers.ollama import DEFAULT_OLLAMA_MODEL, DEFAULT_OLLAMA_URL, OllamaProvider
from highhxpack.providers.openai import DEFAULT_OPENAI_MODEL, OpenAIProvider

__all__ = [
    "LLM_PROVIDERS",
    "CallableProvider",
    "LLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "create_llm_provider",
]

LLM_PROVIDERS = ("none", "ollama", "openai")


def create_llm_provider(
    name: str,
    *,
    model: str | None = None,
    ollama_url: str | None = None,
    openai_base_url: str | None = None,
    api_key: str | None = None,
    timeout: float | None = None,
) -> LLMProvider | None:
    """Build an LLM provider by name; ``"none"`` returns ``None``."""
    key = name.strip().lower()
    if key == "none":
        return None
    kwargs: dict[str, float] = {"timeout": timeout} if timeout is not None else {}
    if key == "ollama":
        return OllamaProvider(
            model or DEFAULT_OLLAMA_MODEL, base_url=ollama_url or DEFAULT_OLLAMA_URL, **kwargs
        )
    if key == "openai":
        return OpenAIProvider(
            model or DEFAULT_OPENAI_MODEL, api_key=api_key, base_url=openai_base_url, **kwargs
        )
    raise ConfigurationError(
        f"Unknown LLM provider: {name!r}",
        hint=f"Use one of: {', '.join(LLM_PROVIDERS)}, or pass an LLMProvider instance.",
    )
