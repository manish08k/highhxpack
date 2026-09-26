"""Wrap any Python callable as an LLM provider."""

from __future__ import annotations

from collections.abc import Callable

from highhxpack.exceptions import ProviderError
from highhxpack.providers.base import LLMProvider


class CallableProvider(LLMProvider):
    """Adapt ``fn(prompt) -> str`` to the :class:`LLMProvider` interface.

    When a system prompt is given it is prepended to the prompt, separated by a
    blank line.  Exceptions raised by ``fn`` are wrapped in :class:`ProviderError`.

    Example::

        provider = CallableProvider(lambda prompt: my_client.generate(prompt))
    """

    def __init__(self, fn: Callable[[str], str], *, name: str = "custom"):
        if not callable(fn):
            raise TypeError("fn must be callable")
        self._fn = fn
        self.name = name

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_output: bool = False,
        temperature: float = 0.0,
    ) -> str:
        full_prompt = f"{system}\n\n{prompt}" if system else prompt
        try:
            result = self._fn(full_prompt)
        except Exception as exc:
            raise ProviderError(
                f"Custom provider {self.name!r} raised {type(exc).__name__}.", reason=str(exc)
            ) from exc
        if not isinstance(result, str):
            raise ProviderError(
                f"Custom provider {self.name!r} returned {type(result).__name__}, expected str."
            )
        return result
