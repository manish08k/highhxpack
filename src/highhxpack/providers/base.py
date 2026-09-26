"""LLM provider interface and a minimal, dependency-free JSON-over-HTTP client."""

from __future__ import annotations

import contextlib
import json
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from highhxpack.exceptions import ConfigurationError, ProviderError
from highhxpack.utils.logging import redact

DEFAULT_TIMEOUT_SECONDS = 60.0
_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


class LLMProvider(ABC):
    """A text-generation model.

    LLMs are optional in HighHXPack: they are only used for LLM-based extraction
    and summarization.  Implement :meth:`complete` to plug in any model.
    """

    #: Identifier shown in logs and statistics, e.g. ``"ollama:llama3.2"``.
    name: str = "llm"

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_output: bool = False,
        temperature: float = 0.0,
    ) -> str:
        """Return the model's text response to ``prompt``.

        With ``json_output=True`` the provider should ask the model for a JSON
        document; callers still validate the result.
        """


def validate_base_url(url: str, *, provider: str) -> str:
    """Reject URLs that are not plain http(s) endpoints."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(
            f"Invalid {provider} base URL: {url!r}",
            hint="Use an http:// or https:// URL such as http://localhost:11434.",
        )
    if parsed.username or parsed.password:
        raise ConfigurationError(
            f"The {provider} base URL must not contain credentials.",
            hint="Pass credentials through the provider's API key setting instead.",
        )
    return url.rstrip("/")


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    provider: str,
    headers: dict[str, str] | None = None,
    unreachable_hint: str | None = None,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded JSON object response."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - scheme validated by validate_base_url
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = ""
        with contextlib.suppress(OSError):
            detail = exc.read(2048).decode("utf-8", "replace")
        exc.close()  # release the underlying connection
        raise ProviderError(
            f"{provider} returned HTTP {exc.code}.",
            reason=redact(detail) or exc.reason,
            hint=unreachable_hint if exc.code == 404 else None,
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ProviderError(
            f"Could not reach {provider} at {url}.",
            reason=redact(str(getattr(exc, "reason", exc))),
            hint=unreachable_hint,
        ) from exc
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise ProviderError(f"{provider} response exceeded {_MAX_RESPONSE_BYTES} bytes.")
    try:
        decoded = json.loads(raw)
    except ValueError as exc:
        raise ProviderError(f"{provider} returned invalid JSON.", reason=str(exc)) from exc
    if not isinstance(decoded, dict):
        raise ProviderError(f"{provider} returned an unexpected response shape.")
    return decoded
