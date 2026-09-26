from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

import pytest

from highhxpack import (
    CallableEmbedding,
    CallableProvider,
    ConfigurationError,
    EmbeddingError,
    HashingEmbedding,
    ProviderError,
    ProviderNotAvailableError,
)
from highhxpack.embeddings import create_embedding_provider
from highhxpack.embeddings.base import check_vectors
from highhxpack.embeddings.local import SentenceTransformerEmbedding
from highhxpack.embeddings.providers import OllamaEmbedding, OpenAIEmbedding
from highhxpack.providers import OllamaProvider, OpenAIProvider, create_llm_provider
from highhxpack.providers.base import validate_base_url
from highhxpack.storage.vector import dot


class TestHashingEmbedding:
    def test_deterministic_unit_vectors(self) -> None:
        emb = HashingEmbedding(64)
        a, b = emb.embed(["I prefer Python", "I prefer Python"])
        assert a == b
        assert len(a) == 64
        assert dot(a, a) == pytest.approx(1.0)
        assert emb.name == "hashing-v1-64"

    def test_similarity_ordering(self) -> None:
        emb = HashingEmbedding()
        query = emb.embed_query("python programming")
        close = emb.embed_one("I love programming in Python")
        far = emb.embed_one("My cat sleeps all day")
        assert dot(query, close) > dot(query, far)

    def test_empty_text_is_zero_vector(self) -> None:
        assert set(HashingEmbedding(16).embed_one("the and of")) == {0.0}

    def test_dimension_bounds(self) -> None:
        with pytest.raises(ValueError, match="dimension"):
            HashingEmbedding(8)


class TestCheckVectors:
    @pytest.mark.parametrize(
        "vectors",
        [
            None,
            [[1.0]],
            [[1.0], [1.0, 2.0]],
            [["a"], ["b"]],
            [[]],
            ["ab", "cd"],
            [[float("inf")], [1.0]],
        ],
    )
    def test_rejects_malformed(self, vectors: Any) -> None:
        with pytest.raises(EmbeddingError):
            check_vectors(vectors, 2, provider="test")

    def test_normalizes(self) -> None:
        assert check_vectors([[3, 4]], 1, provider="t") == [[0.6, 0.8]]


class TestCallableAdapters:
    def test_callable_embedding(self) -> None:
        emb = CallableEmbedding(lambda texts: [[1.0, 1.0] for _ in texts], name="mine")
        assert emb.embed([]) == []
        assert emb.embed(["x"])[0] == pytest.approx([2**-0.5, 2**-0.5])

    def test_callable_embedding_errors(self) -> None:
        def boom(texts: list[str]) -> list[list[float]]:
            raise RuntimeError("down")

        with pytest.raises(EmbeddingError, match="RuntimeError"):
            CallableEmbedding(boom, name="x").embed(["a"])
        with pytest.raises(ValueError, match="name"):
            CallableEmbedding(lambda t: [], name="")
        with pytest.raises(TypeError):
            CallableEmbedding("nope", name="x")  # type: ignore[arg-type]

    def test_callable_provider(self) -> None:
        seen: list[str] = []

        def fn(prompt: str) -> str:
            seen.append(prompt)
            return "ok"

        provider = CallableProvider(fn, name="echo")
        assert provider.complete("hi", system="sys") == "ok"
        assert seen == ["sys\n\nhi"]
        with pytest.raises(ProviderError, match="returned int"):
            CallableProvider(lambda p: 1).complete("x")  # type: ignore[arg-type, return-value]
        with pytest.raises(ProviderError, match="raised"):
            CallableProvider(lambda p: 1 / 0).complete("x")  # type: ignore[arg-type, return-value]
        with pytest.raises(TypeError):
            CallableProvider(42)  # type: ignore[arg-type]


class _Handler(BaseHTTPRequestHandler):
    routes: ClassVar[dict[str, tuple[int, bytes]]] = {}
    requests: ClassVar[list[tuple[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers["Content-Length"])
        body = json.loads(self.rfile.read(length))
        type(self).requests.append((self.path, body))
        status, payload = type(self).routes.get(self.path, (404, b'{"error": "not found"}'))
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture
def server() -> Iterator[tuple[str, type[_Handler]]]:
    handler = type("Handler", (_Handler,), {"routes": {}, "requests": []})
    httpd = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", handler
    httpd.shutdown()
    httpd.server_close()
    thread.join()


class TestOllama:
    def test_chat(self, server: tuple[str, type[_Handler]]) -> None:
        url, handler = server
        handler.routes["/api/chat"] = (200, b'{"message": {"content": "hello"}}')
        provider = OllamaProvider("tiny", base_url=url)
        assert provider.complete("hi", system="be brief", json_output=True) == "hello"
        path, body = handler.requests[0]
        assert path == "/api/chat"
        assert body["format"] == "json"
        assert body["messages"][0] == {"role": "system", "content": "be brief"}

    def test_embeddings(self, server: tuple[str, type[_Handler]]) -> None:
        url, handler = server
        handler.routes["/api/embed"] = (200, b'{"embeddings": [[1, 0], [0, 2]]}')
        emb = OllamaEmbedding("e", base_url=url)
        assert emb.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
        assert emb.embed([]) == []

    def test_http_error_includes_hint(self, server: tuple[str, type[_Handler]]) -> None:
        url, _ = server
        with pytest.raises(ProviderError, match="HTTP 404") as info:
            OllamaProvider("missing-model", base_url=url).complete("x")
        assert info.value.hint is not None
        assert "ollama pull missing-model" in info.value.hint

    def test_invalid_responses(self, server: tuple[str, type[_Handler]]) -> None:
        url, handler = server
        handler.routes["/api/chat"] = (200, b"not json")
        with pytest.raises(ProviderError, match="invalid JSON"):
            OllamaProvider(base_url=url).complete("x")
        handler.routes["/api/chat"] = (200, b"[1, 2]")
        with pytest.raises(ProviderError, match="unexpected"):
            OllamaProvider(base_url=url).complete("x")
        handler.routes["/api/chat"] = (200, b'{"message": {}}')
        with pytest.raises(ProviderError, match="without message"):
            OllamaProvider(base_url=url).complete("x")
        handler.routes["/api/embed"] = (200, b'{"embeddings": [[1, 0]]}')
        with pytest.raises(EmbeddingError):
            OllamaEmbedding(base_url=url).embed(["a", "b"])

    def test_unreachable(self) -> None:
        # Port 9 (discard) on localhost is essentially never listening.
        with pytest.raises(ProviderError, match="Could not reach"):
            OllamaProvider(base_url="http://127.0.0.1:9", timeout=2).complete("x")


def test_validate_base_url() -> None:
    assert validate_base_url("http://localhost:11434/", provider="x") == "http://localhost:11434"
    for bad in ["ftp://host", "localhost:11434", "file:///etc/passwd", "http://user:pw@host"]:
        with pytest.raises(ConfigurationError):
            validate_base_url(bad, provider="x")


class _FakeOpenAI:
    """Mimics the parts of the openai client the providers use."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self
        self.embeddings = self

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("401 invalid api key sk-abcdefghijklmnop")
        if "messages" in kwargs:
            message = type("M", (), {"content": "answer"})
            choice = type("C", (), {"message": message})
            return type("R", (), {"choices": [choice]})
        items = [type("E", (), {"embedding": [1.0, 0.0]}) for _ in kwargs["input"]]
        return type("R", (), {"data": items})


class TestOpenAI:
    def test_chat_and_embeddings_with_fake_client(self) -> None:
        client = _FakeOpenAI()
        provider = OpenAIProvider("gpt-test", client=client)
        assert provider.complete("q", json_output=True) == "answer"
        assert client.calls[0]["response_format"] == {"type": "json_object"}
        assert OpenAIEmbedding("e", client=client).embed(["a"]) == [[1.0, 0.0]]
        assert "key" not in repr(provider).lower()

    def test_errors_are_wrapped_and_redacted(self) -> None:
        with pytest.raises(ProviderError) as info:
            OpenAIProvider(client=_FakeOpenAI(fail=True)).complete("q")
        assert "abcdefghijklmnop" not in str(info.value)
        with pytest.raises(EmbeddingError):
            OpenAIEmbedding(client=_FakeOpenAI(fail=True)).embed(["a"])

    def test_missing_package_or_key(self) -> None:
        try:
            import openai  # noqa: F401
        except ImportError:
            with pytest.raises(ProviderNotAvailableError, match=r"highhxpack\[openai\]"):
                OpenAIProvider()
        else:  # pragma: no cover - only when the optional dependency is installed
            with pytest.raises(ConfigurationError, match="API key"):
                OpenAIProvider()


def test_sentence_transformers_optional() -> None:
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(ProviderNotAvailableError, match="sentence-transformers"):
            SentenceTransformerEmbedding().embed(["x"])
    else:  # pragma: no cover
        pytest.skip("sentence-transformers is installed")


def test_sentence_transformers_with_injected_model() -> None:
    class Model:
        def encode(self, texts: list[str], normalize_embeddings: bool) -> list[list[float]]:
            return [[1.0, 0.0] for _ in texts]

    emb = SentenceTransformerEmbedding("m", model=Model())
    assert emb.embed(["a"]) == [[1.0, 0.0]]
    assert emb.name == "sentence-transformers:m"


def test_factories() -> None:
    assert create_embedding_provider("none") is None
    assert create_embedding_provider("hashing", model="128").name == "hashing-v1-128"  # type: ignore[union-attr]
    assert create_embedding_provider("ollama").name == "ollama:nomic-embed-text"  # type: ignore[union-attr]
    assert isinstance(
        create_embedding_provider("sentence-transformers"), SentenceTransformerEmbedding
    )
    with pytest.raises(ConfigurationError):
        create_embedding_provider("hashing", model="big")
    with pytest.raises(ConfigurationError):
        create_embedding_provider("hashing", model="4")
    with pytest.raises(ConfigurationError, match="Unknown embedding provider"):
        create_embedding_provider("magic")
    assert create_llm_provider("none") is None
    assert create_llm_provider("ollama", model="m").name == "ollama:m"  # type: ignore[union-attr]
    with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
        create_llm_provider("magic")
