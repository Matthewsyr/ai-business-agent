from __future__ import annotations

import sys
import types
from typing import Any

from rag.openai_clients import OpenAIEmbeddingModel, OpenAILLMClient


class FakeOpenAI:
    last_kwargs: dict[str, Any] | None = None
    last_client: "FakeOpenAI" | None = None

    def __init__(self, **kwargs: Any) -> None:
        FakeOpenAI.last_kwargs = kwargs
        FakeOpenAI.last_client = self
        self.embedding_calls: list[dict[str, Any]] = []
        self.chat_calls: list[dict[str, Any]] = []
        self.embeddings = types.SimpleNamespace(create=self._create_embedding)
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create_chat_completion)
        )

    def _create_embedding(self, **kwargs: Any) -> Any:
        self.embedding_calls.append(kwargs)
        return types.SimpleNamespace(
            data=[
                types.SimpleNamespace(index=1, embedding=[0.0, 1.0]),
                types.SimpleNamespace(index=0, embedding=[1.0, 0.0]),
            ]
        )

    def _create_chat_completion(self, **kwargs: Any) -> Any:
        self.chat_calls.append(kwargs)
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content="## Answer\nCited claim [1]")
                )
            ]
        )


def install_fake_openai(monkeypatch: Any) -> None:
    module = types.ModuleType("openai")
    module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)


def test_openai_embedding_model_uses_sdk_lazily_and_preserves_order(monkeypatch: Any) -> None:
    install_fake_openai(monkeypatch)

    model = OpenAIEmbeddingModel(
        model="embedding-test",
        api_key="key",
        base_url="https://example.test/v1",
        dimensions=2,
    )
    vectors = model.embed_batch(["first", "second"])

    assert FakeOpenAI.last_kwargs == {"api_key": "key", "base_url": "https://example.test/v1"}
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert FakeOpenAI.last_client is not None
    assert FakeOpenAI.last_client.embedding_calls == [
        {"model": "embedding-test", "input": ["first", "second"], "dimensions": 2}
    ]


def test_openai_llm_client_calls_chat_completions(monkeypatch: Any) -> None:
    install_fake_openai(monkeypatch)

    client = OpenAILLMClient(model="chat-test", temperature=0.4)
    answer = client.generate("Write markdown", system_prompt="Use citations", temperature=0.1)

    assert answer == "## Answer\nCited claim [1]"
    assert FakeOpenAI.last_client is not None
    assert FakeOpenAI.last_client.chat_calls == [
        {
            "model": "chat-test",
            "messages": [
                {"role": "system", "content": "Use citations"},
                {"role": "user", "content": "Write markdown"},
            ],
            "temperature": 0.1,
        }
    ]
