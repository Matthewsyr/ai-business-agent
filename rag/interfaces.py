from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from rag.loader import Document


@dataclass
class SearchResult:
    text: str
    metadata: dict[str, Any]
    score: float


@runtime_checkable
class EmbeddingModel(Protocol):
    def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class BatchEmbeddingModel(EmbeddingModel, Protocol):
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class VectorStore(Protocol):
    def add_documents(self, documents: list[Document], vectors: list[list[float]]) -> int: ...

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]: ...

    def count(self) -> int: ...


@runtime_checkable
class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str: ...
