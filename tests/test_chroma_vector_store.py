from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Any

from rag.loader import Document
from rag.vector_store import JsonVectorStore


class FakeCollection:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}

    def get(self, ids: list[str]) -> dict[str, list[str]]:
        return {"ids": [item_id for item_id in ids if item_id in self.items]}

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> None:
        for item_id, document, metadata, embedding in zip(
            ids, documents, metadatas, embeddings, strict=True
        ):
            self.items[item_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embedding,
            }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
    ) -> dict[str, list[list[Any]]]:
        query = query_embeddings[0]
        ranked = sorted(
            self.items.values(),
            key=lambda item: math.dist(query, item["embedding"]),
        )[:n_results]
        return {
            "documents": [[item["document"] for item in ranked]],
            "metadatas": [[item["metadata"] for item in ranked]],
            "distances": [[math.dist(query, item["embedding"]) for item in ranked]],
        }

    def count(self) -> int:
        return len(self.items)


class FakePersistentClient:
    collection = FakeCollection()

    def __init__(self, path: str) -> None:
        self.path = path

    def get_or_create_collection(self, name: str) -> FakeCollection:
        return self.collection


def install_fake_chromadb(monkeypatch: Any) -> None:
    FakePersistentClient.collection = FakeCollection()
    module = types.ModuleType("chromadb")
    module.PersistentClient = FakePersistentClient
    monkeypatch.setitem(sys.modules, "chromadb", module)


def test_chroma_vector_store_upserts_with_json_compatible_ids(
    tmp_path: Path, monkeypatch: Any
) -> None:
    install_fake_chromadb(monkeypatch)
    from rag.chroma_store import ChromaVectorStore

    document = Document(
        text="Alpha market report",
        metadata={
            "source": "alpha.md",
            "chunk_id": 1,
            "tags": ["market"],
            "nested": {"a": 1},
            "none": None,
        },
    )
    store = ChromaVectorStore(tmp_path / "chroma")

    assert store.add_documents([document, document], [[1.0, 0.0], [1.0, 0.0]]) == 1
    assert store.add_documents([document], [[1.0, 0.0]]) == 0
    assert store.count() == 1

    expected_id = JsonVectorStore._item_id(document)
    item = FakePersistentClient.collection.items[expected_id]
    assert item["metadata"]["tags"] == '["market"]'
    assert item["metadata"]["nested"] == '{"a": 1}'
    assert "none" not in item["metadata"]


def test_chroma_vector_store_search_converts_distance_to_relevance(
    tmp_path: Path, monkeypatch: Any
) -> None:
    install_fake_chromadb(monkeypatch)
    from rag.chroma_store import ChromaVectorStore

    store = ChromaVectorStore(tmp_path / "chroma")
    near = Document(text="near result", metadata={"source": "near.md", "chunk_id": 1})
    far = Document(text="far result", metadata={"source": "far.md", "chunk_id": 1})
    store.add_documents([far, near], [[0.0, 1.0], [1.0, 0.0]])

    results = store.search([1.0, 0.0], top_k=2)

    assert [result.text for result in results] == ["near result", "far result"]
    assert results[0].score > results[1].score
