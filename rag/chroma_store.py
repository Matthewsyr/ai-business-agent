from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag.interfaces import SearchResult
from rag.loader import Document
from rag.vector_store import stable_document_id

ScalarMetadataValue = str | int | float | bool


class ChromaVectorStore:
    def __init__(self, path: Path, collection_name: str = "documents") -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("Install chromadb to use ChromaVectorStore.") from exc

        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client: Any = chromadb.PersistentClient(path=str(self.path))
        self.collection: Any = self.client.get_or_create_collection(name=collection_name)

    def add_documents(self, documents: list[Document], vectors: list[list[float]]) -> int:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")
        if not documents:
            return 0

        ids: list[str] = []
        texts: list[str] = []
        metadatas: list[dict[str, ScalarMetadataValue]] = []
        embeddings: list[list[float]] = []
        seen_ids: set[str] = set()

        for document, vector in zip(documents, vectors, strict=True):
            item_id = stable_document_id(document)
            if item_id in seen_ids:
                continue
            seen_ids.add(item_id)
            ids.append(item_id)
            texts.append(document.text)
            metadatas.append(self._sanitize_metadata(document.metadata))
            embeddings.append(vector)

        if not ids:
            return 0

        existing_ids = self._existing_ids(ids)
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return sum(1 for item_id in ids if item_id not in existing_ids)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            return []
        payload = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        documents = self._first(payload.get("documents", []))
        metadatas = self._first(payload.get("metadatas", []))
        distances = self._first(payload.get("distances", []))

        results: list[SearchResult] = []
        for text, metadata, distance in zip(documents, metadatas, distances, strict=False):
            score = self._distance_to_score(distance)
            if score > 0:
                results.append(
                    SearchResult(
                        text=text or "",
                        metadata=dict(metadata or {}),
                        score=score,
                    )
                )
        return results

    def count(self) -> int:
        return int(self.collection.count())

    def _existing_ids(self, ids: list[str]) -> set[str]:
        payload = self.collection.get(ids=ids)
        return set(payload.get("ids", []))

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, ScalarMetadataValue]:
        sanitized: dict[str, ScalarMetadataValue] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            name = str(key)
            if isinstance(value, (str, bool, int, float)):
                sanitized[name] = value
                continue
            if isinstance(value, (list, tuple, dict)):
                sanitized[name] = json.dumps(value, ensure_ascii=False, default=str)
                continue
            sanitized[name] = str(value)
        return sanitized

    @staticmethod
    def _distance_to_score(distance: Any) -> float:
        try:
            numeric = float(distance)
        except (TypeError, ValueError):
            return 0.0
        if numeric < 0:
            return 1.0
        return 1.0 / (1.0 + numeric)

    @staticmethod
    def _first(items: Any) -> list[Any]:
        if not items:
            return []
        first = items[0]
        return first if isinstance(first, list) else items
