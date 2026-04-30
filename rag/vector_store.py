from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.loader import Document


@dataclass
class SearchResult:
    text: str
    metadata: dict[str, Any]
    score: float


class JsonVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: list[dict[str, Any]] = []
        self._load()

    def add_documents(self, documents: list[Document], vectors: list[list[float]]) -> int:
        if len(documents) != len(vectors):
            raise ValueError("documents and vectors must have the same length")
        known_ids = {item["id"] for item in self._items}
        added = 0
        for document, vector in zip(documents, vectors, strict=True):
            item_id = self._item_id(document)
            if item_id in known_ids:
                continue
            self._items.append(
                {
                    "id": item_id,
                    "text": document.text,
                    "metadata": document.metadata,
                    "vector": vector,
                }
            )
            known_ids.add(item_id)
            added += 1
        if added:
            self._save()
        return added

    def search(self, query_vector: list[float], top_k: int = 5) -> list[SearchResult]:
        ranked = sorted(
            (
                SearchResult(
                    text=item["text"],
                    metadata=item.get("metadata", {}),
                    score=self._cosine(query_vector, item["vector"]),
                )
                for item in self._items
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        return [result for result in ranked[:top_k] if result.score > 0]

    def count(self) -> int:
        return len(self._items)

    def _load(self) -> None:
        if not self.path.exists():
            self._items = []
            return
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._items = payload.get("items", [])

    def _save(self) -> None:
        payload = {"items": self._items}
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _item_id(document: Document) -> str:
        source = str(document.metadata.get("source", ""))
        chunk_id = str(document.metadata.get("chunk_id", ""))
        raw = f"{source}:{chunk_id}:{document.text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        if not left or not right:
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return numerator / (left_norm * right_norm)
