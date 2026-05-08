from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.interfaces import EmbeddingModel, SearchResult, VectorStore
from rag.loader import DocumentLoader
from rag.splitter import TextSplitter


class RAGRetriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        splitter: TextSplitter,
        loader: DocumentLoader | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.splitter = splitter
        self.loader = loader or DocumentLoader()

    def ingest_file(self, path: Path) -> dict[str, Any]:
        documents = self.loader.load(Path(path))
        chunks = self.splitter.split_documents(documents)
        vectors = self._embed_batch([chunk.text for chunk in chunks])
        added = self.vector_store.add_documents(chunks, vectors)
        return {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "chunks_added": added,
            "vector_store_size": self.vector_store.count(),
        }

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_vector = self.embedding_model.embed(query)
        return self.vector_store.search(query_vector=query_vector, top_k=top_k)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        embed_batch = getattr(self.embedding_model, "embed_batch", None)
        if callable(embed_batch):
            return embed_batch(texts)
        return [self.embedding_model.embed(text) for text in texts]
