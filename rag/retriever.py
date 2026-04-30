from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.embedding import HashingEmbeddingModel
from rag.loader import DocumentLoader
from rag.splitter import TextSplitter
from rag.vector_store import JsonVectorStore, SearchResult


class RAGRetriever:
    def __init__(
        self,
        embedding_model: HashingEmbeddingModel,
        vector_store: JsonVectorStore,
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
        vectors = [self.embedding_model.embed(chunk.text) for chunk in chunks]
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
