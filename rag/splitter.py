from __future__ import annotations

from rag.loader import Document


class TextSplitter:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks: list[Document] = []
        for document in documents:
            text = self._normalize(document.text)
            start = 0
            chunk_index = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end].strip()
                if chunk_text:
                    metadata = dict(document.metadata)
                    metadata["chunk_id"] = chunk_index
                    chunks.append(Document(text=chunk_text, metadata=metadata))
                    chunk_index += 1
                if end == len(text):
                    break
                start = max(0, end - self.chunk_overlap)
        return chunks

    @staticmethod
    def _normalize(text: str) -> str:
        lines = [line.strip() for line in text.splitlines()]
        return "\n".join(line for line in lines if line)
