from pathlib import Path

from rag.embedding import HashingEmbeddingModel
from rag.retriever import RAGRetriever
from rag.splitter import TextSplitter
from rag.vector_store import JsonVectorStore


def test_ingest_and_search_txt(tmp_path: Path) -> None:
    doc_path = tmp_path / "company.txt"
    doc_path.write_text("A公司聚焦企业智能调研，核心能力包括RAG检索和自动报告生成。", encoding="utf-8")
    retriever = RAGRetriever(
        embedding_model=HashingEmbeddingModel(dim=64),
        vector_store=JsonVectorStore(tmp_path / "vectors.json"),
        splitter=TextSplitter(chunk_size=80, chunk_overlap=10),
    )

    ingest = retriever.ingest_file(doc_path)
    results = retriever.search("企业智能调研 RAG 报告", top_k=3)

    assert ingest["chunks_added"] == 1
    assert results
    assert "RAG" in results[0].text
