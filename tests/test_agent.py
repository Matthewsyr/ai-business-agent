from pathlib import Path

from agent.executor import BusinessAnalysisAgent
from agent.planner import BusinessPlanner
from rag.embedding import HashingEmbeddingModel
from rag.retriever import RAGRetriever
from rag.splitter import TextSplitter
from rag.vector_store import JsonVectorStore
from tools.excel_tool import ExcelAnalysisTool
from tools.report_tool import ReportTool
from tools.search_tool import WebSearchTool
from tools.sql_tool import SQLQueryTool


def test_agent_generates_structured_answer(tmp_path: Path) -> None:
    doc_path = tmp_path / "competitor.md"
    doc_path.write_text(
        "竞品B主打低价获客，A公司优势是行业知识库、自动化办公和结构化报告生成。",
        encoding="utf-8",
    )
    retriever = RAGRetriever(
        embedding_model=HashingEmbeddingModel(dim=64),
        vector_store=JsonVectorStore(tmp_path / "vectors.json"),
        splitter=TextSplitter(chunk_size=80, chunk_overlap=10),
    )
    retriever.ingest_file(doc_path)
    agent = BusinessAnalysisAgent(
        retriever=retriever,
        planner=BusinessPlanner(),
        search_tool=WebSearchTool(enabled=False),
        sql_tool=SQLQueryTool(tmp_path / "business.sqlite3"),
        excel_tool=ExcelAnalysisTool(),
        report_tool=ReportTool(tmp_path / "reports"),
    )

    response = agent.run("请做A公司和竞品B的竞品对比", generate_report=True)

    assert "竞品/对比分析" in response.answer
    assert response.sources
    assert response.report_path is not None
