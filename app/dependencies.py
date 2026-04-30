from __future__ import annotations

from functools import lru_cache

from agent.executor import BusinessAnalysisAgent
from agent.planner import BusinessPlanner
from app.config import settings
from rag.embedding import HashingEmbeddingModel
from rag.retriever import RAGRetriever
from rag.splitter import TextSplitter
from rag.vector_store import JsonVectorStore
from tools.excel_tool import ExcelAnalysisTool
from tools.report_tool import ReportTool
from tools.search_tool import WebSearchTool
from tools.sql_tool import SQLQueryTool


@lru_cache(maxsize=1)
def get_retriever() -> RAGRetriever:
    settings.ensure_directories()
    embedding_model = HashingEmbeddingModel(dim=settings.embedding_dim)
    vector_store = JsonVectorStore(settings.vector_store_path)
    splitter = TextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return RAGRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        splitter=splitter,
    )


@lru_cache(maxsize=1)
def get_agent() -> BusinessAnalysisAgent:
    return BusinessAnalysisAgent(
        retriever=get_retriever(),
        planner=BusinessPlanner(),
        search_tool=WebSearchTool(
            enabled=settings.search_enabled,
            max_results=settings.search_max_results,
        ),
        sql_tool=SQLQueryTool(settings.sqlite_path),
        excel_tool=ExcelAnalysisTool(),
        report_tool=ReportTool(settings.reports_dir),
    )
