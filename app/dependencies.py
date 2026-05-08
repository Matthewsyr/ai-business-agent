from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent.synthesizer import LLMAnswerSynthesizer
from app.config import settings
from rag.interfaces import EmbeddingModel, VectorStore
from rag.openai_clients import OpenAIEmbeddingModel, OpenAILLMClient


request_id_context: ContextVar[str | None] = ContextVar("request_id", default=None)


def get_request_id() -> str:
    return request_id_context.get() or "unknown"


@dataclass
class FallbackPlan:
    intent: str = "unavailable"
    tools: list[str] = field(default_factory=list)


@dataclass
class FallbackAgentResponse:
    question: str
    answer: str
    plan: FallbackPlan
    sources: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    report_path: str | None = None


class FallbackRetriever:
    def ingest_file(self, path: Path) -> dict[str, Any]:
        return {
            "documents_loaded": 1 if Path(path).exists() else 0,
            "chunks_created": 0,
            "chunks_added": 0,
            "vector_store_size": 0,
        }

    def search(self, query: str, top_k: int = 5) -> list[Any]:
        return []


class FallbackAgent:
    def run(
        self,
        question: str,
        use_web: bool = False,
        sql_query: str | None = None,
        excel_path: str | None = None,
        generate_report: bool = False,
    ) -> FallbackAgentResponse:
        return FallbackAgentResponse(
            question=question,
            answer="Analysis services are not available in this runtime.",
            plan=FallbackPlan(),
            sources=[],
            tool_outputs=[],
            metrics={"task_completion_rate": 0.0},
        )


@lru_cache(maxsize=1)
def get_retriever() -> Any:
    settings.ensure_directories()
    try:
        from rag.chroma_store import ChromaVectorStore
        from rag.embedding import HashingEmbeddingModel
        from rag.retriever import RAGRetriever
        from rag.splitter import TextSplitter
        from rag.vector_store import JsonVectorStore
    except ImportError:
        return FallbackRetriever()

    if settings.openai_api_key and settings.openai_embedding_model:
        embedding_model: EmbeddingModel = OpenAIEmbeddingModel(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        vector_store: VectorStore = ChromaVectorStore(
            settings.chroma_persist_dir,
            collection_name=settings.chroma_collection_name,
        )
    else:
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
def get_agent() -> Any:
    try:
        from agent.executor import BusinessAnalysisAgent
        from agent.planner import BusinessPlanner
        from tools.excel_tool import ExcelAnalysisTool
        from tools.report_tool import ReportTool
        from tools.search_tool import WebSearchTool
        from tools.sql_tool import SQLQueryTool
    except ImportError:
        return FallbackAgent()

    synthesizer = None
    if settings.openai_api_key and settings.openai_model:
        synthesizer = LLMAnswerSynthesizer(
            OpenAILLMClient(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                temperature=settings.llm_temperature,
            ),
            temperature=settings.llm_temperature,
        )

    return BusinessAnalysisAgent(
        retriever=get_retriever(),
        planner=BusinessPlanner(),
        search_tool=WebSearchTool(
            enabled=settings.search_enabled,
            max_results=settings.search_max_results,
            timeout=int(settings.tool_timeout_seconds),
        ),
        sql_tool=SQLQueryTool(settings.sqlite_path, max_rows=settings.sql_row_limit),
        excel_tool=ExcelAnalysisTool(
            base_dir=settings.base_dir,
            allowed_roots=[settings.data_dir],
        ),
        report_tool=ReportTool(settings.reports_dir),
        synthesizer=synthesizer,
        top_k=settings.top_k,
    )
