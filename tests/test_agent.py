from pathlib import Path
from typing import Any

from agent.executor import BusinessAnalysisAgent
from agent.planner import AgentPlan, BusinessPlanner
from agent.synthesizer import LLMAnswerSynthesizer, TemplateAnswerSynthesizer
from rag.embedding import HashingEmbeddingModel
from rag.interfaces import SearchResult
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


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "temperature": temperature,
            }
        )
        return "## Answer\nFinding from the source [1]."


def test_llm_answer_synthesizer_requests_markdown_and_citations() -> None:
    llm = FakeLLMClient()
    synthesizer = LLMAnswerSynthesizer(llm)

    answer = synthesizer.synthesize(
        "Summarize the market",
        AgentPlan(intent="industry_analysis", steps=["retrieve evidence"], tools=["rag"]),
        [SearchResult(text="Market is growing.", metadata={"source": "market.md"}, score=0.8)],
        [],
    )

    assert answer == "## Answer\nFinding from the source [1]."
    assert llm.calls
    prompt = llm.calls[0]["prompt"]
    assert "Markdown" in prompt
    assert "[1], [2]" in prompt
    assert "[1] source=market.md" in prompt


def test_template_synthesizer_covers_tool_only_intents() -> None:
    synthesizer = TemplateAnswerSynthesizer()

    requirement_answer = synthesizer.synthesize(
        "Summarize requirements",
        AgentPlan(intent="requirement_summary", steps=["summarize"], tools=["rag"]),
        [],
        [{"tool": "excel", "result": {"ok": True}}],
    )
    data_answer = synthesizer.synthesize(
        "Analyze metrics",
        AgentPlan(intent="data_analysis", steps=["query"], tools=["sql"]),
        [],
        [{"tool": "sql", "result": {"ok": True}}],
    )

    assert "explicit requirements" in requirement_answer
    assert "SQL or spreadsheet outputs" in data_answer
    assert "No cited sources" in data_answer
