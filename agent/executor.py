from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.memory import ConversationMemory
from agent.planner import AgentPlan, BusinessPlanner
from agent.synthesizer import AnswerSynthesizer, TemplateAnswerSynthesizer
from eval.metrics import citation_coverage, retrieval_hit_rate
from rag.interfaces import SearchResult
from rag.retriever import RAGRetriever
from tools.excel_tool import ExcelAnalysisTool
from tools.report_tool import ReportTool
from tools.search_tool import WebSearchTool
from tools.sql_tool import SQLQueryTool


@dataclass
class AgentResponse:
    question: str
    answer: str
    plan: AgentPlan
    sources: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    report_path: str | None = None


class BusinessAnalysisAgent:
    def __init__(
        self,
        retriever: RAGRetriever,
        planner: BusinessPlanner,
        search_tool: WebSearchTool,
        sql_tool: SQLQueryTool,
        excel_tool: ExcelAnalysisTool,
        report_tool: ReportTool,
        synthesizer: AnswerSynthesizer | None = None,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.planner = planner
        self.search_tool = search_tool
        self.sql_tool = sql_tool
        self.excel_tool = excel_tool
        self.report_tool = report_tool
        self.synthesizer = synthesizer or TemplateAnswerSynthesizer()
        self.top_k = max(1, int(top_k))
        self.memory = ConversationMemory()

    def run(
        self,
        question: str,
        use_web: bool = False,
        sql_query: str | None = None,
        excel_path: str | None = None,
        generate_report: bool = False,
    ) -> AgentResponse:
        self.memory.add("user", question)
        plan = self.planner.plan(question)
        retrieved = self.retriever.search(question, top_k=self.top_k)
        tool_outputs = self._run_tools(
            question=question,
            plan=plan,
            use_web=use_web,
            sql_query=sql_query,
            excel_path=excel_path,
        )
        answer = self._synthesize_answer(question, plan, retrieved, tool_outputs)
        sources = [self._source_payload(item) for item in retrieved]
        metrics = {
            "retrieval_hit_rate": retrieval_hit_rate([item.score for item in retrieved]),
            "citation_coverage": citation_coverage(answer, sources),
            "task_completion_rate": 1.0 if answer else 0.0,
        }

        report_path = None
        if generate_report:
            report_path = str(
                self.report_tool.generate_business_report(
                    topic=question,
                    answer=answer,
                    plan=plan,
                    sources=sources,
                    tool_outputs=tool_outputs,
                )
            )
        self.memory.add("assistant", answer)
        return AgentResponse(
            question=question,
            answer=answer,
            plan=plan,
            sources=sources,
            tool_outputs=tool_outputs,
            metrics=metrics,
            report_path=report_path,
        )

    def _run_tools(
        self,
        question: str,
        plan: AgentPlan,
        use_web: bool,
        sql_query: str | None,
        excel_path: str | None,
    ) -> list[dict[str, Any]]:
        outputs: list[dict[str, Any]] = []
        if use_web or "search" in plan.tools:
            outputs.append({"tool": "search", "result": self.search_tool.search(question)})
        if sql_query:
            outputs.append({"tool": "sql", "result": self.sql_tool.query(sql_query)})
        if excel_path:
            outputs.append({"tool": "excel", "result": self.excel_tool.summarize(Path(excel_path))})
        return outputs

    def _synthesize_answer(
        self,
        question: str,
        plan: AgentPlan,
        retrieved: list[SearchResult],
        tool_outputs: list[dict[str, Any]],
    ) -> str:
        return self.synthesizer.synthesize(question, plan, retrieved, tool_outputs)

    @staticmethod
    def _source_payload(item: SearchResult) -> dict[str, Any]:
        return {
            "text": item.text,
            "metadata": item.metadata,
            "score": item.score,
        }
