from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.memory import ConversationMemory
from agent.planner import AgentPlan, BusinessPlanner
from eval.metrics import citation_coverage, retrieval_hit_rate
from rag.retriever import RAGRetriever
from rag.vector_store import SearchResult
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
    ) -> None:
        self.retriever = retriever
        self.planner = planner
        self.search_tool = search_tool
        self.sql_tool = sql_tool
        self.excel_tool = excel_tool
        self.report_tool = report_tool
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
        retrieved = self.retriever.search(question, top_k=5)
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
        if not retrieved and not tool_outputs:
            return (
                "当前知识库没有检索到足够证据。建议先上传行业报告、公司资料、竞品文档或业务数据，"
                "再重新发起分析。"
            )

        evidence_lines = []
        for idx, item in enumerate(retrieved, start=1):
            source = item.metadata.get("source", "knowledge_base")
            preview = item.text.strip().replace("\n", " ")[:220]
            evidence_lines.append(f"[{idx}] {source}：{preview}")

        tool_lines = []
        for output in tool_outputs:
            tool_lines.append(f"- {output['tool']}: {output['result']}")

        sections = [
            f"## 背景\n问题：{question}\n\n本次任务类型为 `{plan.intent}`，已按资料检索、工具补充和结构化总结执行。",
            "## 问题分析\n"
            + self._analysis_text(plan.intent, retrieved)
            + ("\n\n工具补充：\n" + "\n".join(tool_lines) if tool_lines else ""),
            "## 关键发现\n"
            + "\n".join(
                f"- {line}" for line in evidence_lines[:5]
            ),
            "## 建议方案\n"
            "- 优先围绕高频证据中的业务主题形成分析框架。\n"
            "- 对竞品、市场和用户需求分别建立指标表，补充定量数据后再做优先级排序。\n"
            "- 报告交付时保留来源编号，便于复核和二次追问。",
            "## 引用来源\n" + "\n".join(evidence_lines),
        ]
        if plan.intent == "competitor_analysis":
            sections.insert(
                2,
                "## 竞品/对比分析\n"
                "建议从目标客户、核心能力、商业模式、渠道、价格和风险六个维度展开。"
                "当前检索证据可作为对比表的事实基础。",
            )
        return "\n\n".join(sections)

    @staticmethod
    def _analysis_text(intent: str, retrieved: list[SearchResult]) -> str:
        if intent == "industry_analysis":
            return "行业分析应聚焦市场规模、增长驱动、政策/技术变化和竞争格局。"
        if intent == "competitor_analysis":
            return "竞品分析应先统一比较维度，再抽取各公司定位、产品能力和差异化策略。"
        if intent == "requirement_summary":
            return "需求总结应区分显性需求、隐性痛点、决策角色和可落地功能。"
        if intent == "data_analysis":
            return "数据分析应明确指标口径，结合 SQL 或 Excel 输出进行解释。"
        if retrieved:
            return "知识库检索到了相关企业资料，可基于来源片段生成业务结论。"
        return "暂无足够检索证据。"

    @staticmethod
    def _source_payload(item: SearchResult) -> dict[str, Any]:
        return {
            "text": item.text,
            "metadata": item.metadata,
            "score": item.score,
        }
