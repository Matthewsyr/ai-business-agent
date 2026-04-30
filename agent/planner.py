from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentPlan:
    intent: str
    steps: list[str]
    tools: list[str] = field(default_factory=list)
    report_sections: list[str] = field(default_factory=list)


class BusinessPlanner:
    """Rule-based planner that keeps the MVP deterministic and testable."""

    def plan(self, question: str) -> AgentPlan:
        normalized = question.lower()
        tools = ["rag"]
        sections = ["背景", "问题分析", "关键发现", "建议方案", "引用来源"]

        if any(word in normalized for word in ["竞品", "竞争", "对比", "competitor", "benchmark"]):
            intent = "competitor_analysis"
            sections.insert(2, "竞品/对比分析")
        elif any(word in normalized for word in ["行业", "市场", "趋势", "industry", "market"]):
            intent = "industry_analysis"
        elif any(word in normalized for word in ["用户", "需求", "痛点", "persona", "requirement"]):
            intent = "requirement_summary"
        elif any(word in normalized for word in ["sql", "数据库", "指标", "营收", "转化率"]):
            intent = "data_analysis"
            tools.append("sql")
        else:
            intent = "business_qa"

        if any(word in normalized for word in ["外部", "最新", "新闻", "网页", "搜索", "web"]):
            tools.append("search")
        if any(word in normalized for word in ["excel", "表格", "csv", "xlsx"]):
            tools.append("excel")

        steps = [
            "解析业务问题和输出目标",
            "检索企业知识库并筛选高相关片段",
            "按任务类型调用外部工具补充证据",
            "综合证据生成结构化结论和建议",
        ]
        return AgentPlan(intent=intent, steps=steps, tools=tools, report_sections=sections)
