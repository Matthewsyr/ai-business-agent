from __future__ import annotations

from typing import Any, Protocol

from agent.planner import AgentPlan
from rag.interfaces import LLMClient, SearchResult


class AnswerSynthesizer(Protocol):
    def synthesize(
        self,
        question: str,
        plan: AgentPlan,
        retrieved: list[SearchResult],
        tool_outputs: list[dict[str, Any]],
    ) -> str: ...


class TemplateAnswerSynthesizer:
    def synthesize(
        self,
        question: str,
        plan: AgentPlan,
        retrieved: list[SearchResult],
        tool_outputs: list[dict[str, Any]],
    ) -> str:
        if not retrieved and not tool_outputs:
            return (
                "No sufficient knowledge-base evidence was found. Upload source documents "
                "or provide tool inputs, then run the analysis again."
            )

        evidence_lines = self._evidence_lines(retrieved)
        tool_lines = [f"- {output['tool']}: {output['result']}" for output in tool_outputs]

        sections = [
            f"## Background\nQuestion: {question}\n\nIntent: `{plan.intent}`.",
            "## Analysis\n"
            + self._analysis_text(plan.intent, retrieved)
            + ("\n\nTool outputs:\n" + "\n".join(tool_lines) if tool_lines else ""),
            "## Key Findings\n"
            + (
                "\n".join(f"- {line}" for line in evidence_lines[:5])
                if evidence_lines
                else "- No retrieved source snippets."
            ),
            "## Recommendations\n"
            "- Prioritize the themes that appear across the strongest retrieved evidence.\n"
            "- Add quantitative business data before ranking market, competitor, or product decisions.\n"
            "- Keep source numbers in the final report so claims can be reviewed.",
            "## Sources\n" + ("\n".join(evidence_lines) if evidence_lines else "No cited sources."),
        ]
        if plan.intent == "competitor_analysis":
            sections.insert(
                2,
                "## 竞品/对比分析\n"
                "Compare target customers, product capabilities, pricing, channels, differentiation, and risks.",
            )
        return "\n\n".join(sections)

    @staticmethod
    def _evidence_lines(retrieved: list[SearchResult]) -> list[str]:
        lines = []
        for idx, item in enumerate(retrieved, start=1):
            source = item.metadata.get("source", "knowledge_base")
            preview = item.text.strip().replace("\n", " ")[:220]
            lines.append(f"[{idx}] {source}: {preview}")
        return lines

    @staticmethod
    def _analysis_text(intent: str, retrieved: list[SearchResult]) -> str:
        if intent == "industry_analysis":
            return "Focus on market size, growth drivers, policy or technology shifts, and competitive structure."
        if intent == "competitor_analysis":
            return "Normalize comparison dimensions first, then map each competitor to strengths and gaps."
        if intent == "requirement_summary":
            return "Separate explicit requirements, implicit pain points, decision roles, and shippable features."
        if intent == "data_analysis":
            return "Clarify metric definitions and combine SQL or spreadsheet outputs with retrieved context."
        if retrieved:
            return "The knowledge base contains relevant business evidence for a structured answer."
        return "There is not enough retrieved evidence yet."


class LLMAnswerSynthesizer:
    def __init__(self, llm_client: LLMClient, *, temperature: float = 0.2) -> None:
        self.llm_client = llm_client
        self.temperature = temperature

    def synthesize(
        self,
        question: str,
        plan: AgentPlan,
        retrieved: list[SearchResult],
        tool_outputs: list[dict[str, Any]],
    ) -> str:
        return self.llm_client.generate(
            self._build_prompt(question, plan, retrieved, tool_outputs),
            system_prompt=(
                "You are a business analysis assistant. Write in Markdown. "
                "Cite source snippets using bracket citations like [1], [2]."
            ),
            temperature=self.temperature,
        )

    @staticmethod
    def _build_prompt(
        question: str,
        plan: AgentPlan,
        retrieved: list[SearchResult],
        tool_outputs: list[dict[str, Any]],
    ) -> str:
        source_blocks = []
        for idx, item in enumerate(retrieved, start=1):
            source = item.metadata.get("source", "knowledge_base")
            source_blocks.append(
                f"[{idx}] source={source}\nscore={item.score:.4f}\ntext={item.text}"
            )
        tool_blocks = [f"- {output['tool']}: {output['result']}" for output in tool_outputs]
        return "\n\n".join(
            [
                "Create a structured business analysis answer in Markdown.",
                "Use source citations like [1], [2] for factual claims. Do not invent citation numbers.",
                f"Question:\n{question}",
                f"Plan intent:\n{plan.intent}",
                "Plan steps:\n" + "\n".join(f"- {step}" for step in plan.steps),
                "Sources:\n"
                + ("\n\n".join(source_blocks) if source_blocks else "No retrieved sources."),
                "Tool outputs:\n" + ("\n".join(tool_blocks) if tool_blocks else "No tool outputs."),
            ]
        )
