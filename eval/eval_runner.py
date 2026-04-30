from __future__ import annotations

import argparse
import json
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


def build_agent(workdir: Path) -> BusinessAnalysisAgent:
    retriever = RAGRetriever(
        embedding_model=HashingEmbeddingModel(),
        vector_store=JsonVectorStore(workdir / "vector_store.json"),
        splitter=TextSplitter(),
    )
    return BusinessAnalysisAgent(
        retriever=retriever,
        planner=BusinessPlanner(),
        search_tool=WebSearchTool(enabled=False),
        sql_tool=SQLQueryTool(workdir / "business.sqlite3"),
        excel_tool=ExcelAnalysisTool(),
        report_tool=ReportTool(workdir / "reports"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True, help="JSONL file with question fields.")
    parser.add_argument("--workdir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    agent = build_agent(args.workdir)
    results = []
    with args.questions.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            response = agent.run(item["question"])
            results.append({"question": item["question"], "metrics": response.metrics})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
