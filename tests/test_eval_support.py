from __future__ import annotations

from agent.memory import ConversationMemory
from agent.planner import BusinessPlanner
from eval.feedback import FeedbackRecord, FeedbackStore
from eval.metrics import citation_coverage, retrieval_hit_rate, task_completion_rate


def test_metrics_cover_empty_and_positive_cases() -> None:
    assert retrieval_hit_rate([]) == 0.0
    assert retrieval_hit_rate([0.01, 0.5], threshold=0.05) == 0.5
    assert citation_coverage("Claim [1], claim [2].", [{"id": 1}, {"id": 2}]) == 1.0
    assert citation_coverage("No citations", []) == 0.0
    assert task_completion_rate([]) == 0.0
    assert task_completion_rate([{"completed": True}, {"completed": False}]) == 0.5


def test_feedback_store_round_trips_jsonl(tmp_path) -> None:
    store = FeedbackStore(tmp_path / "feedback.jsonl")
    store.add(FeedbackRecord(question="How is the market?", rating=4, comment="useful"))

    records = store.list()

    assert len(records) == 1
    assert records[0].question == "How is the market?"
    assert records[0].rating == 4
    assert records[0].comment == "useful"


def test_memory_keeps_recent_messages_only() -> None:
    memory = ConversationMemory(max_messages=2)
    memory.add("user", "one")
    memory.add("assistant", "two")
    memory.add("user", "three")

    assert [message.content for message in memory.history()] == ["two", "three"]
    memory.clear()
    assert memory.history() == []


def test_business_planner_routes_common_intents_and_tools() -> None:
    planner = BusinessPlanner()

    assert planner.plan("market trend analysis").intent == "industry_analysis"
    assert planner.plan("competitor benchmark").intent == "competitor_analysis"
    assert planner.plan("persona requirement summary").intent == "requirement_summary"

    data_plan = planner.plan("run sql metrics from csv and web")

    assert data_plan.intent == "data_analysis"
    assert {"rag", "sql", "search", "excel"}.issubset(set(data_plan.tools))
