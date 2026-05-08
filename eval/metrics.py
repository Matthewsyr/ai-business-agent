from __future__ import annotations

from typing import Any


def retrieval_hit_rate(scores: list[float], threshold: float = 0.05) -> float:
    if not scores:
        return 0.0
    hits = sum(1 for score in scores if score >= threshold)
    return hits / len(scores)


def citation_coverage(answer: str, sources: list[dict[str, Any]]) -> float:
    if not sources:
        return 0.0
    cited = sum(1 for index in range(1, len(sources) + 1) if f"[{index}]" in answer)
    return cited / len(sources)


def task_completion_rate(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    completed = sum(1 for result in results if result.get("completed"))
    return completed / len(results)
