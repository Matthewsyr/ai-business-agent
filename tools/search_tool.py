from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote

from tools.safety import tool_error


@dataclass
class SearchItem:
    title: str
    url: str
    snippet: str


class WebSearchTool:
    def __init__(self, enabled: bool = False, max_results: int = 5, timeout: int = 10) -> None:
        self.enabled = enabled
        self.max_results = max(1, int(max_results))
        self.timeout = max(1, int(timeout))

    def search(self, query: str) -> dict[str, Any]:
        if not self.enabled:
            return {
                "ok": True,
                "enabled": False,
                "items": [],
                "note": "Set SEARCH_ENABLED=true to enable live DuckDuckGo HTML search.",
            }
        if not query.strip():
            payload = tool_error("validation_error", "Search query cannot be empty.")
            payload.update({"enabled": True, "items": []})
            return payload

        try:
            import requests
        except ImportError:
            payload = tool_error("dependency_error", "Install requests to enable search.")
            payload.update({"enabled": False, "items": []})
            return payload

        try:
            response = requests.get(
                "https://duckduckgo.com/html/",
                params={"q": query},
                timeout=self.timeout,
                headers={"User-Agent": "ai-business-agent/0.1"},
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            payload = tool_error("request_error", f"Search request failed: {exc}")
            payload.update({"enabled": True, "items": []})
            return payload

        items = self._parse_duckduckgo(response.text)[: self.max_results]
        return {"ok": True, "enabled": True, "items": [item.__dict__ for item in items]}

    @staticmethod
    def _parse_duckduckgo(page: str) -> list[SearchItem]:
        pattern = re.compile(
            r'<a rel="nofollow" class="result__a" href="(?P<url>.*?)">(?P<title>.*?)</a>.*?'
            r'<a class="result__snippet".*?>(?P<snippet>.*?)</a>',
            re.DOTALL,
        )
        items: list[SearchItem] = []
        for match in pattern.finditer(page):
            url = html.unescape(match.group("url"))
            url = unquote(url.replace("//duckduckgo.com/l/?uddg=", ""))
            title = WebSearchTool._clean_html(match.group("title"))
            snippet = WebSearchTool._clean_html(match.group("snippet"))
            items.append(SearchItem(title=title, url=url, snippet=snippet))
        return items

    @staticmethod
    def _clean_html(value: str) -> str:
        text = re.sub(r"<.*?>", "", value)
        return html.unescape(text).strip()
