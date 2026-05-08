import sqlite3
from pathlib import Path

import pytest

from tools.excel_tool import ExcelAnalysisTool
from tools.search_tool import WebSearchTool
from tools.sql_tool import SQLQueryTool


def create_database(path: Path, row_count: int = 1) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("create table revenue(month text, amount int)")
        connection.executemany(
            "insert into revenue values (?, ?)",
            [(f"2026-{index + 1:02d}", (index + 1) * 100) for index in range(row_count)],
        )


def test_sql_tool_allows_select_and_limits_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "business.sqlite3"
    create_database(db_path, row_count=3)

    tool = SQLQueryTool(db_path, max_rows=2)
    result = tool.query("select month, amount from revenue order by month")

    assert result["ok"] is True
    assert result["row_count"] == 2
    assert result["truncated"] is True
    assert result["rows"][0]["amount"] == 100


def test_sql_tool_returns_structured_error_for_write_query(tmp_path: Path) -> None:
    db_path = tmp_path / "business.sqlite3"
    create_database(db_path)

    result = SQLQueryTool(db_path).query("delete from revenue")

    assert result["ok"] is False
    assert result["row_count"] == 0
    assert result["error"]["type"] == "validation_error"


def test_sql_tool_authorizer_blocks_unsafe_function(tmp_path: Path) -> None:
    db_path = tmp_path / "business.sqlite3"
    create_database(db_path)

    result = SQLQueryTool(db_path).query("select load_extension('extension')")

    assert result["ok"] is False
    assert result["error"]["type"] == "database_error"


def test_sql_tool_missing_database_is_structured_error(tmp_path: Path) -> None:
    result = SQLQueryTool(tmp_path / "missing.sqlite3").query("select 1")

    assert result["ok"] is False
    assert result["error"]["type"] == "not_found"


def test_excel_tool_summarizes_csv(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("channel,revenue\nsearch,10\nads,20\nsearch,30\n", encoding="utf-8")

    result = ExcelAnalysisTool(allowed_roots=[tmp_path]).summarize(csv_path)

    assert result["ok"] is True
    assert result["shape"] == {"rows": 3, "columns": 2}
    assert "revenue" in result["numeric_summary"]


def test_excel_tool_rejects_xls_without_xlrd_support(tmp_path: Path) -> None:
    xls_path = tmp_path / "legacy.xls"
    xls_path.write_text("not a real spreadsheet", encoding="utf-8")

    result = ExcelAnalysisTool(allowed_roots=[tmp_path]).summarize(xls_path)

    assert result["ok"] is False
    assert result["error"]["type"] == "unsupported_file_type"


def test_excel_tool_rejects_paths_outside_allowed_roots(tmp_path: Path) -> None:
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    csv_path = other_dir / "metrics.csv"
    csv_path.write_text("channel,revenue\nsearch,10\n", encoding="utf-8")

    result = ExcelAnalysisTool(allowed_roots=[tmp_path / "allowed"]).summarize(csv_path)

    assert result["ok"] is False
    assert result["error"]["type"] == "path_error"


def test_search_tool_returns_structured_error_on_request_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests = pytest.importorskip("requests")

    def raise_timeout(*_args: object, **_kwargs: object) -> None:
        raise requests.Timeout("timed out")

    monkeypatch.setattr(requests, "get", raise_timeout)

    result = WebSearchTool(enabled=True).search("market share")

    assert result["ok"] is False
    assert result["enabled"] is True
    assert result["items"] == []
    assert result["error"]["type"] == "request_error"


def test_search_tool_parses_duckduckgo_html() -> None:
    html = """
    <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com">
      Example &amp; Co
    </a>
    <a class="result__snippet">Useful <b>market</b> result</a>
    """

    items = WebSearchTool._parse_duckduckgo(html)

    assert len(items) == 1
    assert items[0].title == "Example & Co"
    assert items[0].url == "https://example.com"
    assert items[0].snippet == "Useful market result"
