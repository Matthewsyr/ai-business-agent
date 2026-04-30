import sqlite3
from pathlib import Path

import pytest

from tools.excel_tool import ExcelAnalysisTool
from tools.sql_tool import SQLQueryTool


def test_sql_tool_allows_select_only(tmp_path: Path) -> None:
    db_path = tmp_path / "business.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute("create table revenue(month text, amount int)")
        connection.execute("insert into revenue values ('2026-01', 100)")

    tool = SQLQueryTool(db_path)
    result = tool.query("select month, amount from revenue")

    assert result["row_count"] == 1
    assert result["rows"][0]["amount"] == 100
    with pytest.raises(ValueError):
        tool.query("delete from revenue")


def test_excel_tool_summarizes_csv(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("channel,revenue\nsearch,10\nads,20\nsearch,30\n", encoding="utf-8")

    result = ExcelAnalysisTool().summarize(csv_path)

    assert result["shape"] == {"rows": 3, "columns": 2}
    assert "revenue" in result["numeric_summary"]
