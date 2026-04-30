from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any


class SQLQueryTool:
    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any]:
        self._validate_read_only(sql)
        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(sql, params)
            rows = [dict(row) for row in cursor.fetchall()]
            columns = [description[0] for description in cursor.description or []]
        return {"columns": columns, "rows": rows, "row_count": len(rows)}

    @staticmethod
    def _validate_read_only(sql: str) -> None:
        normalized = sql.strip().lower()
        if not re.match(r"^(select|with)\b", normalized):
            raise ValueError("Only SELECT/WITH read-only queries are allowed.")
        if ";" in normalized.rstrip(";"):
            raise ValueError("Multiple SQL statements are not allowed.")
        blocked = {"insert", "update", "delete", "drop", "alter", "create", "replace", "attach"}
        tokens = set(re.findall(r"[a-z_]+", normalized))
        if tokens & blocked:
            raise ValueError("Write or schema-changing SQL is not allowed.")
