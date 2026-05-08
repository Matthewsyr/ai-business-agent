from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from tools.safety import tool_error


class SQLQueryTool:
    def __init__(self, database_path: Path, max_rows: int = 1_000) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_rows = max(1, int(max_rows))

    def query(self, sql: str, params: Sequence[Any] = ()) -> dict[str, Any]:
        try:
            self._validate_read_only(sql)
            if not self.database_path.exists():
                return self._error("not_found", f"SQLite database not found: {self.database_path}")

            uri = f"file:{self.database_path.resolve().as_posix()}?mode=ro"
            with sqlite3.connect(uri, uri=True) as connection:
                connection.row_factory = sqlite3.Row
                connection.set_authorizer(self._authorizer)
                cursor = connection.execute(sql, tuple(params))
                fetched = cursor.fetchmany(self.max_rows + 1)
                truncated = len(fetched) > self.max_rows
                rows = [dict(row) for row in fetched[: self.max_rows]]
                columns = [description[0] for description in cursor.description or []]
        except ValueError as exc:
            return self._error("validation_error", str(exc))
        except sqlite3.Error as exc:
            return self._error("database_error", str(exc))

        return {
            "ok": True,
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "max_rows": self.max_rows,
        }

    @staticmethod
    def _validate_read_only(sql: str) -> None:
        normalized = SQLQueryTool._strip_comments(sql).strip().lower()
        if not re.match(r"^(select|with)\b", normalized):
            raise ValueError("Only SELECT/WITH read-only queries are allowed.")
        if ";" in normalized.rstrip(";"):
            raise ValueError("Multiple SQL statements are not allowed.")

    @staticmethod
    def _strip_comments(sql: str) -> str:
        without_block_comments = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
        return re.sub(r"--.*?$", "", without_block_comments, flags=re.MULTILINE)

    @staticmethod
    def _authorizer(action: int, arg1: str | None, arg2: str | None, *_: Any) -> int:
        blocked_actions = {
            sqlite3.SQLITE_ATTACH,
            sqlite3.SQLITE_ALTER_TABLE,
            sqlite3.SQLITE_CREATE_INDEX,
            sqlite3.SQLITE_CREATE_TABLE,
            sqlite3.SQLITE_CREATE_TEMP_INDEX,
            sqlite3.SQLITE_CREATE_TEMP_TABLE,
            sqlite3.SQLITE_CREATE_TEMP_TRIGGER,
            sqlite3.SQLITE_CREATE_TEMP_VIEW,
            sqlite3.SQLITE_CREATE_TRIGGER,
            sqlite3.SQLITE_CREATE_VIEW,
            sqlite3.SQLITE_DELETE,
            sqlite3.SQLITE_DETACH,
            sqlite3.SQLITE_DROP_INDEX,
            sqlite3.SQLITE_DROP_TABLE,
            sqlite3.SQLITE_DROP_TEMP_INDEX,
            sqlite3.SQLITE_DROP_TEMP_TABLE,
            sqlite3.SQLITE_DROP_TEMP_TRIGGER,
            sqlite3.SQLITE_DROP_TEMP_VIEW,
            sqlite3.SQLITE_DROP_TRIGGER,
            sqlite3.SQLITE_DROP_VIEW,
            sqlite3.SQLITE_INSERT,
            sqlite3.SQLITE_PRAGMA,
            sqlite3.SQLITE_REINDEX,
            sqlite3.SQLITE_TRANSACTION,
            sqlite3.SQLITE_UPDATE,
        }
        if action in blocked_actions:
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            function_name = (arg1 or arg2 or "").lower()
            if function_name in {"load_extension", "readfile", "writefile"}:
                return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_OK

    @staticmethod
    def _error(error_type: str, message: str) -> dict[str, Any]:
        payload = tool_error(error_type, message)
        payload.update(
            {
                "columns": [],
                "rows": [],
                "row_count": 0,
                "truncated": False,
            }
        )
        return payload
