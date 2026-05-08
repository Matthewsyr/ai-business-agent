from __future__ import annotations

from pathlib import Path
from typing import Any


def tool_error(error_type: str, message: str, **details: Any) -> dict[str, Any]:
    error: dict[str, Any] = {"type": error_type, "message": message}
    if details:
        error["details"] = details
    return {"ok": False, "error": error}


def resolve_input_file(path: str | Path, *, base_dir: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        root = (base_dir or Path.cwd()).resolve()
        candidate = root / candidate
    return candidate.resolve()


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True
