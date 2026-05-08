from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.safety import is_relative_to, resolve_input_file, tool_error


class ExcelAnalysisTool:
    supported_suffixes = {".csv", ".xlsx"}

    def __init__(
        self, base_dir: Path | None = None, allowed_roots: list[Path] | None = None
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.allowed_roots = [Path(root).resolve() for root in allowed_roots or []]

    def summarize(self, path: str | Path) -> dict[str, Any]:
        try:
            import pandas as pd
        except ImportError as exc:
            return tool_error(
                "dependency_error",
                "Install pandas and openpyxl to use Excel/CSV analysis.",
                dependency=str(exc),
            )

        try:
            resolved_path = resolve_input_file(path, base_dir=self.base_dir)
        except (OSError, RuntimeError, ValueError) as exc:
            return tool_error("path_error", f"Invalid file path: {exc}")

        if self.allowed_roots and not any(
            is_relative_to(resolved_path, root) for root in self.allowed_roots
        ):
            return tool_error("path_error", "File path is outside the allowed directories.")

        suffix = resolved_path.suffix.lower()
        if suffix not in self.supported_suffixes:
            return tool_error(
                "unsupported_file_type",
                "ExcelAnalysisTool supports .csv and .xlsx files.",
                suffix=suffix or "<none>",
            )
        if not resolved_path.exists():
            return tool_error("not_found", f"File not found: {resolved_path}")
        if not resolved_path.is_file():
            return tool_error("path_error", f"Path is not a file: {resolved_path}")

        try:
            if suffix == ".csv":
                frame = pd.read_csv(resolved_path)
            else:
                frame = pd.read_excel(resolved_path)
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            return tool_error("read_error", f"Could not read spreadsheet: {exc}")

        numeric = frame.select_dtypes(include="number")
        categorical = frame.select_dtypes(exclude="number")
        categorical_top = {
            column: frame[column].dropna().astype(str).value_counts().head(5).to_dict()
            for column in categorical.columns[:10]
        }
        return {
            "ok": True,
            "path": str(resolved_path),
            "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
            "columns": list(frame.columns),
            "missing_values": frame.isna().sum().to_dict(),
            "numeric_summary": numeric.describe().round(4).to_dict() if not numeric.empty else {},
            "categorical_top_values": categorical_top,
        }
