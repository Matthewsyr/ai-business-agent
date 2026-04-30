from __future__ import annotations

from pathlib import Path
from typing import Any


class ExcelAnalysisTool:
    def summarize(self, path: Path) -> dict[str, Any]:
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Install pandas and openpyxl to use Excel analysis.") from exc

        path = Path(path)
        if path.suffix.lower() == ".csv":
            frame = pd.read_csv(path)
        elif path.suffix.lower() in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        else:
            raise ValueError("ExcelAnalysisTool supports .csv, .xlsx, and .xls files.")

        numeric = frame.select_dtypes(include="number")
        categorical = frame.select_dtypes(exclude="number")
        categorical_top = {
            column: frame[column].dropna().astype(str).value_counts().head(5).to_dict()
            for column in categorical.columns[:10]
        }
        return {
            "path": str(path),
            "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
            "columns": list(frame.columns),
            "missing_values": frame.isna().sum().to_dict(),
            "numeric_summary": numeric.describe().round(4).to_dict() if not numeric.empty else {},
            "categorical_top_values": categorical_top,
        }
