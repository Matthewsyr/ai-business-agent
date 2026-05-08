from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st


DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1").rstrip("/")


def unwrap_envelope(payload: Any) -> Any:
    if not isinstance(payload, dict):
        return payload
    if payload.get("ok") is False or payload.get("success") is False:
        return payload
    if "data" in payload and any(key in payload for key in ("ok", "success", "status")):
        return payload["data"]
    if "result" in payload and any(key in payload for key in ("ok", "success", "status")):
        return payload["result"]
    return payload


def error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail") or payload.get("message")
        if isinstance(detail, str):
            return detail
        error = payload.get("error")
        if isinstance(error, str):
            return error
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
    return "Request failed."


def api_request(method: str, api_base: str, path: str, **kwargs: Any) -> Any | None:
    url = f"{api_base.rstrip('/')}/{path.lstrip('/')}"
    try:
        response = requests.request(method, url, **kwargs)
    except requests.RequestException as exc:
        st.error(f"Could not reach API: {exc}")
        return None

    try:
        payload = response.json()
    except ValueError:
        preview = response.text[:500] if response.text else "<empty response>"
        st.error(f"API returned non-JSON response ({response.status_code}): {preview}")
        return None

    if not response.ok:
        st.error(f"API error {response.status_code}: {error_message(payload)}")
        return None

    unwrapped = unwrap_envelope(payload)
    if isinstance(unwrapped, dict) and (
        unwrapped.get("ok") is False or unwrapped.get("success") is False
    ):
        st.error(error_message(unwrapped))
        return None
    return unwrapped


st.set_page_config(page_title="AI Business Analysis Agent", layout="wide")

api_base = st.sidebar.text_input("API Base URL", value=DEFAULT_API_BASE).rstrip("/")
st.title("AI Business Analysis Agent")

uploaded = st.sidebar.file_uploader("Upload knowledge document", type=["txt", "md", "pdf", "docx"])
if uploaded and st.sidebar.button("Ingest document"):
    files = {"file": (uploaded.name, uploaded.getvalue())}
    data = api_request("POST", api_base, "/upload", files=files, timeout=60)
    if data is not None:
        st.sidebar.json(data)

question = st.text_area(
    "Business question",
    height=120,
    placeholder="Example: Analyze Company A's differentiation opportunities against key competitors.",
)
col1, col2, col3 = st.columns(3)
use_web = col1.toggle("Enable web search", value=False)
generate_report = col2.toggle("Generate report", value=True)
excel_path = col3.text_input("Excel/CSV path", value="")
sql_query = st.text_area("SQL query (SELECT/WITH only)", height=80)

if st.button("Start analysis", type="primary") and question.strip():
    payload = {
        "question": question,
        "use_web": use_web,
        "generate_report": generate_report,
        "excel_path": excel_path or None,
        "sql_query": sql_query or None,
    }
    with st.spinner("Running retrieval, tools, and report generation..."):
        data = api_request("POST", api_base, "/chat", json=payload, timeout=120)

    if isinstance(data, dict):
        st.markdown(str(data.get("answer", "")))
        st.subheader("Sources")
        st.json(data.get("sources", []))
        st.subheader("Metrics")
        st.json(data.get("metrics", {}))
        if data.get("tool_outputs"):
            st.subheader("Tool Outputs")
            st.json(data["tool_outputs"])
        if data.get("report_path"):
            st.success(f"Report generated: {data['report_path']}")
