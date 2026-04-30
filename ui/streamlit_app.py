from __future__ import annotations

import requests
import streamlit as st


st.set_page_config(page_title="企业智能业务分析 Agent", layout="wide")

API_BASE = st.sidebar.text_input("API Base", value="http://localhost:8000/api")
st.title("企业智能业务分析 Agent")

uploaded = st.sidebar.file_uploader("上传企业/行业资料", type=["txt", "md", "pdf", "docx"])
if uploaded and st.sidebar.button("写入知识库"):
    files = {"file": (uploaded.name, uploaded.getvalue())}
    response = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
    st.sidebar.json(response.json())

question = st.text_area("业务问题", height=120, placeholder="例如：请分析 A 公司与主要竞品的差异化机会")
col1, col2, col3 = st.columns(3)
use_web = col1.toggle("启用网页搜索", value=False)
generate_report = col2.toggle("生成报告", value=True)
excel_path = col3.text_input("Excel/CSV 路径", value="")
sql_query = st.text_area("SQL 查询（只允许 SELECT/WITH）", height=80)

if st.button("开始分析", type="primary") and question.strip():
    payload = {
        "question": question,
        "use_web": use_web,
        "generate_report": generate_report,
        "excel_path": excel_path or None,
        "sql_query": sql_query or None,
    }
    with st.spinner("Agent 正在检索、调用工具并生成报告..."):
        response = requests.post(f"{API_BASE}/chat", json=payload, timeout=120)
        data = response.json()
    st.markdown(data["answer"])
    st.subheader("来源")
    st.json(data.get("sources", []))
    st.subheader("评估指标")
    st.json(data.get("metrics", {}))
    if data.get("report_path"):
        st.success(f"报告已生成：{data['report_path']}")
