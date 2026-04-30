# 基于 RAG 与多工具调用的企业智能业务分析 Agent

面向咨询和业务分析场景的 AI Agent MVP，支持上传企业资料、行业报告和竞品文档，自动完成知识库检索、任务规划、工具调用、结构化分析和报告生成。

## 核心能力

- 知识库问答 RAG：支持 TXT、Markdown、PDF、Word 文档上传、切分、向量化和检索。
- 业务分析 Agent：自动识别行业分析、竞品分析、需求总结和数据分析任务。
- 多工具调用：支持网页搜索、SQLite 查询、Excel/CSV 统计分析。
- 报告生成：输出 Markdown 报告，并在安装 `python-docx` 后同步生成 Word 文档。
- Agent 评估：记录检索命中率、引用覆盖率和任务完成率。

默认实现使用本地 Hash Embedding 和 JSON 向量库，便于无 API Key 演示和测试。生产环境可替换为 OpenAI/Qwen/DeepSeek Embedding + FAISS/Chroma/Milvus，Agent 编排也可迁移到 LangChain 或 LlamaIndex。

## 项目结构

```text
ai-business-agent/
├── app/                 # FastAPI 应用与接口
├── agent/               # 任务规划、记忆、执行器、Prompt
├── rag/                 # 文档加载、切分、向量化、检索
├── tools/               # 搜索、SQL、Excel、报告工具
├── eval/                # 指标、反馈、自动评测脚本
├── ui/                  # Streamlit 前端
├── data/                # 原始文档、向量库、报告输出
└── tests/               # 单元测试
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

打开接口文档：

```text
http://localhost:8000/docs
```

启动 Streamlit：

```bash
streamlit run ui/streamlit_app.py
```

## API 示例

上传文档：

```bash
curl -F "file=@data/raw_docs/company.txt" http://localhost:8000/api/upload
```

发起分析：

```bash
curl -X POST http://localhost:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"请分析A公司与主要竞品的差异化机会\",\"generate_report\":true}"
```

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SEARCH_ENABLED` | `false` | 是否启用 DuckDuckGo HTML 搜索 |
| `SEARCH_MAX_RESULTS` | `5` | 搜索结果数量 |
| `EMBEDDING_DIM` | `256` | 本地 Hash Embedding 维度 |
| `CHUNK_SIZE` | `900` | 文档切分长度 |
| `CHUNK_OVERLAP` | `120` | 文档切分重叠长度 |

## 测试

```bash
pytest
```

## Docker

```bash
docker build -t ai-business-agent .
docker run -p 8000:8000 ai-business-agent
```

## 描述

开发企业智能业务分析 Agent，集成 RAG、多工具调用和自动报告生成能力；支持 PDF/Word/TXT 文档入库、基于企业资料问答、竞品与行业分析、SQL/Excel 数据分析，并通过检索命中率、引用覆盖率和任务完成率评估回答质量。

