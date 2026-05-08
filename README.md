# AI Business Analysis Agent

FastAPI and Streamlit MVP for business analysis workflows. The app combines document ingestion, local RAG retrieval, task planning, optional web search, SQLite read-only analysis, CSV/XLSX summarization, and Markdown/DOCX report generation.

## Project Layout

```text
app/       FastAPI application and routes
agent/     Planning, memory, and execution orchestration
rag/       Loaders, splitting, embeddings, retrieval, and vector storage
tools/     Web search, SQLite, Excel/CSV, and report tools
ui/        Streamlit client
data/      Raw documents, processed vector data, and generated reports
tests/     Unit and API tests
```

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Run the UI in another terminal:

```bash
streamlit run ui/streamlit_app.py
```

The Streamlit app uses `API_BASE_URL` when set and otherwise defaults to:

```text
http://localhost:8000/api/v1
```

If you are running an older API route layout, override the sidebar API Base URL or set `API_BASE_URL=http://localhost:8000/api`.

## Environment

Copy `.env.example` to `.env` and adjust as needed.

| Variable | Default | Description |
| --- | --- | --- |
| `SEARCH_ENABLED` | `false` | Enables live DuckDuckGo HTML search. |
| `SEARCH_MAX_RESULTS` | `5` | Maximum returned web search results. |
| `EMBEDDING_DIM` | `256` | Local hash embedding dimension. |
| `CHUNK_SIZE` | `900` | Document chunk size. |
| `CHUNK_OVERLAP` | `120` | Document chunk overlap. |
| `RAG_TOP_K` | `5` | Retrieval result count. |
| `API_BASE_URL` | `http://localhost:8000/api/v1` | Streamlit API base URL. |
| `OPENAI_API_KEY` | unset | Enables OpenAI-compatible LLM/embedding mode when model names are also set. |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible API base URL. |
| `CHAT_MODEL` | unset | Chat model used by the LLM answer synthesizer. |
| `EMBEDDING_MODEL` | unset | Embedding model used with the Chroma vector store. |

When `OPENAI_API_KEY`, `CHAT_MODEL`, and `EMBEDDING_MODEL` are all configured, the app uses OpenAI-compatible embeddings, Chroma persistence, and LLM synthesis. Without them, it stays on the deterministic local hash embedding and template synthesizer path used by tests.

## Docker

Build and run the API only:

```bash
docker build -t ai-business-agent .
docker run --rm -p 8000:8000 -v "%cd%\data:/app/data" ai-business-agent
```

Run API and UI together:

```bash
docker compose up --build
```

Services:

- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

Both services mount `./data:/app/data`. In Docker Compose, the UI uses `http://api:8000/api/v1`.

## Tool Safety

- SQLite queries are opened read-only, checked for `SELECT`/`WITH`, protected by a SQLite authorizer, and capped by `max_rows`.
- SQL, Excel/CSV, and search failures return structured tool errors instead of uncaught exceptions.
- Excel analysis supports `.csv` and `.xlsx`. Legacy `.xls` files are rejected unless explicit `xlrd` support is added later.
- Web search catches request failures and reports them in the tool result.

## Tests and Quality

```bash
pytest
pytest --cov=app --cov=agent --cov=rag --cov=tools --cov=eval --cov-report=term-missing --cov-fail-under=80
ruff check .
ruff format --check .
mypy app agent rag tools eval
```

Focused tool tests:

```bash
pytest tests/test_tools.py
```
