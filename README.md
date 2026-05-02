# RAG with Semantic Cache (Financial Compliance)

This is a production-minded financial compliance RAG system that ingests regulatory PDFs, indexes them with hybrid retrieval, serves grounded answers over a streaming API, and speeds up repeated questions with semantic caching.

<p align="center">
  <img src="docs/images/system-design.svg" alt="System design overview for the financial compliance RAG system" width="1400">
</p>

<p align="center">
  <img src="docs/images/request-flow.svg" alt="Runtime request flow showing cache-hit and cache-miss paths" width="1400">
</p>

## Why This Project Is Interesting

This repository is more than a chatbot demo. It is a complete Retrieval-Augmented Generation system built for financial compliance use cases, especially over SEBI and related securities-market documents.

What makes it stand out:

- Hybrid retrieval combines dense and sparse embeddings, which is a better fit for regulation-heavy text than dense-only search
- Query rewriting happens before cache lookup, improving both retrieval quality and cache reuse
- Semantic caching reduces repeated LLM work for equivalent or near-equivalent questions
- Conversation history is persisted in Postgres so follow-up questions stay contextual
- Responses are streamed from FastAPI for a better interactive experience
- The retriever is evaluated offline with saved benchmark artifacts instead of only manual testing

## What This Repository Implements

This project currently includes:

- A PDF ingestion pipeline for financial and regulatory documents
- Chunking with overlap for better retrieval continuity
- Hybrid search on Pinecone using OpenAI dense embeddings and SPLADE sparse embeddings
- A LangGraph-based orchestration layer for rewrite, cache, retrieval, memory, and answer generation
- Multi-provider LLM support through a strategy layer for `openai` and `nvidia`
- Semantic answer caching through Redis LangCache
- Session-based conversation memory in Postgres
- A streaming FastAPI backend
- A Streamlit chat interface for local usage
- An offline 4-stage retriever evaluation pipeline with `ranx`
- Dockerfiles for backend and frontend packaging

## Project Snapshot

| Area | Detail |
| --- | --- |
| Domain | Financial compliance / regulatory QA |
| Corpus | 41 PDF documents in [`documents/`](documents/) |
| Backend | FastAPI |
| Workflow orchestration | LangGraph |
| Vector store | Pinecone |
| Dense embeddings | OpenAI `text-embedding-3-small` |
| Sparse embeddings | `naver/splade-cocondenser-ensembledistil` |
| Conversation memory | Postgres via `asyncpg` |
| Semantic cache | Redis LangCache |
| Frontend | Streamlit |
| Offline benchmark | 100-query retriever evaluation |

## How the Corpus Is Indexed

The ingestion side of the project is intentionally simple and production-friendly:

- PDF files are loaded page-by-page from [`documents/`](documents/)
- Text is split with a `1000` character chunk size and `200` character overlap
- Each chunk gets both a dense embedding and a sparse embedding
- The final hybrid vector payload is upserted into Pinecone for retrieval at runtime

## Core Design

### 1. Query Rewrite Before Retrieval

The system first rewrites the user's question into a cleaner retrieval-friendly query. That rewritten query then becomes the basis for cache lookup and retrieval, which helps normalize similar user phrasing.

### 2. Semantic Cache in Front of Full RAG

If the rewritten query has already been answered, the system can return the cached answer immediately instead of re-running retrieval and generation. This is the core performance differentiator of the project.

### 3. Hybrid Retrieval for Compliance Text

Financial and regulatory documents often need both semantic understanding and exact terminology matching. This project uses:

- Dense embeddings for semantic similarity
- Sparse embeddings for keyword-sensitive retrieval

That combination is especially useful for queries involving circulars, obligations, penalties, reporting rules, and regulatory definitions.

### 4. Stateful Conversations

The backend stores messages per `session_id` in Postgres and replays recent history into the final-answer prompt. That makes the assistant usable for multi-turn compliance research, not just isolated one-shot questions.

## Retrieval Evaluation

The repository includes a reproducible evaluation workflow under [`offline_retriever_eval/`](offline_retriever_eval/):

1. Sample 100 chunks from Pinecone
2. Generate user-style queries for those chunks
3. Run the hybrid retriever
4. Score the run with `ranx`

Latest checked-in metrics from [`offline_retriever_eval/GoldenDataset/retrieval_metrics.csv`](offline_retriever_eval/GoldenDataset/retrieval_metrics.csv):

| Metric | Value |
| --- | ---: |
| `hit_rate@1` | `0.91` |
| `hit_rate@3` | `0.97` |
| `hit_rate@5` | `0.98` |
| `hit_rate@10` | `1.00` |
| `recall@10` | `1.00` |
| `precision@10` | `0.10` |
| `mrr@10` | `0.9413` |
| `map@10` | `0.9413` |
| `ndcg@10` | `0.9556` |

Why `precision@10` looks low: the current benchmark treats the originating chunk as the only relevant document per query, so even a perfect top-10 result naturally caps precision at `0.10`. In this setup, `hit_rate`, `MRR`, `recall`, and `nDCG` are more informative.

## Repository Structure

```text
.
├── app/                       # FastAPI app and routes
├── rag_src/                   # RAG workflow, nodes, prompts, repositories, LLM strategies
├── DocumentIngestion/         # Ingestion and chunking pipeline
├── offline_retriever_eval/    # Retriever benchmark workflow and artifacts
├── streamlit_chat_ui/         # Streamlit frontend
├── documents/                 # Financial compliance PDF corpus
├── Dockerfile.backend         # Backend image
├── Dockerfile.frontend        # Frontend image
├── pyproject.toml             # Dependency groups and project metadata
└── README.md
```

## Important Files

- [`app/main.py`](app/main.py): FastAPI app startup and dependency wiring
- [`app/routes/chat.py`](app/routes/chat.py): streaming `/chat` endpoint
- [`rag_src/graph.py`](rag_src/graph.py): LangGraph workflow definition
- [`rag_src/nodes.py`](rag_src/nodes.py): rewrite, cache, retrieval, memory, and answer nodes
- [`rag_src/services.py`](rag_src/services.py): orchestration service layer
- [`rag_src/repositories/pinecone_repository.py`](rag_src/repositories/pinecone_repository.py): hybrid Pinecone retrieval and upsert logic
- [`rag_src/repositories/conversation_db.py`](rag_src/repositories/conversation_db.py): Postgres-backed conversation store
- [`rag_src/repositories/lang_cache.py`](rag_src/repositories/lang_cache.py): Redis LangCache adapter
- [`DocumentIngestion/main.py`](DocumentIngestion/main.py): ingestion pipeline entry point
- [`streamlit_chat_ui/app.py`](streamlit_chat_ui/app.py): local chat UI

## Quick Start

### Prerequisites

- Python `3.12`
- `uv`
- Postgres
- Pinecone account and API key
- OpenAI API key
- NVIDIA API key
- Optional LangCache credentials for semantic caching

### Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes | Dense embeddings and OpenAI-backed LLM workflow |
| `PINECONE_API_KEY` | Yes | Pinecone access |
| `DATABASE_URL` | Yes | Postgres connection for conversation history |
| `NVIDIA_API_KEY` | Yes | NVIDIA-backed LLM workflow |
| `LANGCACHE_API_KEY` | No | Semantic cache API key |
| `LANGCACHE_SERVER_URL` | No | Semantic cache backend URL |
| `LANGCACHE_CACHE_ID` | No | Semantic cache identifier |
| `LANGCACHE_TTL_SECONDS` | No | Cache TTL override |
| `LANGCACHE_FAILURE_COOLDOWN_SECONDS` | No | Cache failure cooldown |
| `DATABASE_POOL_MIN_SIZE` | No | Postgres pool minimum |
| `DATABASE_POOL_MAX_SIZE` | No | Postgres pool maximum |

Note: the backend currently initializes both `openai` and `nvidia` workflows at startup, so both provider keys should be set in the current implementation.

### Install Dependencies

```bash
uv sync
```

### Ingest the Documents

```bash
uv run python DocumentIngestion/main.py
```

This pipeline:

- Reads PDF files from [`documents/`](documents/)
- Loads them page-by-page
- Splits them into overlapping chunks
- Generates dense and sparse embeddings
- Upserts hybrid vectors into Pinecone

### Run the Backend

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run the Frontend

```bash
uv run streamlit run streamlit_chat_ui/app.py
```

Then open [http://127.0.0.1:8501](http://127.0.0.1:8501).

## Docker

### Build

```bash
docker build -f Dockerfile.backend -t rag-semantic-cache-backend .
docker build -f Dockerfile.frontend -t rag-semantic-cache-frontend .
```

### Run Backend

```bash
docker run --rm --env-file .env -p 8000:8000 rag-semantic-cache-backend
```

### Run Frontend

```bash
docker run --rm -e BACKEND_CHAT_URL=http://host.docker.internal:8000/chat -p 8501:8501 rag-semantic-cache-frontend
```

If you are not on macOS or Windows, replace `host.docker.internal` with a reachable host address.

## API Contract

### `POST /chat`

Request body:

```json
{
  "payload": "What are the disclosure requirements for listed entities?",
  "session_id": "demo-session-001",
  "llm": "nvidia"
}
```

Accepted `llm` values:

- `nvidia`
- `openai`

Behavior:

- Streams the answer as plain text
- Uses recent session history for continuity
- Returns cached answers when a semantic cache hit is found

## Why I Built It This Way

- Financial compliance queries are terminology-heavy, so hybrid retrieval is a better fit than dense-only search
- Repeated regulatory questions are common, so semantic caching gives a real latency and cost advantage
- Compliance workflows are often iterative, so persistent conversation history matters
- Streaming improves perceived responsiveness, especially when generation takes longer than retrieval
- Evaluation artifacts make the project more credible than a demo that only "looks good" interactively

## Next Steps

Strong next improvements for this codebase would be:

- Add citations with source file and page references in final answers
- Add automated tests for graph nodes, repositories, and API behavior
- Add a `docker-compose.yml` for one-command local startup
- Lazy-load LLM providers so the backend can boot with only one configured
- Add observability around latency, cache hit rate, and retrieval quality over time

## Summary

If someone lands on this repository, the story should be immediately clear:

- this is a real RAG system, not a notebook prototype
- it is built for financial compliance document intelligence
- semantic caching is a first-class part of the design
- retrieval quality has been measured, not guessed
- the architecture is modular enough to evolve into a production service
