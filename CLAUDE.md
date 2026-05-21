# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A full-stack RAG (Retrieval-Augmented Generation) chatbot for querying course materials. Users ask questions; the system uses semantic search over course documents and Claude to generate answers with cited sources.

## Environment Setup

Requires Python 3.13+ and `uv`. Set `ANTHROPIC_API_KEY` in a `.env` file (see `.env.example`).

**Always use `uv` — never `pip` or `python` directly.**

```bash
uv sync          # install dependencies
uv add <pkg>     # add a new package
uv run <script>  # run any script
```

## Running the App

```bash
./run.sh
# or manually:
cd backend && uv run uvicorn app:app --reload --port 8000
```

- Web UI: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## No Test Suite

There are currently no automated tests. Validate changes by running the app and exercising the `/api/query` endpoint.

## Architecture

**RAG Pipeline (backend/):**

```
User Query
  → FastAPI /api/query  (app.py)
  → RAGSystem.query()   (rag_system.py)   ← orchestrator
  → AIGenerator         (ai_generator.py) ← Claude claude-sonnet-4-20250514 with tool use
  → CourseSearchTool    (search_tools.py) ← called by Claude as a tool
  → VectorStore.search  (vector_store.py) ← ChromaDB semantic search
  → Return answer + sources to frontend
```

**Key modules:**

- `rag_system.py` — Top-level orchestrator. Loads docs from `docs/` at startup, delegates to all other modules.
- `ai_generator.py` — Wraps the Anthropic API. Uses tool calling so Claude decides when to search. Contains `MockAIGenerator` fallback (no API key needed) that directly formats search results.
- `vector_store.py` — ChromaDB wrapper with two collections: `course_catalog` (course metadata) and `course_content` (text chunks). Persists to `backend/chroma_db/`.
- `document_processor.py` — Parses course documents expecting this format: `Title:`, `Link:`, `Instructor:`, then `Lesson N:` markers. Chunks text at 800 chars with 100-char overlap.
- `search_tools.py` — Defines `CourseSearchTool` (schema + execution) for Claude's tool-use API and `ToolManager` to dispatch tool calls.
- `session_manager.py` — Per-session conversation history (max 10 messages, configurable).
- `models.py` — Pydantic models: `Course`, `Lesson`, `CourseChunk`.
- `config.py` — Centralized configuration (model name, chunk sizes, collection names, etc.).

**Frontend (frontend/):** Vanilla JS/HTML/CSS, no build step. Uses `marked.js` for markdown rendering.

## Important Behaviors

- **Tool-calling search:** Claude calls `CourseSearchTool` at its own discretion — it doesn't always search. The system prompt in `ai_generator.py` governs when Claude decides to invoke the tool.
- **Document ingestion:** `docs/` folder contents are loaded automatically on startup via `RAGSystem.add_course_folder()`. Adding a new `.txt` course file there is the standard way to add content.
- **Offline mode:** If `ANTHROPIC_API_KEY` is absent or `MockAIGenerator` is used, the system still returns formatted search results without LLM generation.
- **Embeddings model:** `all-MiniLM-L6-v2` via `sentence-transformers` — loaded locally, no API key required for search.
