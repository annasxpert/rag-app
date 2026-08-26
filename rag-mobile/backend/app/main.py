"""
main.py
-------
FastAPI application: the Chat / API layer.

This layer ONLY does HTTP plumbing and session routing. It reads from the
already-built Pinecone vector database via chat_service.answer_question()
and manages chat session storage (in-memory on Render, file-based locally).

Run locally:
    cd backend
    uvicorn app.main:app --reload
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Ensure backend root directory is in sys.path for imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.chat_memory import (
    clear_session,
    create_session,
    delete_session,
    get_session,
    list_sessions,
    MEMORY_BACKEND,
)
from app.chat_service import answer_question
from retrieval.llm_client import get_available_models_for_provider, get_default_provider_and_model
from retrieval.vector_store import index_size

APP_DIR = Path(__file__).resolve().parent

# ── Frontend path resolution (single-port unified serving) ─────────────────────
FRONTEND_LOCATIONS = [
    BASE_DIR.parent / "frontend" / "index.html",
    BASE_DIR / "frontend" / "index.html",
    BASE_DIR / "static" / "index.html",
    Path("/home/ubuntu/rag-mobile-assistant/rag-mobile/frontend/index.html"),
]


def _find_frontend_file() -> Path | None:
    for path in FRONTEND_LOCATIONS:
        if path.is_file():
            return path
    return None


# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow any origin by default so your Netlify frontend can connect immediately.
# Optionally restrict with FRONTEND_URL env var for production hardening.
_frontend_url = os.getenv("FRONTEND_URL", "").strip().rstrip("/")
_allowed_origins = [_frontend_url] if _frontend_url else ["*"]


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log configuration, warm up embedding model, and verify Pinecone connectivity on startup."""
    print("=" * 60)
    print("  Pakistan Mobile Assistant — Backend Starting")
    print(f"  Memory backend : {MEMORY_BACKEND.upper()}")
    print(f"  CORS origins   : {_allowed_origins}")
    
    # 1. Warm up embedding model
    try:
        from retrieval.query_embedder import embed_query
        _ = embed_query("ping")
        print("  Query embedder : ✅ Ready (local model / API cached)")
    except Exception as e:
        print(f"  Query embedder : ⚠️  Notice — {e}")

    # 2. Check Pinecone index
    try:
        count = index_size()
        if count > 0:
            print(f"  Pinecone index : ✅ OK ({count} vectors)")
        else:
            print("  Pinecone index : ⚠️  Empty — run ingestion_pipeline/run_ingestion.py locally")
    except Exception as e:
        print(f"  Pinecone index : ❌ Error — {e}")
    print("=" * 60)
    yield  # Application runs here
    # Shutdown logic (if needed in future) goes after yield


app = FastAPI(
    title="Pakistan Mobile Phone Shopping Assistant",
    description=(
        "A RAG-based domain assistant grounded in a Pakistani mobile phone catalog, "
        "backed by Pinecone vector search and a pluggable LLM (Groq / Gemini / Mistral)."
    ),
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if directory exists
if (APP_DIR / "static").is_dir():
    app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")

_frontend_index = _find_frontend_file()
if _frontend_index and _frontend_index.parent.is_dir():
    app.mount("/ui-static", StaticFiles(directory=_frontend_index.parent), name="ui_static")


# ── Pydantic models ───────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    provider: str | None = None
    model: str | None = None
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    suggestions: list[str] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    """Serve the frontend chat UI if index.html exists, or return API status JSON."""
    frontend_path = _find_frontend_file()
    if frontend_path:
        return FileResponse(frontend_path, media_type="text/html")
    return {
        "status": "online",
        "service": "Pakistan Mobile Phone Shopping Assistant API",
        "version": "1.2.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/api")
def api_info():
    """Return backend API status and documentation endpoints."""
    return {
        "status": "online",
        "service": "Pakistan Mobile Phone Shopping Assistant API",
        "version": "1.2.0",
        "memory_backend": MEMORY_BACKEND,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "chat": "POST /chat",
            "llm_config": "GET /llm-config",
            "sessions": "GET /sessions",
            "session_detail": "GET /sessions/{session_id}",
            "health": "GET /health",
        },
    }


@app.get("/llm-config")
def llm_config():
    """Return the dynamic provider/model catalog for the UI."""
    default_provider, default_model = get_default_provider_and_model()
    providers = {
        provider: get_available_models_for_provider(provider)
        for provider in ("groq", "gemini", "mistral")
    }
    return {
        "providers": providers,
        "default_provider": default_provider,
        "default_model": default_model,
    }


@app.get("/health")
def health_check():
    """Quick check that the Pinecone index is populated."""
    try:
        count = index_size()
        return {
            "status": "ok" if count > 0 else "vector index is empty — run ingestion first",
            "indexed_chunks": count,
            "memory_backend": MEMORY_BACKEND,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "detail": str(e)},
        )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint. Runs the full query pipeline:
    query enrichment -> Pinecone vector search -> context ->
    LLM generation -> session memory -> response.
    """
    result = answer_question(
        question=request.question,
        provider=request.provider,
        model=request.model,
        session_id=request.session_id,
    )
    return result


@app.get("/sessions")
def get_all_sessions():
    """List all active chat sessions."""
    return {"sessions": list_sessions(), "memory_backend": MEMORY_BACKEND}


@app.get("/sessions/{session_id}")
def get_session_details(session_id: str):
    """Retrieve full message history for a given session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    return session


@app.post("/sessions/new")
def new_session():
    """Create a new chat session and return its metadata."""
    doc = create_session()
    return doc


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    """Delete a session."""
    deleted = delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found or could not be deleted.")
    return {"status": "deleted", "session_id": session_id}


@app.post("/sessions/{session_id}/clear")
def clear_session_messages(session_id: str):
    """Clear message history in a session."""
    cleared = clear_session(session_id)
    if not cleared:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"status": "cleared", "session_id": session_id}
