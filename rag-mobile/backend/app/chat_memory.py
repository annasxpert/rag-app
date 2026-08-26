"""
chat_memory.py
--------------
Hybrid chat session memory for the Pakistan Mobile Phone Shopping Assistant.

Two backends are supported, selected via the MEMORY_BACKEND environment variable:

  MEMORY_BACKEND=memory  (default on Render)
    - Sessions stored in a Python dict in RAM.
    - Fast, zero disk I/O, no file system permissions needed.
    - Sessions are TEMPORARY — they reset when the Render dyno restarts
      or goes to sleep (free plan sleeps after 15 min of inactivity).
    - Perfect for Render free plan: small memory footprint, no disk usage.

  MEMORY_BACKEND=file  (default locally)
    - Sessions stored as JSON files in data/chat_history/.
    - Persistent across restarts.
    - Use this during local development.

Set in your .env or Render environment variables:
    MEMORY_BACKEND=memory   ← Render (recommended)
    MEMORY_BACKEND=file     ← Local dev
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CHAT_HISTORY_DIR, MAX_HISTORY_TURNS

# ─── Backend selection ───────────────────────────────────────────────────────
MEMORY_BACKEND = os.getenv("MEMORY_BACKEND", "file").strip().lower()
_USE_MEMORY = MEMORY_BACKEND == "memory"

# ─── In-memory store (used when MEMORY_BACKEND=memory) ──────────────────────
# { session_id: session_doc_dict }
_MEMORY_STORE: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()


# ─── Utilities ────────────────────────────────────────────────────────────────

def _sanitize_session_id(session_id: str | None) -> str:
    """Sanitize or generate a session ID safely to prevent directory traversal."""
    if not session_id or not isinstance(session_id, str):
        return str(uuid.uuid4())
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "", session_id.strip())
    if not cleaned:
        return str(uuid.uuid4())
    return cleaned


def _slugify(text: str, max_words: int = 6) -> str:
    """Convert question text into a clean filename-friendly slug."""
    if not text:
        return "session"
    cleaned = re.sub(r"[^\w\s-]", "", text.lower())
    words = cleaned.split()[:max_words]
    slug = "_".join(words)
    slug = re.sub(r"[_\-]+", "_", slug).strip("_")
    return slug or "session"


def _make_filename(session_id: str, title: str | None = None) -> str:
    """Generate a descriptive filename for file-based storage."""
    short_id = session_id.replace("-", "")[:8]
    if title and title != "New Shopping Chat":
        slug = _slugify(title)
        return f"chat_{slug}_{short_id}.json"
    return f"chat_session_{short_id}.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─── File-based helpers (used when MEMORY_BACKEND=file) ──────────────────────

def _find_session_file(session_id: str) -> Path | None:
    """Locate the session JSON file by session_id."""
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    safe_id = _sanitize_session_id(session_id)

    direct = CHAT_HISTORY_DIR / f"{safe_id}.json"
    if direct.is_file():
        return direct

    short_id = safe_id.replace("-", "")[:8]
    for file_path in CHAT_HISTORY_DIR.glob("*.json"):
        if short_id in file_path.stem or safe_id in file_path.stem:
            return file_path
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
                if doc.get("session_id") == session_id or doc.get("session_id") == safe_id:
                    return file_path
        except Exception:
            continue

    return None


def _file_read_session(session_id: str) -> dict[str, Any] | None:
    file_path = _find_session_file(session_id)
    if not file_path or not file_path.is_file():
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _file_write_session(doc: dict[str, Any], old_file_path: Path | None = None) -> None:
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    target_filename = _make_filename(doc["session_id"], doc.get("title"))
    target_path = CHAT_HISTORY_DIR / target_filename

    if old_file_path and old_file_path.is_file() and old_file_path != target_path:
        try:
            old_file_path.unlink()
        except OSError:
            pass

    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)


# ─── Public API ──────────────────────────────────────────────────────────────

def get_session(session_id: str) -> dict[str, Any] | None:
    """Load a session document by ID, or return None if it doesn't exist."""
    with _LOCK:
        if _USE_MEMORY:
            return _MEMORY_STORE.get(_sanitize_session_id(session_id))
        return _file_read_session(session_id)


def create_session(
    session_id: str | None = None,
    title: str = "New Shopping Chat",
) -> dict[str, Any]:
    """Create and persist a new empty session document."""
    safe_id = _sanitize_session_id(session_id)
    now = _now_iso()
    doc: dict[str, Any] = {
        "session_id": safe_id,
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    with _LOCK:
        if _USE_MEMORY:
            _MEMORY_STORE[safe_id] = doc
        else:
            filename = _make_filename(safe_id, title)
            path = CHAT_HISTORY_DIR / filename
            CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
    return doc


def get_or_create_session(session_id: str | None = None) -> tuple[str, dict[str, Any]]:
    """Retrieve an existing session or initialize a new one if missing."""
    if session_id:
        existing = get_session(session_id)
        if existing is not None:
            return existing["session_id"], existing
    new_doc = create_session(session_id=session_id)
    return new_doc["session_id"], new_doc


def save_message(
    session_id: str,
    role: str,
    content: str,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Append a single user or assistant message to the session."""
    safe_id, doc = get_or_create_session(session_id)
    msg_id = f"msg_{uuid.uuid4().hex[:10]}"
    timestamp = _now_iso()

    message: dict[str, Any] = {
        "id": msg_id,
        "role": role,
        "content": content,
        "timestamp": timestamp,
    }
    if sources is not None:
        message["sources"] = sources

    messages: list[dict[str, Any]] = doc.get("messages", [])
    messages.append(message)
    doc["messages"] = messages
    doc["updated_at"] = timestamp

    # Derive descriptive title from the first user query
    if role == "user" and (not doc.get("title") or doc.get("title") == "New Shopping Chat"):
        cleaned_title = content.strip().replace("\n", " ")
        doc["title"] = (cleaned_title[:50] + "...") if len(cleaned_title) > 50 else cleaned_title

    with _LOCK:
        if _USE_MEMORY:
            _MEMORY_STORE[safe_id] = doc
        else:
            old_file_path = _find_session_file(safe_id)
            _file_write_session(doc, old_file_path)

    return message


def save_turn(
    session_id: str,
    user_query: str,
    assistant_reply: str,
    sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Convenience helper to record a full user query + assistant reply turn."""
    save_message(session_id, role="user", content=user_query)
    save_message(session_id, role="assistant", content=assistant_reply, sources=sources)
    return get_session(session_id) or {}


def get_messages(session_id: str) -> list[dict[str, Any]]:
    """Return all messages for a session in chronological order."""
    doc = get_session(session_id)
    if not doc:
        return []
    return doc.get("messages", [])


def get_recent_history(
    session_id: str,
    max_messages: int = MAX_HISTORY_TURNS * 2,
) -> list[dict[str, str]]:
    """
    Return recent conversation history formatted for LLM consumption:
    [{"role": "user" | "assistant", "content": "..."}]
    """
    messages = get_messages(session_id)
    if not messages:
        return []
    recent = messages[-max_messages:]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def list_sessions() -> list[dict[str, Any]]:
    """List all saved chat sessions sorted by updated_at descending."""
    with _LOCK:
        if _USE_MEMORY:
            sessions = []
            for doc in _MEMORY_STORE.values():
                messages = doc.get("messages", [])
                last_preview = messages[-1]["content"] if messages else ""
                if len(last_preview) > 60:
                    last_preview = last_preview[:60] + "..."
                sessions.append(
                    {
                        "session_id": doc.get("session_id", ""),
                        "file_name": None,
                        "title": doc.get("title", "Shopping Chat"),
                        "created_at": doc.get("created_at", ""),
                        "updated_at": doc.get("updated_at", ""),
                        "message_count": len(messages),
                        "last_preview": last_preview,
                    }
                )
            sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
            return sessions

        # File-based
        CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        sessions = []
        for file_path in CHAT_HISTORY_DIR.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                    messages = doc.get("messages", [])
                    last_preview = messages[-1]["content"] if messages else ""
                    if len(last_preview) > 60:
                        last_preview = last_preview[:60] + "..."
                    sessions.append(
                        {
                            "session_id": doc.get("session_id", file_path.stem),
                            "file_name": file_path.name,
                            "title": doc.get("title", "Shopping Chat"),
                            "created_at": doc.get("created_at", ""),
                            "updated_at": doc.get("updated_at", ""),
                            "message_count": len(messages),
                            "last_preview": last_preview,
                        }
                    )
            except Exception:
                continue

        sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
        return sessions


def delete_session(session_id: str) -> bool:
    """Delete a session."""
    with _LOCK:
        safe_id = _sanitize_session_id(session_id)
        if _USE_MEMORY:
            if safe_id in _MEMORY_STORE:
                del _MEMORY_STORE[safe_id]
                return True
            return False

        file_path = _find_session_file(session_id)
        if file_path and file_path.is_file():
            try:
                file_path.unlink()
                return True
            except OSError:
                return False
    return False


def clear_session(session_id: str) -> bool:
    """Clear all messages inside a session while preserving session metadata."""
    with _LOCK:
        safe_id = _sanitize_session_id(session_id)
        if _USE_MEMORY:
            if safe_id in _MEMORY_STORE:
                _MEMORY_STORE[safe_id]["messages"] = []
                _MEMORY_STORE[safe_id]["updated_at"] = _now_iso()
                return True
            return False

        file_path = _find_session_file(session_id)
        if not file_path or not file_path.is_file():
            return False
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                doc = json.load(f)
            doc["messages"] = []
            doc["updated_at"] = _now_iso()
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False
