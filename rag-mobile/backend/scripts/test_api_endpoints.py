"""
test_api_endpoints.py
---------------------
Tests FastAPI chat endpoints with test client to verify session management,
JSON persistence, and response structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from fastapi.testclient import TestClient
from app.main import app
from app.chat_memory import delete_session
from config import CHAT_HISTORY_DIR

client = TestClient(app)

def test_full_chat_flow():
    # 1. Start a chat
    resp1 = client.post("/chat", json={
        "question": "What is the price of Tecno Spark 20?",
    })
    print("Response 1 status:", resp1.status_code)
    assert resp1.status_code == 200, f"Error: {resp1.text}"
    data1 = resp1.json()
    session_id = data1["session_id"]
    print("Created session_id:", session_id)
    print("Answer 1 preview:", data1["answer"][:100], "...")
    print("Suggestions 1:", data1.get("suggestions", []))
    assert "suggestions" in data1 and len(data1["suggestions"]) > 0, "Expected suggestions list"

    # 2. Check that the session JSON file exists in data/chat_history/
    from app.chat_memory import _find_session_file
    session_file = _find_session_file(session_id)
    assert session_file is not None and session_file.is_file(), f"Session file for {session_id} not found!"
    print("Found descriptive session file:", session_file.name)
    with open(session_file, "r", encoding="utf-8") as f:
        saved_doc = json.load(f)
    print("JSON file content verification:")
    print("  Title:", saved_doc["title"])
    print("  Messages count:", len(saved_doc["messages"]))
    assert len(saved_doc["messages"]) == 2, "Expected 2 messages (1 user, 1 assistant)"

    # 3. Ask a follow-up in the same session
    resp2 = client.post("/chat", json={
        "question": "Is it good for gaming and what about its RAM?",
        "session_id": session_id,
    })
    assert resp2.status_code == 200, f"Error: {resp2.text}"
    data2 = resp2.json()
    assert data2["session_id"] == session_id, "Session ID should remain consistent"
    print("Answer 2 preview:", data2["answer"][:100], "...")
    print("Suggestions 2:", data2.get("suggestions", []))
    assert "suggestions" in data2 and len(data2["suggestions"]) > 0, "Expected suggestions list in turn 2"

    # 4. Check JSON document now has 4 messages
    session_file = _find_session_file(session_id)
    assert session_file is not None and session_file.is_file()
    with open(session_file, "r", encoding="utf-8") as f:
        saved_doc2 = json.load(f)
    print("  Updated Messages count:", len(saved_doc2["messages"]))
    assert len(saved_doc2["messages"]) == 4, "Expected 4 messages after 2 turns"

    # 5. Test GET /sessions
    list_resp = client.get("/sessions")
    assert list_resp.status_code == 200
    sessions_list = list_resp.json().get("sessions", [])
    assert any(s["session_id"] == session_id for s in sessions_list), "Session not found in list"
    print("GET /sessions found active session successfully.")

    # 6. Test GET /sessions/{session_id}
    detail_resp = client.get(f"/sessions/{session_id}")
    assert detail_resp.status_code == 200
    detail_data = detail_resp.json()
    assert detail_data["session_id"] == session_id
    assert len(detail_data["messages"]) == 4
    print("GET /sessions/{session_id} returned all messages.")

    # 7. Clean up test session
    delete_session(session_id)
    print("Test session cleaned up successfully.")
    print("\n ALL ENDPOINT & JSON STORAGE TESTS PASSED!")

if __name__ == "__main__":
    test_full_chat_flow()
