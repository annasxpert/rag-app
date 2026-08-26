"""
test_chat_memory.py
-------------------
Automated unit and integration tests for JSON persistent chat memory,
session management, and multi-turn query execution.
"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.chat_memory import (
    clear_session,
    create_session,
    delete_session,
    get_messages,
    get_or_create_session,
    get_recent_history,
    get_session,
    list_sessions,
    save_message,
    save_turn,
)
from app.chat_service import _enrich_retrieval_query
from config import CHAT_HISTORY_DIR


class TestChatMemory(unittest.TestCase):
    def setUp(self):
        self.test_session_id = "test_sess_001"
        delete_session(self.test_session_id)

    def tearDown(self):
        delete_session(self.test_session_id)

    def test_create_and_get_session(self):
        doc = create_session(session_id=self.test_session_id, title="Test Session")
        self.assertEqual(doc["session_id"], self.test_session_id)
        self.assertEqual(doc["title"], "Test Session")
        self.assertEqual(doc["messages"], [])

        loaded = get_session(self.test_session_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["session_id"], self.test_session_id)

    def test_save_and_retrieve_messages(self):
        get_or_create_session(self.test_session_id)

        # 1. Save user message
        msg1 = save_message(
            session_id=self.test_session_id,
            role="user",
            content="Show me phones under 50k",
        )
        self.assertEqual(msg1["role"], "user")
        self.assertEqual(msg1["content"], "Show me phones under 50k")

        # 2. Save assistant reply with sources
        sources = [{"model": "Tecno Spark 20", "relevance": 0.85}]
        msg2 = save_message(
            session_id=self.test_session_id,
            role="assistant",
            content="Here are the best options under 50k...",
            sources=sources,
        )
        self.assertEqual(msg2["role"], "assistant")
        self.assertEqual(msg2["sources"], sources)

        # 3. Check full message list
        messages = get_messages(self.test_session_id)
        self.assertEqual(len(messages), 2)

        # 4. Check recent history formatted for LLM
        history = get_recent_history(self.test_session_id, max_messages=10)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"role": "user", "content": "Show me phones under 50k"})
        self.assertEqual(history[1], {"role": "assistant", "content": "Here are the best options under 50k..."})

        # 5. Check descriptive JSON file directly on disk
        from app.chat_memory import _find_session_file
        file_path = _find_session_file(self.test_session_id)
        self.assertIsNotNone(file_path)
        self.assertTrue(file_path.is_file())
        self.assertTrue(file_path.name.startswith("chat_"))
        with open(file_path, "r", encoding="utf-8") as f:
            disk_doc = json.load(f)
            self.assertEqual(len(disk_doc["messages"]), 2)
            self.assertEqual(disk_doc["title"], "Show me phones under 50k")

    def test_save_turn_helper(self):
        save_turn(
            session_id=self.test_session_id,
            user_query="Samsung phone under 80k",
            assistant_reply="Samsung Galaxy A25 is a solid choice.",
            sources=[{"model": "Samsung Galaxy A25", "relevance": 0.89}],
        )
        messages = get_messages(self.test_session_id)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["content"], "Samsung phone under 80k")
        self.assertEqual(messages[1]["content"], "Samsung Galaxy A25 is a solid choice.")

    def test_clear_and_delete_session(self):
        save_turn(
            session_id=self.test_session_id,
            user_query="Hello",
            assistant_reply="Hi there!",
        )
        self.assertEqual(len(get_messages(self.test_session_id)), 2)

        # Clear
        cleared = clear_session(self.test_session_id)
        self.assertTrue(cleared)
        self.assertEqual(len(get_messages(self.test_session_id)), 0)

        # Delete
        deleted = delete_session(self.test_session_id)
        self.assertTrue(deleted)
        self.assertIsNone(get_session(self.test_session_id))

    def test_query_enrichment_for_followup(self):
        history = [
            {"role": "user", "content": "Show me Redmi 13C and Vivo Y03"},
            {"role": "assistant", "content": "Both are great budget devices..."},
        ]
        enriched = _enrich_retrieval_query("Which one has the better camera?", history)
        self.assertIn("Redmi 13C", enriched)
        self.assertIn("better camera", enriched)


if __name__ == "__main__":
    unittest.main()
