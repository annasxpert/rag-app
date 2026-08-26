"""
chat_service.py
----------------
Orchestrates the full Query / Retrieval Pipeline for a single chat request
with multi-turn conversation memory and JSON persistence.

This is the ONLY place that ties retrieval + history + generation together
for the API layer — main.py stays a thin HTTP wrapper around this function.

    question + session_id -> chat_memory (load recent turns)
                          -> contextual retrieval -> retriever.retrieve()
                          -> retriever.build_context()
                          -> llm_client.generate_answer(history=history)
                          -> chat_memory (save turn to JSON)
                          -> return answer + sources + session_id
"""

from __future__ import annotations

import re

from app.chat_memory import get_or_create_session, get_recent_history, save_turn
from retrieval.llm_client import generate_answer
from retrieval.retriever import build_context, retrieve

_FOLLOWUP_INDICATORS = {
    "which", "what about", "how about", "compare", "difference", "first one",
    "second one", "both", "better", "cheaper", "battery", "camera", "gaming",
    "screen", "display", "processor", "charging", "ram", "storage", "iska",
    "uski", "dono", "konsa", "kisme", "pehle", "dusra", "aur koi", "yeh",
    "woh", "ye", "wo"
}


def _enrich_retrieval_query(question: str, history: list[dict[str, str]]) -> str:
    """
    If the user's question is a follow-up (e.g. 'Which one has a better battery?'
    or 'Iska camera kaisa hai?'), augment the search query with context from
    the preceding turn so vector search surfaces the relevant phone specs.
    """
    if not history:
        return question

    q_lower = question.lower().strip()
    words = set(re.findall(r"\w+", q_lower))

    is_followup = (
        len(words) <= 7
        or any(ind in q_lower for ind in _FOLLOWUP_INDICATORS)
        or ("?" in question and len(words) <= 5)
    )

    if not is_followup:
        return question

    # Extract the most recent user question or a short summary from previous turns
    last_user_message = ""
    for msg in reversed(history):
        if msg.get("role") == "user":
            last_user_message = msg.get("content", "").strip()
            break

    if last_user_message and last_user_message != question:
        return f"{last_user_message} {question}"

    return question


def answer_question(
    question: str,
    top_k: int = 5,
    provider: str | None = None,
    model: str | None = None,
    session_id: str | None = None,
) -> dict:
    """
    Process a user query, load previous chat history from JSON storage,
    retrieve grounded catalog specs, generate a natural response via LLM,
    and persist the conversation turn.
    """
    question = question.strip()
    session_id, _ = get_or_create_session(session_id)

    if not question:
        return {
            "session_id": session_id,
            "answer": "Please type a question about a phone or budget.",
            "sources": [],
        }

    # 1. Fetch recent history for multi-turn conversational context
    history = get_recent_history(session_id)

    # 2. Enrich query for vector search if it's a follow-up question
    search_query = _enrich_retrieval_query(question, history)

    # 3. Retrieve relevant phone chunks from Pinecone vector store
    chunks = retrieve(search_query, top_k=top_k)
    context = build_context(chunks)

    # 4. Generate grounded natural answer and follow-up suggestions
    answer, suggestions = generate_answer(
        question=question,
        context=context,
        provider=provider,
        model=model,
        history=history,
    )

    # 5. Save the user question and assistant answer to the session JSON file
    save_turn(
        session_id=session_id,
        user_query=question,
        assistant_reply=answer,
    )

    return {
        "session_id": session_id,
        "answer": answer,
        "suggestions": suggestions,
    }
