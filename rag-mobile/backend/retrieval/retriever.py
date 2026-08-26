"""
retriever.py
------------
Part of the Query / Retrieval Pipeline: RETRIEVAL + CONTEXT CONSTRUCTION.

    User Question -> query_embedder.embed_query() -> vector_store.search()
                   -> top-k chunks -> build_context() -> (text handed to LLM)

This module does NOT call the LLM. Its only job is: given a question,
return the most relevant knowledge chunks (each one a phone's spec sheet)
and format them into a single context block the LLM client can drop into
a prompt. Keeping this separate from llm_client.py means retrieval logic
(and any future reranking step) can be tested and swapped independently
of which LLM provider generates the final answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import TOP_K
from retrieval.query_embedder import embed_query
from retrieval.vector_store import search


@dataclass
class RetrievedChunk:
    text: str
    row_id: str          # ground-truth phone id — used by eval/ to check retrieval accuracy
    brand: str
    model: str
    relevance_score: float  # Pinecone cosine similarity, higher = more relevant


def retrieve(question: str, top_k: int = TOP_K, metadata_filter: dict | None = None) -> list[RetrievedChunk]:
    """Embed the question and fetch the top_k most similar phone chunks from Pinecone."""
    query_vector = embed_query(question)
    hits = search(query_vector, top_k=top_k, metadata_filter=metadata_filter)

    retrieved = []
    for hit in hits:
        meta = hit["metadata"]
        retrieved.append(
            RetrievedChunk(
                text=hit["text"],
                row_id=str(meta.get("row_id", "")),
                brand=meta.get("brand", ""),
                model=meta.get("model", ""),
                relevance_score=round(float(hit["score"]), 3),
            )
        )
    return retrieved


def build_context(chunks: list[RetrievedChunk]) -> str:
    """
    Assemble retrieved chunks into a single labeled context block for the LLM.
    Optional reranking (e.g. a cross-encoder) would slot in here, between
    retrieve() and build_context(), without changing either function's signature.
    """
    if not chunks:
        return ""

    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(f"[Phone {i}]\n{chunk.text}")
    return "\n\n".join(blocks)
