from __future__ import annotations

from functools import lru_cache

from config import (
    EMBEDDING_DIM,
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_REGION,
)

UPSERT_BATCH_SIZE = 100


@lru_cache(maxsize=1)
def _get_client():
    if not PINECONE_API_KEY:
        raise RuntimeError(
            "PINECONE_API_KEY is not set. Copy .env.example to .env and add your Pinecone API key "
            "(https://app.pinecone.io/ -> API Keys)."
        )
    from pinecone import Pinecone

    return Pinecone(api_key=PINECONE_API_KEY)


def ensure_index_exists() -> None:
    """Create the Pinecone serverless index if it doesn't exist yet. Idempotent."""
    from pinecone import ServerlessSpec

    pc = _get_client()
    existing = [idx["name"] for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )


def reset_index() -> None:
    """Delete all vectors in the namespace — used when re-running ingestion from scratch."""
    ensure_index_exists()
    index = _get_client().Index(PINECONE_INDEX_NAME)
    try:
        index.delete(delete_all=True, namespace=PINECONE_NAMESPACE)
    except Exception:
        pass  # namespace may not exist yet on a brand-new index — fine


def upsert_chunks(ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
    """
    WRITE path — used only by the ingestion pipeline to populate the index.
    Batches upserts (Pinecone recommends <= a few hundred vectors per call).
    The chunk text itself is stored in metadata under "text" so retrieval
    can read it straight back without a second lookup.
    """
    ensure_index_exists()
    index = _get_client().Index(PINECONE_INDEX_NAME)

    vectors = []
    for _id, text, emb, meta in zip(ids, texts, embeddings, metadatas):
        full_meta = {**meta, "text": text}
        vectors.append({"id": _id, "values": emb, "metadata": full_meta})

    for i in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[i : i + UPSERT_BATCH_SIZE]
        index.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)


def search(query_embedding: list[float], top_k: int = 5, metadata_filter: dict | None = None) -> list[dict]:
    """
    READ path — used only by the query pipeline. Takes an already-computed
    query embedding (see query_embedder.py) and returns the top_k most
    similar chunks with their text, metadata, and similarity score.

    metadata_filter (optional): a Pinecone filter dict, e.g.
        {"price_pkr": {"$lte": 50000}, "brand": {"$eq": "Samsung"}}
    lets the retriever narrow the search to phones matching structured
    criteria in addition to semantic similarity.
    """
    index = _get_client().Index(PINECONE_INDEX_NAME)
    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
        filter=metadata_filter,
    )

    hits = []
    for match in response.get("matches", []):
        meta = dict(match.get("metadata", {}))
        text = meta.pop("text", "")
        hits.append({"text": text, "metadata": meta, "score": match.get("score", 0.0)})
    return hits


def index_size() -> int:
    ensure_index_exists()
    index = _get_client().Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    return stats.get("namespaces", {}).get(PINECONE_NAMESPACE, {}).get("vector_count", 0)
