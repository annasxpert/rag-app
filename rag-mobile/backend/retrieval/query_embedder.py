"""
query_embedder.py
-----------------
Embed user queries at request time.

Supports TWO modes automatically:
1. Local SentenceTransformer (RECOMMENDED for EC2 / VPS / Local):
   - Fast (~10-20ms per query)
   - Zero external API rate limits or network dependencies
   - Uses the cached 'sentence-transformers/all-MiniLM-L6-v2' model
2. HuggingFace Inference API (Fallback for ultra-low RAM environments):
   - Used only if sentence-transformers is not installed.
"""

from __future__ import annotations

from functools import lru_cache
import os
import time
from typing import Any

import requests

from config import EMBEDDING_MODEL_NAME

HF_API_URL = (
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/"
    f"{EMBEDDING_MODEL_NAME}"
)
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "").strip()

_MAX_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds


@lru_cache(maxsize=1)
def _get_local_model() -> Any:
    """Lazily load the local SentenceTransformer model if available."""
    try:
        from sentence_transformers import SentenceTransformer
        print(f"[query_embedder] Loading local SentenceTransformer model '{EMBEDDING_MODEL_NAME}'...")
        return SentenceTransformer(EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"[query_embedder] Local sentence-transformers not available ({e}). Using HF API fallback.")
        return None


def _hf_embed(text: str) -> list[float]:
    """
    Call HuggingFace Inference API to get a 384-dim embedding.
    Retries on model loading (503) with exponential backoff.
    """
    headers = {"Content-Type": "application/json"}
    if HF_API_KEY:
        headers["Authorization"] = f"Bearer {HF_API_KEY}"

    payload = {
        "inputs": text,
        "options": {"wait_for_model": True, "use_cache": True},
    }

    for attempt in range(_MAX_RETRIES):
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)

            if response.status_code == 503:
                wait = _RETRY_DELAY * (attempt + 1)
                time.sleep(wait)
                continue

            response.raise_for_status()
            result = response.json()

            if isinstance(result, list):
                if isinstance(result[0], list):
                    return result[0]
                return result

        except requests.RequestException as e:
            if attempt == _MAX_RETRIES - 1:
                raise RuntimeError(
                    f"HuggingFace Inference API failed after {_MAX_RETRIES} attempts: {e}"
                ) from e
            time.sleep(_RETRY_DELAY)

    raise RuntimeError("HuggingFace Inference API: max retries exceeded.")


def embed_query(query: str) -> list[float]:
    """
    Compute a fresh embedding for one user question at query time.
    Uses local SentenceTransformer if available; falls back to HuggingFace Inference API.
    """
    cleaned = query.strip()
    if not cleaned:
        cleaned = "phone"

    local_model = _get_local_model()
    if local_model is not None:
        vector = local_model.encode(cleaned, normalize_embeddings=True)
        return vector.tolist()

    return _hf_embed(cleaned)

