"""
config.py
---------
Central configuration for the Render backend (query + chat pipeline only).

The ingestion pipeline has its own config at:
    ingestion_pipeline/config_ingestion.py

IMPORTANT: EMBEDDING_MODEL_NAME is defined here so the backend knows
which model name the query_embedder.py should call via HuggingFace API.
It MUST stay in sync with the model used during ingestion locally.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# --- Chat history storage (used when MEMORY_BACKEND=file) ------------------
CHAT_HISTORY_DIR = BASE_DIR / "data" / "chat_history"
CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# --- Embedding model name ---------------------------------------------------
# Used by retrieval/query_embedder.py to call HuggingFace Inference API.
# MUST match what was used in ingestion_pipeline (all-MiniLM-L6-v2, 384 dim).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# --- Pinecone (vector database) --------------------------------------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "mobile-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "mobiles")

# --- Retrieval settings ----------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))

# --- LLM provider (pluggable) -----------------------------------------------
AVAILABLE_LLM_PROVIDERS = ("groq", "gemini", "mistral")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip().lower() or "groq"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip() or "llama-3.1-8b-instant"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash").strip() or "gemini-2.0-flash"

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip() or "mistral-small-latest"
