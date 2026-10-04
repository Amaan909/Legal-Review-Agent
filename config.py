"""Central configuration for the Legal Document Review Agent.

100% free, local, offline-capable stack:
  - LlamaIndex (OSS) for loading, chunking, and retrieval orchestration
  - ChromaDB (OSS, local) as the persistent vector store
  - HuggingFace BAAI/bge-small-en-v1.5 embeddings (free, local)
  - BM25Retriever (built-in) for keyword search
  - SentenceTransformerRerank cross-encoder (free, local) for re-ranking
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Step 1: local legal contract corpus -------------------------------------
CONTRACTS_DIR = BASE_DIR / "legal_dummy_contracts"
NUM_CONTRACTS = 15

# --- Step 2: chunking & embedding --------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models retrieve best when queries carry this instruction prefix.
EMBED_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"
CHUNK_SIZE = 512            # tokens per chunk
CHUNK_OVERLAP = 102         # ~20% overlap of 512 tokens

# --- Step 3: hybrid retrieval (vector + BM25) --------------------------------
CHROMA_DIR = str(BASE_DIR / "chroma_db")
CHROMA_COLLECTION = "legal_contracts"
NODES_PATH = BASE_DIR / "storage" / "nodes.pkl"   # chunked nodes reused by BM25
VECTOR_TOP_K = 10
BM25_TOP_K = 10
FUSION_TOP_K = 10
# Relative weight given to [vector, bm25] inside reciprocal-rank fusion.
FUSION_WEIGHTS = [0.6, 0.4]

# --- Step 4: re-ranking ------------------------------------------------------
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 4

# --- Optional: local LLM answer synthesis (Ollama) ---------------------------
# Off by default so the agent runs 100% free/offline with zero API keys.
# Enable with USE_OLLAMA=1 plus a running Ollama daemon (`ollama pull llama3.2`).
USE_OLLAMA = os.getenv("USE_OLLAMA", "0") == "1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
