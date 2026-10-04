"""Steps 3 & 4 - Hybrid retrieval (vector + BM25 fusion) and cross-encoder re-ranking.

Step 3 (Hybrid RAG): a ChromaDB-backed ``VectorStoreIndex`` retriever (semantic
    search) is fused with a built-in ``BM25Retriever`` (keyword search, which
    catches exact legal terms such as 'indemnity' or 'force majeure') using
    ``QueryFusionRetriever`` with reciprocal-rank fusion.

Step 4 (Re-ranking): ``SentenceTransformerRerank`` (cross-encoder/ms-marco-
    MiniLM-L-6-v2) re-scores the fused candidates so the single most relevant
    clause rises to the top.

Everything runs locally and free. No paid LLM is required for retrieval: query
expansion is disabled (``num_queries=1``) and a ``MockLLM`` is registered so
LlamaIndex never tries to resolve an OpenAI default.
"""
from __future__ import annotations

import pickle

import chromadb
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

import config
from ingest import get_embed_model


def configure_settings() -> None:
    """Register the local embedding model and an offline-safe LLM default."""
    Settings.embed_model = get_embed_model()
    if config.USE_OLLAMA:
        from llama_index.llms.ollama import Ollama

        Settings.llm = Ollama(
            model=config.OLLAMA_MODEL,
            base_url=config.OLLAMA_BASE_URL,
            request_timeout=120.0,
        )
    else:
        # No paid LLM. MockLLM keeps LlamaIndex from resolving an OpenAI default.
        Settings.llm = MockLLM()


def load_vector_retriever():
    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=get_embed_model()
    )
    return index.as_retriever(similarity_top_k=config.VECTOR_TOP_K)


def load_nodes():
    with open(config.NODES_PATH, "rb") as fh:
        return pickle.load(fh)


def build_hybrid_retriever() -> QueryFusionRetriever:
    """Fuse semantic (vector) and keyword (BM25) retrievers via reciprocal rank."""
    configure_settings()
    vector_retriever = load_vector_retriever()
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=load_nodes(), similarity_top_k=config.BM25_TOP_K
    )
    return QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        retriever_weights=config.FUSION_WEIGHTS,
        similarity_top_k=config.FUSION_TOP_K,
        num_queries=1,                 # disable LLM query expansion -> fully offline
        mode="reciprocal_rerank",      # robust across cosine + BM25 score scales
        use_async=False,
        verbose=False,
    )


def build_reranker() -> SentenceTransformerRerank:
    """Local cross-encoder re-ranker (free, no API key)."""
    return SentenceTransformerRerank(
        model=config.RERANK_MODEL, top_n=config.RERANK_TOP_N
    )
