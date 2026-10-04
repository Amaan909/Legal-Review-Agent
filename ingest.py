"""Step 2 - Chunking & embedding, persisted to ChromaDB.

Loads the local contract PDFs with LlamaIndex ``SimpleDirectoryReader``, splits
them into 512-token chunks (20% overlap) with ``SentenceSplitter``, embeds each
chunk with the free local HuggingFace ``BAAI/bge-small-en-v1.5`` model, and stores
the vectors in a persistent ChromaDB collection. The chunked nodes are also
pickled so the BM25 keyword retriever (Step 3) reuses the exact same chunks with
matching node IDs.

Run:
    python ingest.py
"""
from __future__ import annotations

import pickle
import shutil
from pathlib import Path

import chromadb
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

_EMBED_MODEL: HuggingFaceEmbedding | None = None


def get_embed_model() -> HuggingFaceEmbedding:
    """Lazily build (and cache) the local bge-small embedding model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        _EMBED_MODEL = HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL,
            query_instruction=config.EMBED_QUERY_INSTRUCTION,
        )
    return _EMBED_MODEL


def load_documents():
    if not config.CONTRACTS_DIR.exists() or not any(config.CONTRACTS_DIR.glob("*.pdf")):
        raise FileNotFoundError(
            f"No PDF contracts found in {config.CONTRACTS_DIR}. "
            "Run `python generate_contracts.py` first."
        )
    reader = SimpleDirectoryReader(
        input_dir=str(config.CONTRACTS_DIR),
        required_exts=[".pdf"],
    )
    # return LlamaIndex Document objects
    return reader.load_data()


def chunk_documents(documents):
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    # returns a list of Node objects.
    return splitter.get_nodes_from_documents(documents)


def ingest(rebuild: bool = True) -> tuple[int, int]:
    """Build the ChromaDB index and the BM25 node store. Returns (docs, chunks)."""
    documents = load_documents()
    nodes = chunk_documents(documents)
    embed_model = get_embed_model()

    if rebuild and Path(config.CHROMA_DIR).exists():
        shutil.rmtree(config.CHROMA_DIR)

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    collection = client.get_or_create_collection(
        config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Embedding happens here; vectors are written into ChromaDB.
    VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )

    # Persist the exact chunks so BM25 (Step 3) uses identical nodes + IDs.
    config.NODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.NODES_PATH, "wb") as fh:
        pickle.dump(nodes, fh)

    return len(documents), len(nodes)


def index_exists() -> bool:
    return Path(config.CHROMA_DIR).exists() and config.NODES_PATH.exists()


if __name__ == "__main__":
    n_docs, n_nodes = ingest(rebuild=True)
    print(f"Ingested {n_docs} documents into {n_nodes} chunks.")
    print(f"Vectors  -> {config.CHROMA_DIR} (collection '{config.CHROMA_COLLECTION}')")
    print(f"BM25 nodes -> {config.NODES_PATH}")
