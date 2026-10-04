"""The Legal Document Review Agent.

Ties Steps 3 + 4 together: given a natural-language legal question, it retrieves
candidate clauses via hybrid search (vector + BM25), re-ranks them with a local
cross-encoder, and returns the most relevant clauses with their source file and
page. Optional local LLM synthesis (Ollama) produces a written answer grounded
ONLY on the retrieved clauses; by default the agent is fully extractive and runs
offline with zero API keys.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import config
from hybrid_retriever import build_hybrid_retriever, build_reranker


@dataclass
class Clause:
    rank: int
    score: float
    text: str
    source: str
    page: str


@dataclass
class ReviewResult:
    query: str
    clauses: List[Clause]
    answer: Optional[str]


class LegalReviewAgent:
    """Hybrid-RAG legal clause finder with cross-encoder re-ranking."""

    def __init__(self) -> None:
        self.retriever = build_hybrid_retriever()
        self.reranker = build_reranker()

    def review(self, query: str, top_n: Optional[int] = None) -> ReviewResult:
        if not query or not query.strip():
            return ReviewResult(query=query, clauses=[], answer=None)

        candidates = self.retriever.retrieve(query)
        reranked = self.reranker.postprocess_nodes(candidates, query_str=query)
        if top_n is not None:
            reranked = reranked[:top_n]

        clauses: List[Clause] = []
        for i, ns in enumerate(reranked, start=1):
            meta = ns.node.metadata or {}
            clauses.append(
                Clause(
                    rank=i,
                    score=float(ns.score) if ns.score is not None else 0.0,
                    text=ns.node.get_content().strip(),
                    source=meta.get("file_name", meta.get("file_path", "unknown")),
                    page=str(meta.get("page_label", "-")),
                )
            )

        answer = self._synthesize(query, clauses)
        return ReviewResult(query=query, clauses=clauses, answer=answer)

    def _synthesize(self, query: str, clauses: List[Clause]) -> Optional[str]:
        """Optional grounded answer via local Ollama; None when disabled."""
        if not config.USE_OLLAMA or not clauses:
            return None
        try:
            from llama_index.llms.ollama import Ollama

            llm = Ollama(
                model=config.OLLAMA_MODEL,
                base_url=config.OLLAMA_BASE_URL,
                request_timeout=120.0,
            )
            context = "\n\n".join(f"[{c.source}, p.{c.page}]\n{c.text}" for c in clauses)
            prompt = (
                "You are a meticulous legal document review assistant. Using ONLY "
                "the contract excerpts provided, answer the question concisely and "
                "cite the source file name. If the excerpts do not contain the "
                "answer, say so.\n\n"
                f"Question: {query}\n\nContract excerpts:\n{context}\n\nAnswer:"
            )
            return str(llm.complete(prompt)).strip()
        except Exception as exc:  # pragma: no cover - optional path
            return f"(Local LLM synthesis unavailable: {exc})"


if __name__ == "__main__":
    agent = LegalReviewAgent()
    demo_queries = [
        "What does the indemnification clause cover?",
        "Is there a force majeure provision and what events does it include?",
        "How many days notice are required to terminate for convenience?",
        "What is the cap on each party's liability?",
        "Which law governs the agreement and where are disputes arbitrated?",
    ]
    for q in demo_queries:
        result = agent.review(q, top_n=3)
        print("\n" + "=" * 88)
        print("Q:", q)
        for c in result.clauses:
            snippet = " ".join(c.text[:220].split())
            print(f"  #{c.rank} [score={c.score:.3f}] {c.source} (p.{c.page})")
            print(f"      {snippet}...")
