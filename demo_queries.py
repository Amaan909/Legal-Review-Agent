"""Quick demo: ask the agent several different questions and show the answers.

Run:  .\.venv\Scripts\python.exe demo_queries.py
"""
from __future__ import annotations

import re

from agent import LegalReviewAgent


def focus(text: str, query: str, max_chars: int = 360) -> str:
    """Return the part of a clause chunk most relevant to the query."""
    sentences = re.split(r"(?<=[.;])\s+", text)
    q_terms = {w for w in re.findall(r"[a-zA-Z]{4,}", query.lower())}
    best_i, best_score = 0, -1
    for i, s in enumerate(sentences):
        s_terms = set(re.findall(r"[a-zA-Z]{4,}", s.lower()))
        score = len(q_terms & s_terms)
        if score > best_score:
            best_score, best_i = score, i
    snippet = sentences[best_i]
    j = best_i + 1
    while j < len(sentences) and len(snippet) < max_chars:
        snippet += " " + sentences[j]
        j += 1
    snippet = " ".join(snippet.split())
    return snippet[:max_chars] + ("..." if len(snippet) > max_chars else "")


QUERIES = [
    "What is the maximum liability cap and what are the exceptions?",
    "Which state or country law governs the contract?",
    "What happens if a force majeure event lasts more than 60 days?",
    "How much notice is needed to terminate the agreement for convenience?",
    "Who must indemnify whom for third-party intellectual property claims?",
    "How long do confidentiality obligations survive after termination?",
    "Is arbitration required to resolve disputes and under which rules?",
]


def main() -> None:
    print("Loading agent (embedder + re-ranker)...\n")
    agent = LegalReviewAgent()

    for q in QUERIES:
        result = agent.review(q, top_n=2)
        print("=" * 90)
        print("QUESTION:", q)
        if not result.clauses:
            print("  (no clauses found)")
            continue
        for c in result.clauses:
            print(f"\n  Rank #{c.rank}  |  {c.source}  |  page {c.page}  |  relevance {c.score:.3f}")
            print("  ANSWER:", focus(c.text, q))
        print()


if __name__ == "__main__":
    main()
