"""End-to-end smoke test for the Legal Document Review Agent.

Exercises every requirement step and every function the Streamlit UI calls,
with hard assertions and a clean PASS/FAIL summary. Run:

    .\.venv\Scripts\python.exe test_pipeline.py
"""
from __future__ import annotations

import io
import sys
import traceback

import config

PASS, FAIL = "PASS", "FAIL"
results: list[tuple[str, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((PASS if cond else FAIL, name, detail))
    mark = "[ OK ]" if cond else "[FAIL]"
    print(f"{mark} {name}" + (f" - {detail}" if detail else ""))


def section(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# --------------------------------------------------------------------------
section("Requirement 1 - Local legal contract corpus")
try:
    pdfs = sorted(config.CONTRACTS_DIR.glob("*.pdf"))
    check("Contracts directory exists", config.CONTRACTS_DIR.exists(), str(config.CONTRACTS_DIR))
    check("At least 15 contract PDFs present", len(pdfs) >= 15, f"{len(pdfs)} PDFs")
    check("PDFs are non-empty", all(p.stat().st_size > 1000 for p in pdfs))
except Exception:
    check("Corpus check", False, traceback.format_exc().splitlines()[-1])


# --------------------------------------------------------------------------
section("Requirement 2 - Chunk + embed (bge-small) into ChromaDB")
try:
    from ingest import index_exists

    check("Chroma index exists", index_exists(), str(config.CHROMA_DIR))
    check("BM25 nodes pickle exists", config.NODES_PATH.exists(), str(config.NODES_PATH))
    check("Chunk size = 512 (per spec)", config.CHUNK_SIZE == 512, f"{config.CHUNK_SIZE}")
    check("Chunk overlap ~20%", abs(config.CHUNK_OVERLAP - 102) <= 5, f"{config.CHUNK_OVERLAP}")
    check("Embed model is bge-small", "bge-small" in config.EMBED_MODEL, config.EMBED_MODEL)
except Exception:
    check("Ingest config check", False, traceback.format_exc().splitlines()[-1])


# --------------------------------------------------------------------------
section("Requirements 3 + 4 - Hybrid retrieval + cross-encoder re-rank")
agent = None
try:
    from agent import LegalReviewAgent, ReviewResult, Clause

    agent = LegalReviewAgent()
    check("Agent constructed (retriever + reranker built)", agent is not None)
    check("Reranker model is ms-marco cross-encoder",
          "ms-marco" in config.RERANK_MODEL, config.RERANK_MODEL)

    q = "Which law governs the agreement and where are disputes arbitrated?"
    res = agent.review(q, top_n=3)
    check("review() returns ReviewResult", isinstance(res, ReviewResult))
    check("Clauses returned", len(res.clauses) > 0, f"{len(res.clauses)} clauses")
    check("Clauses respect top_n", len(res.clauses) <= 3, f"{len(res.clauses)}")
    top = res.clauses[0]
    check("Clause has source file", top.source.endswith(".pdf"), top.source)
    check("Clause has page label", bool(top.page), f"page {top.page}")
    check("Clause text non-empty", len(top.text) > 0, f"{len(top.text)} chars")
    check("Re-rank is relevant (governing law clause on top)",
          "GOVERNING LAW" in top.text.upper() or "govern" in top.text.lower(),
          top.text[:70].replace("\n", " "))

    # Empty query is handled gracefully (UI relies on this)
    empty = agent.review("   ", top_n=3)
    check("Empty query returns no clauses", empty.clauses == [])

    # Ranks are monotonic 1..n
    ranks = [c.rank for c in res.clauses]
    check("Ranks are 1..n in order", ranks == list(range(1, len(ranks) + 1)), str(ranks))
except Exception:
    check("Agent retrieval", False, traceback.format_exc().splitlines()[-1])


# --------------------------------------------------------------------------
section("Requirement 5 - ROI analysis + Excel export")
try:
    import roi

    inp = roi.ROIInputs()
    r = roi.compute_roi(inp)
    check("100 minutes saved per doc (120 -> 20)", r["minutes_saved_per_doc"] == 100.0,
          f"{r['minutes_saved_per_doc']}")
    check("Annual labor savings positive", r["annual_labor_savings"] > 0,
          f"${r['annual_labor_savings']:,.0f}")
    check("Year-1 ROI computed", "roi_pct_year1" in r, f"{r['roi_pct_year1']:.0f}%")
    check("Build-vs-Buy delta computed", "build_vs_buy_annual_saving" in r,
          f"${r['build_vs_buy_annual_saving']:,.0f}")

    # Export to an in-memory buffer (exactly what the Streamlit download button does)
    buf = io.BytesIO()
    roi.export_excel(inp, r, buf)
    data = buf.getvalue()
    check("Excel export produced bytes", len(data) > 0, f"{len(data)} bytes")
    check("Excel has XLSX (zip) signature", data[:2] == b"PK")
except Exception:
    check("ROI analysis", False, traceback.format_exc().splitlines()[-1])


# --------------------------------------------------------------------------
section("Streamlit UI helper functions")
try:
    import app

    df = app.list_corpus()
    check("app.list_corpus() returns rows", not df.empty, f"{len(df)} rows")
    check("app.get_agent is cached resource", hasattr(app.get_agent, "clear"))
    check("EXAMPLE_QUERIES present", len(app.EXAMPLE_QUERIES) >= 5, f"{len(app.EXAMPLE_QUERIES)}")
    # The agent the UI caches should be reusable for a query
    if agent is not None:
        res2 = agent.review("What is the cap on each party's liability?", top_n=2)
        check("UI-style second query works", len(res2.clauses) > 0, f"{len(res2.clauses)} clauses")
except Exception:
    check("UI helpers", False, traceback.format_exc().splitlines()[-1])


# --------------------------------------------------------------------------
section("SUMMARY")
passed = sum(1 for s, _, _ in results if s == PASS)
failed = sum(1 for s, _, _ in results if s == FAIL)
print(f"\n{passed} passed, {failed} failed, {len(results)} total")
if failed:
    print("\nFailures:")
    for s, name, detail in results:
        if s == FAIL:
            print(f"  - {name}: {detail}")
sys.exit(1 if failed else 0)
