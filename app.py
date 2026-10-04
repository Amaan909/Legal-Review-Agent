"""Streamlit UI for the Legal Document Review Agent (Hybrid RAG, 100% free stack).

LlamaIndex + ChromaDB + HuggingFace bge-small embeddings + BM25 + cross-encoder
re-ranking, with a Build-vs-Buy ROI calculator. No Azure, no API keys.

Run:
    streamlit run app.py
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import streamlit as st

import config
from ingest import index_exists

st.set_page_config(
    page_title="Legal Document Review Agent",
    page_icon=":scales:",
    layout="wide",
)

EXAMPLE_QUERIES = [
    "What does the indemnification clause cover?",
    "Is there a force majeure provision and what events does it include?",
    "How many days notice are required to terminate for convenience?",
    "What is the cap on each party's liability?",
    "Which law governs the agreement and where are disputes arbitrated?",
    "What are the confidentiality obligations and how long do they survive?",
]


# --------------------------- cached resources ---------------------------
@st.cache_resource(show_spinner="Loading models + building hybrid retriever (first run only)...")
def get_agent():
    """Build the agent once per session (loads embedder + reranker + retrievers)."""
    from agent import LegalReviewAgent

    return LegalReviewAgent()


# --------------------------- pipeline actions ---------------------------
def run_data_prep() -> int:
    from generate_contracts import generate

    count = generate()
    get_agent.clear()
    return count


def run_ingest() -> tuple[int, int]:
    from ingest import ingest

    result = ingest(rebuild=True)
    get_agent.clear()
    return result


def list_corpus() -> pd.DataFrame:
    rows = []
    if config.CONTRACTS_DIR.exists():
        for pdf in sorted(config.CONTRACTS_DIR.glob("*.pdf")):
            rows.append({"File": pdf.name, "Size (KB)": round(pdf.stat().st_size / 1024, 1)})
    return pd.DataFrame(rows)


# --------------------------- sidebar ---------------------------
def render_sidebar() -> int:
    st.sidebar.title("Legal Review Agent")
    st.sidebar.caption("Hybrid RAG - 100% free, local stack")

    st.sidebar.markdown(
        "**Stack**\n"
        "- LlamaIndex (OSS)\n"
        "- ChromaDB (local vectors)\n"
        "- bge-small-en-v1.5 (embeddings)\n"
        "- BM25Retriever (keyword)\n"
        "- ms-marco MiniLM (re-ranker)"
    )

    st.sidebar.divider()
    st.sidebar.subheader("Pipeline status")

    n_pdfs = len(list(config.CONTRACTS_DIR.glob("*.pdf"))) if config.CONTRACTS_DIR.exists() else 0
    built = index_exists()
    st.sidebar.write(f"Step 1 - Contracts: {'OK' if n_pdfs else 'missing'} ({n_pdfs} PDFs)")
    st.sidebar.write(f"Step 2 - Index: {'OK' if built else 'not built'}")
    st.sidebar.write(f"LLM synthesis: {'Ollama' if config.USE_OLLAMA else 'off (extractive)'}")

    st.sidebar.divider()
    if st.sidebar.button("Step 1: Generate contracts", use_container_width=True):
        with st.spinner("Generating contract PDFs..."):
            count = run_data_prep()
        st.sidebar.success(f"Generated {count} contracts.")

    if st.sidebar.button("Step 2: Build / rebuild index", use_container_width=True, disabled=not n_pdfs):
        with st.spinner("Chunking, embedding, and indexing into ChromaDB..."):
            docs, chunks = run_ingest()
        st.sidebar.success(f"Indexed {docs} docs -> {chunks} chunks.")

    st.sidebar.divider()
    top_n = st.sidebar.slider("Clauses to show (re-ranked)", 1, config.RERANK_TOP_N, 3)
    return top_n


# --------------------------- tabs ---------------------------
def tab_review(top_n: int) -> None:
    st.header("Document Review")
    st.caption(
        "Ask a question in plain English. The agent runs hybrid retrieval "
        "(semantic + keyword), then a cross-encoder re-ranks to surface the most "
        "relevant clause."
    )

    if not index_exists():
        st.warning(
            "Index not built yet. Use the sidebar: **Step 1** to generate contracts, "
            "then **Step 2** to build the index."
        )
        return

    st.write("**Example questions**")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLE_QUERIES):
        if cols[i % 3].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["query"] = ex

    query = st.text_input(
        "Your question",
        key="query",
        placeholder="e.g. What is the limitation of liability cap?",
    )

    if st.button("Review", type="primary") or query:
        if not query.strip():
            st.info("Type a question or pick an example above.")
            return
        try:
            agent = get_agent()
        except Exception as exc:  # dependency / index errors
            st.error(f"Could not load the agent: {exc}")
            return

        with st.spinner("Retrieving and re-ranking clauses..."):
            result = agent.review(query, top_n=top_n)

        if result.answer:
            st.subheader("Answer")
            st.write(result.answer)

        if not result.clauses:
            st.info("No relevant clauses found.")
            return

        st.subheader("Most relevant clauses")
        for c in result.clauses:
            with st.container(border=True):
                head = f"#{c.rank} - {c.source} (page {c.page})  -  relevance {c.score:.3f}"
                st.markdown(f"**{head}**")
                st.write(c.text)


def tab_corpus() -> None:
    st.header("Contract Corpus")
    df = list_corpus()
    if df.empty:
        st.warning("No contracts yet. Use the sidebar **Step 1** to generate them.")
        return
    st.caption(f"{len(df)} local PDF contracts in {config.CONTRACTS_DIR.name}/")
    st.dataframe(df, use_container_width=True, hide_index=True)


def tab_roi() -> None:
    import roi

    st.header("ROI Analysis - Build vs Buy")
    st.caption(
        "Baseline 2 hrs manual review -> ~20 min with the agent. Compare this "
        "custom hybrid-RAG build against a third-party legal AI subscription."
    )

    defaults = roi.ROIInputs()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Review effort**")
        manual = st.number_input("Manual review (min/doc)", 1.0, 600.0, defaults.manual_minutes_per_doc, 5.0)
        agent_m = st.number_input("Agent review (min/doc)", 1.0, 600.0, defaults.agent_minutes_per_doc, 5.0)
        rate = st.number_input("Reviewer rate (USD/hr)", 1.0, 1000.0, defaults.reviewer_hourly_rate, 5.0)
        volume = st.number_input("Documents / month", 1, 100000, defaults.documents_per_month, 10)
    with c2:
        st.markdown("**Build (this agent)**")
        build_cost = st.number_input("Build cost (one-time, USD)", 0.0, 1_000_000.0, defaults.build_cost_one_time, 500.0)
        infra = st.number_input("Infra cost (USD/month)", 0.0, 100000.0, defaults.infra_cost_monthly, 10.0)
    with c3:
        st.markdown("**Buy (legal AI SaaS)**")
        saas_platform = st.number_input("SaaS platform (USD/month)", 0.0, 100000.0, defaults.saas_platform_monthly, 50.0)
        saas_per_doc = st.number_input("SaaS cost (USD/doc)", 0.0, 1000.0, defaults.saas_cost_per_doc, 1.0)

    inp = roi.ROIInputs(
        manual_minutes_per_doc=manual,
        agent_minutes_per_doc=agent_m,
        reviewer_hourly_rate=rate,
        documents_per_month=int(volume),
        build_cost_one_time=build_cost,
        infra_cost_monthly=infra,
        saas_platform_monthly=saas_platform,
        saas_cost_per_doc=saas_per_doc,
    )
    res = roi.compute_roi(inp)

    st.divider()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cost saved / doc", f"${res['cost_saved_per_doc']:,.0f}")
    m2.metric("Annual labor savings", f"${res['annual_labor_savings']:,.0f}")
    m3.metric("Year-1 ROI", f"{res['roi_pct_year1']:,.0f}%")
    payback = res["payback_period_months"]
    m4.metric("Payback", "n/a" if payback == float("inf") else f"{payback:,.1f} mo")

    b1, b2, b3 = st.columns(3)
    b1.metric("Build - Year 1", f"${res['build_year1_cost']:,.0f}")
    b2.metric("Buy (SaaS) - Year 1", f"${res['buy_year1_cost']:,.0f}")
    b3.metric("Build vs Buy saving", f"${res['build_vs_buy_annual_saving']:,.0f}")

    table = pd.DataFrame(
        {"Metric": list(roi._RESULT_LABELS.values()),
         "Value": [round(res[k], 2) for k in roi._RESULT_LABELS]}
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    buffer = io.BytesIO()
    roi.export_excel(inp, res, buffer)
    st.download_button(
        "Download ROI template (.xlsx)",
        data=buffer.getvalue(),
        file_name="legal_review_roi.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def main() -> None:
    top_n = render_sidebar()
    st.title("Legal Document Review Agent")
    st.caption("Hybrid RAG (semantic + keyword) with cross-encoder re-ranking - LlamaIndex + ChromaDB")

    review, corpus, roi_tab = st.tabs(["Document Review", "Corpus", "ROI Analysis"])
    with review:
        tab_review(top_n)
    with corpus:
        tab_corpus()
    with roi_tab:
        tab_roi()


if __name__ == "__main__":
    main()
