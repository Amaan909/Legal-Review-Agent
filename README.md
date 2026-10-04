# Legal Document Review Agent - Hybrid RAG (100% Free Stack)

A Legal Document Review Agent built with **LlamaIndex + ChromaDB** - no Azure
subscription, no API keys, runs locally and free. It finds the most relevant
clause across a corpus of legal contracts using **hybrid retrieval** (semantic +
keyword) followed by **cross-encoder re-ranking**, and includes a **Build vs Buy
ROI** calculator.

## Pipeline

| Step | What it does | Tech (free / local) |
| --- | --- | --- |
| 1. Data prep | Local legal contract PDFs (replaces Azure Blob) | `generate_contracts.py` (fpdf2) |
| 2. Chunk + embed | Load PDFs, split into 512-token chunks (20% overlap), embed | LlamaIndex `SimpleDirectoryReader` + `SentenceSplitter` + HuggingFace `BAAI/bge-small-en-v1.5` |
| 3. Hybrid RAG | Semantic (ChromaDB vectors) fused with keyword (BM25) | `VectorStoreIndex` + `BM25Retriever` + `QueryFusionRetriever` |
| 4. Re-ranking | Re-score candidates, surface the best clause | `SentenceTransformerRerank` (`cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| 5. ROI analysis | Build vs Buy cost model + Excel export | `roi.py` + Streamlit |

## Quick start

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Step 1 - generate the local contract corpus
python generate_contracts.py

# 4. Step 2 - chunk, embed, and index into ChromaDB
python ingest.py

# 5a. Run a quick terminal test (Steps 3 + 4)
python agent.py

# 5b. Or launch the Streamlit UI (all steps + ROI)
streamlit run app.py
```

> First run downloads two small models from HuggingFace (the bge-small embedder
> and the MiniLM re-ranker) and caches them locally. After that the agent runs
> fully offline.

## Testing

Run the automated end-to-end check (covers all 5 steps + the UI helpers):

```powershell
python test_pipeline.py
```

It asserts the corpus, index, hybrid retrieval, re-ranking, ROI math, and Excel
export, then prints a `PASS / FAIL` summary and exits non-zero on any failure.

To test the UI manually, launch `streamlit run app.py` and try these prompts in
the **Document Review** tab:

| Prompt | Expected top clause |
| --- | --- |
| `What is the cap on each party's liability?` | LIMITATION OF LIABILITY (aggregate liability / USD cap) |
| `Which law governs the agreement and where are disputes arbitrated?` | GOVERNING LAW AND JURISDICTION |
| `Is there a force majeure provision and what events does it include?` | FORCE MAJEURE |
| `What does the indemnification clause cover?` | INDEMNIFICATION |
| `How many days notice are required to terminate for convenience?` | TERM AND TERMINATION |
| `What are the confidentiality obligations and how long do they survive?` | CONFIDENTIALITY |

Each result shows the source PDF, page number, and a relevance score. In the
**ROI Analysis** tab, adjust the inputs and click **Download ROI template
(.xlsx)** to export the model.

## Files

| File | Purpose |
| --- | --- |
| `config.py` | Central settings (models, chunk size, paths, top-k) |
| `generate_contracts.py` | Step 1 - generate local contract PDFs |
| `ingest.py` | Step 2 - chunk, embed, persist to ChromaDB + BM25 nodes |
| `hybrid_retriever.py` | Steps 3-4 - fusion retriever + cross-encoder re-ranker |
| `agent.py` | Legal review agent (retrieve -> re-rank -> clauses) |
| `roi.py` | Step 5 - ROI model + Excel export |
| `app.py` | Streamlit UI (review, corpus, ROI) |
| `test_pipeline.py` | End-to-end smoke test for all steps + UI helpers |

## Optional: local LLM answers

By default the agent is **extractive** (returns the top re-ranked clauses). To
add a written, grounded answer using a local model, install and run
[Ollama](https://ollama.com), then:

```powershell
ollama pull llama3.2
$env:USE_OLLAMA = "1"
streamlit run app.py
```

## Step up (production)

Swap ChromaDB to **Azure AI Search** and the local embeddings to **Azure OpenAI
`text-embedding-3-large`** for enterprise scale, security trimming, and SLA
guarantees - the Build vs Buy decision this use case explores. The retrieval,
re-ranking, and agent interfaces stay the same.
