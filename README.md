<<<<<<< HEAD
Good. That’s an important clarification.

Right now this is **a structured skeleton + notebook-built pipeline**, not a fully wired modular backend.

So the README must reflect reality:

- ✅ Core RAG works (via notebooks + FAISS + agents)
- ⚠️ `configs/`, `indexing/`, `ingestion/`, `preprocessing/` are structural placeholders
- ❌ No active production wiring between modules yet

Below is the **updated, accurate README**, written for an LLM or developer to understand the current state of the system — not an idealized version.

---

# 📘 OmniRAG – Agentic RAG System (Notebook-Driven Core)

## 🧠 System Status

OmniRAG is a modular Retrieval-Augmented Generation (RAG) project currently built and validated primarily through Jupyter notebooks.

The system includes:

- Semantic PDF ingestion
- Custom chunking aligned with BGE embeddings
- FAISS cosine similarity vector search
- Confidence-gated retrieval
- Agent-level orchestration logic
- Memory scaffolding
- Modular directory structure (partially implemented)

Important:

Some directories exist as architectural placeholders and currently do not contain active logic or wired runtime connections.

---

# 🏗 What Is Actually Functional

## ✅ Fully Working Components

The operational pipeline is defined in the notebooks:

1. `02_data_ingestion.ipynb`
2. `03_chunking.ipynb`
3. `04_embeddings_faiss_chroma.ipynb`
4. `05_retrieval_tests.ipynb`
5. `06_agentic_rag.ipynb`

These notebooks together implement:

- PDF ingestion
- Metadata normalization
- Semantic chunking
- Embedding with `BAAI/bge-base-en`
- FAISS indexing using cosine similarity
- Retrieval testing
- Agentic confidence gating
- Evidence-based answer control

The FAISS index and metadata stored in:

```
/data/processed/
    faiss.index
    faiss_chunks_meta.pkl
```

are the canonical runtime artifacts.

---

# 📂 Directory Structure (Current State)

## `/agents`

Contains:

- `runner.py`

Purpose:
Implements agent orchestration logic (retrieval → gating → answer flow).

Status:
Partially functional, not fully wired into a production application layer.

---

## `/app`

- `ui.py`

Purpose:
User interface entry layer.

Status:
Independent of core notebook-based RAG flow.

---

## `/configs`

Status:
Placeholder directory.

No active configuration logic currently connected to runtime.

---

## `/indexing`

Status:
Structural placeholder.

FAISS indexing is currently handled inside notebooks, not here.

---

## `/ingestion`

Status:
Structural placeholder.

Ingestion logic currently lives in `02_data_ingestion.ipynb`.

---

## `/preprocessing`

Status:
Structural placeholder.

Preprocessing handled directly inside notebooks.

---

## `/retrievers`

- `pdf_retriever.py`

Contains retrieval abstraction.

Status:
May contain partial logic, but canonical retrieval reference remains notebook 05/06 logic.

---

## `/vectorstore`

- `faiss_store.py`

Intended abstraction for FAISS.

Status:
Index building and loading currently handled in notebooks, not fully wired here.

---

## `/embeddings`

- `embedder.py`

Intended embedding abstraction.

Current canonical embedding implementation:

```
SentenceTransformer("BAAI/bge-base-en")
normalize_embeddings=True
dimension=768
```

Notebook implementation is authoritative.

---

## `/llm`

Contains:

- chat model abstraction
- embedding model wrappers
- LLM interface scaffolding

Status:
LLM usage exists conceptually.
Agent gating logic defined in notebook 06.
Full integration depends on runner wiring.

---

## `/memory`

Implements:

- Short-term memory
- Long-term memory
- Memory manager
- Summarization scaffolding

Status:
Memory structure exists.
Integration depth depends on agent runner usage.

---

## `/data`

### `/data/raw`

Source PDFs.

### `/data/processed`

Generated artifacts:

- `all_documents_v1.pkl`
- `chunks_v1.pkl`
- `faiss.index`
- `faiss_chunks_meta.pkl`

These are required for retrieval.

---

# 🧠 Core Design Decisions

## Embedding Model

```
BAAI/bge-base-en
Dimension: 768
Normalized embeddings
Cosine similarity via FAISS IndexFlatIP
```

If embedding model changes:
FAISS index must be rebuilt.

---

## Chunking Strategy

- Sentence-level splitting
- Semantic merging using embedding similarity
- Max ~350 words
- One coherent idea per chunk

Goal:
Each chunk must independently answer a question.

---

## Retrieval Policy

1. Retrieve top-K
2. Normalize query embedding
3. Cosine similarity search
4. Apply similarity threshold (default 0.7)
5. If no confident retrieval → refuse answer

---

## Hallucination Control

Agent logic enforces:

```
if max_similarity < threshold:
    refuse_to_answer
```

LLM must only answer using retrieved context.

---

# ⚠️ Important Clarification

This project currently has:

- Strong RAG logic
- Strong embedding + retrieval correctness
- Proper cosine math
- Confidence gating

But does NOT yet have:

- Fully unified runtime orchestration
- Config-driven execution
- Fully wired indexing/ingestion modules
- Production API layer
- End-to-end app integration

It is architecturally modular but partially notebook-driven.

---

# 🔄 Actual Execution Flow (Current Reality)

Notebook-driven flow:

```
PDF → Ingestion → Chunking → BGE Embeddings → FAISS Index
                                               ↓
                                           Retrieval
                                               ↓
                                         Confidence Gate
                                               ↓
                                              LLM
```

The notebooks are the source of truth.

---

# 🧭 Development Direction

To transition from notebook system → production system:

1. Move ingestion logic into `/ingestion`
2. Move chunking into `/preprocessing`
3. Move indexing into `/indexing`
4. Centralize embedding logic into `/embeddings`
5. Use `/vectorstore/faiss_store.py` as canonical FAISS interface
6. Wire everything through `/agents/runner.py`

---

# 🧠 System Identity (Current Phase)

OmniRAG is:

- Architecturally modular
- Notebook-validated
- Retrieval-correct
- Cosine-math consistent
- Agent-gated
- Hallucination-aware

It is currently in:

“Structured prototype transitioning toward production modularity.”

---

# 📌 Rules for Any LLM Interacting With This Codebase

1. Do not change embedding model without rebuilding FAISS.
2. Always normalize embeddings.
3. Use IndexFlatIP for cosine similarity.
4. Do not allow LLM answers without retrieval confidence.
5. Treat notebooks as canonical implementation reference.
6. Do not assume configs/indexing/ingestion modules are active.

---

If you want next, we can:

- Design the production wiring plan
- Refactor notebooks into modular runtime code
- Create a migration roadmap
- Add API layer documentation
- Design a clean CLI entrypoint

# You’ve built the hard part already. Now it’s about solidifying architecture.

# OMNI_RAG

> > > > > > > 8c7791f17371cca288bbbd70e9cf680a4aa2283b
