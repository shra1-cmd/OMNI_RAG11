# OmniRAG v2

A high-performance Retrieval-Augmented Generation (RAG) web app built with Streamlit, Pinecone, MongoDB, and the ultra-fast Groq LPU.

_[ Add UI screenshots / Architecture diagrams here ]_

## � Features

- **Hybrid Search:** Combines HuggingFace semantic vectors (`BAAI/bge-base-en-v1.5`) with BM25 keyword search.
- **Cross-Encoder Reranking:** Precision filtering for the most relevant context chunks.
- **HyDE Fallback:** Automatically generates hypothetical documents to expand search when initial confidence is low.
- **Multi-format Ingestion:** Seamlessly chunks and indexes PDFs, CSVs, and Markdowns.

## � Core Structure

- `src/app.py`: Streamlit entry point.
- `src/core/`: The `RAGPipeline` orchestration and query rewriting.
- `src/retrieval/`: LlamaIndex hybrid retrievers and Cross-Encoder reranking.
- `src/storage/`: Pinecone (Vectors) and MongoDB (Sessions/Metadata) wrappers.
- `eval/`: BEIR dataset evaluation scripts.

## 🚀 Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   Set up your `.env` file with `GROQ_API_KEY`, `PINECONE_API_KEY`, and `MONGO_URI`.

3. **Run the App**
   ```bash
   streamlit run src/app.py
   ```
