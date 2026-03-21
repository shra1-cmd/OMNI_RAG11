import asyncio
from core.config import Settings
from retrieval.reranker import Reranker
from llm.groq_client import GroqClient
from core.rewriter import QueryRewriter
# 1. New LlamaIndex Imports
from retrieval.llama_retriever import LlamaHybridRetriever
from core.load_llama_docs import load_docs

class RAGPipeline:
    def __init__(self, vector_store=None, keyword_store=None):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        
        # 3. Utilities (Embedder removed as Llama handles it)
        self.reranker = Reranker(Settings.RERANK_MODEL)
        self.llm = GroqClient()
        self.rewriter = QueryRewriter()

    def dynamic_k(self, query: str):
        words = len(query.split())
        if words < 5:
            return Settings.TOP_K_MIN
        elif words < 15:
            return Settings.TOP_K_DEFAULT
        return Settings.TOP_K_MAX

    def generate_hyde(self, query: str):
        prompt = f"Write a detailed scientific passage answering the question: {query}"
        return self.llm.generate(prompt)

    async def run(self, query: str, session_id=None):
        steps = []
        # 1️⃣ Query Rewriting
        original_query = query
        steps.append({"name": "Question Reformulation", "detail": "Rewriting query for semantic clarity..."})
        query = self.rewriter.rewrite(query)
        steps[-1]["detail"] = f"Rewritten Query: **{query}**"
        print(f"🔍 Processing Query: {query}")

        # 2️⃣ Initial Retrieval (Combine Global + Session Docs)
        docs_list = load_docs()
#         docs_list = [
#     {"text": doc["text"], "id": doc_id}
#     for doc_id, doc in corpus.items()
# ]
        if session_id and self.keyword_store:
            session_docs = self.keyword_store.collection.find({"session_id": session_id})
            docs_list.extend([{"text": d["text"]} for d in session_docs])

        steps.append({"name": "Hybrid Retrieval", "detail": f"Searching across {len(docs_list)} text fragments (Vector + BM25)..."})
        
        # Re-initialize retriever with combined docs
        retriever = LlamaHybridRetriever(docs_list)
        
        # Using the string query directly per LlamaIndex requirements
        docs = await asyncio.to_thread(retriever.retrieve, query)

        # 3️⃣ Check Confidence & Conditional HyDE
        max_score = max([d["score"] for d in docs]) if docs else 0
        print(f"📊 Initial Top Score: {max_score:.4f}")

        if Settings.USE_HYDE and max_score < Settings.HYDE_THRESHOLD:
            steps.append({"name": "HyDE (Semantic Enhancement)", "detail": f"Initial score {max_score:.4f} below threshold {Settings.HYDE_THRESHOLD}. Generating hypothetical scientific answer..."})
            print(f"⚡ Score below {Settings.HYDE_THRESHOLD} - Triggering HYDE fallback...")
            
            # Generate the hallucinated "fake" answer
            hyde_text = self.generate_hyde(query)
            
            # Re-retrieve using the hyde_text string
            docs = await asyncio.to_thread(retriever.retrieve, hyde_text)
            max_score = max([d["score"] for d in docs]) if docs else max_score
            steps[-1]["detail"] += " | Re-retrieval complete."

        # 4️⃣ Rerank
        if docs:
            steps.append({"name": "Cross-Encoder Reranking", "detail": f"Precision scoring {len(docs)} candidates for final relevance..."})
            docs = await asyncio.to_thread(
            self.reranker.rerank, original_query, docs
              )
            docs = docs[:Settings.RERANK_TOP_K]

        # 5️⃣ Build Context & Prompt
        context = "\n\n".join([d["text"] for d in docs]) if docs else "No relevant context found."
        
        prompt = f"""
You are an expert AI assistant for OmniRAG.

Use ONLY the context below to answer. 
If the answer is not in the context, strictly say: "I don't have enough information in my database."

Context:
{context}

Question:
{original_query}

Answer clearly, professionally, and concisely:
"""

        # 6️⃣ Final Generation
        steps.append({"name": "LLM Synthesis", "detail": "Synthesizing final answer with context chunks..."})
        answer = self.llm.generate(prompt)

        return {
            "answer": answer,
            "sources": docs,
            "confidence": round(max_score, 4),
            "used_hyde": max_score < Settings.HYDE_THRESHOLD,
            "steps": steps
        }
