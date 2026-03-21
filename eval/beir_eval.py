import os
import sys
import asyncio
import time
import numpy as np
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval

# 1️⃣ Path Setup
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from retrieval.llama_retriever import LlamaHybridRetriever

current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "datasets", "scifact")

# 2️⃣ Load FULL Dataset
print(f"🔍 Loading FULL SciFact dataset from: {data_path}")
corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

# No slicing - use everything
docs_list = [{"id": tid, "text": d["text"]} for tid, d in corpus.items()]
test_queries = list(queries.items())

print(f"🔥 TOTALS: {len(test_queries)} queries | {len(docs_list)} documents")

# 3️⃣ Initialize Retriever
print("🏗️  Initializing GPU Retriever (Indexing 5,183 docs)...")
start_init = time.time()
retriever = LlamaHybridRetriever(docs_list)
print(f"✅ Indexing complete in {time.time() - start_init:.2f}s")

def calculate_mrr(results, qrels):
    rr_scores = []
    for qid, doc_dict in results.items():
        gold_ids = list(qrels.get(qid, {}).keys())
        # Sort retrieved docs by score
        sorted_docs = sorted(doc_dict.items(), key=lambda x: x[1], reverse=True)
        
        rank = 0
        for i, (doc_id, _) in enumerate(sorted_docs):
            if doc_id in gold_ids:
                rank = i + 1
                break
        rr_scores.append(1.0 / rank if rank > 0 else 0.0)
    return np.mean(rr_scores)

async def retrieve_task(query):
    return await asyncio.to_thread(retriever.retrieve, query)

async def evaluate():
    results = {}
    total = len(test_queries)
    
    print(f"🚀 Running evaluation on ALL {total} queries...\n")
    start_time = time.time()

    for i, (qid, query) in enumerate(test_queries, start=1):
        iter_start = time.time()
        
        # Retrieval
        docs = await retrieve_task(query)
        results[qid] = {doc["id"]: doc["score"] for doc in docs}

        # Progress update every 10 queries
        if i % 10 == 0 or i == total:
            elapsed = time.time() - start_time
            avg = elapsed / i
            eta = avg * (total - i)
            print(f"[{i}/{total}] Avg: {avg:.2f}s | ETA: {eta/60:.2f} mins")

    # 4️⃣ Metrics Calculation
    print("\n🧮 Calculating final metrics...")
    evaluator = EvaluateRetrieval(k_values=[1, 3, 5])
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, evaluator.k_values)
    mrr = calculate_mrr(results, qrels)

    print("\n" + "="*45)
    print("📊 FULL DATASET EVALUATION RESULTS")
    print("="*45)
    print(f"MRR:         {mrr:.4f}")
    print(f"Precision@1: {precision['P@1']:.4f}")
    for k in [1, 3, 5]:
        print(f"K={k} | NDCG: {ndcg[f'NDCG@{k}']:.4f} | Recall: {recall[f'Recall@{k}']:.4f}")
    print("="*45)

if __name__ == "__main__":
    asyncio.run(evaluate())