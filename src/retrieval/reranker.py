# retrieval/reranker.py

from functools import lru_cache
from sentence_transformers import CrossEncoder

# ✅ cache model (loads only once)
@lru_cache(maxsize=1)
def get_reranker(model_name: str):
    return CrossEncoder(model_name)

class Reranker:
    def __init__(self, model_name: str):
        self.model = get_reranker(model_name)

    def rerank(self, query, docs):
        if not docs:
            return docs

        pairs = [[query, doc["text"]] for doc in docs]
        scores = self.model.predict(pairs)

        for doc, score in zip(docs, scores):
            doc["score"] = float(score)

        return sorted(docs, key=lambda x: x["score"], reverse=True)