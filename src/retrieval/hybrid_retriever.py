# retrieval/hybrid_retriever.py

import asyncio


class HybridRetriever:
    def __init__(self, vector_store, keyword_store):
        self.vector_store = vector_store
        self.keyword_store = keyword_store

    async def retrieve(self, query_embedding, query, top_k):

        vector_task = asyncio.to_thread(
            self.vector_store.search, query_embedding, top_k
        )

        keyword_task = asyncio.to_thread(
            self.keyword_store.keyword_search, query, top_k
        )

        vector_results, keyword_results = await asyncio.gather(
            vector_task, keyword_task
        )

        # normalize keyword results
        keyword_results = [
            {"text": r["text"], "score": 0.5}
            for r in keyword_results
        ]

        return vector_results + keyword_results