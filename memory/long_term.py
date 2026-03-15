import uuid
from typing import List
from vectorstore.faiss_store import FaissStore

class LongTermMemory:
    def __init__(self, embedder, index_path="memory/faiss"):
        self.store = FaissStore(embedder, index_path)

    def add(self, text: str, metadata: dict):
        metadata["id"] = str(uuid.uuid4())
        self.store.add(text, metadata)

    def search(self, query: str, k: int = 5):
        print(f"[LTM] Searching FAISS for query: '{query}'")
        results = self.store.search(query, k)
        print(f"[LTM] FAISS returned {len(results)} results")
        return results

