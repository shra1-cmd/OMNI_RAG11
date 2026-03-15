# vectorstore/faiss_store.py
import faiss
import os
import pickle

class FaissStore:
    def __init__(self, embed_fn, index_path="data/faiss_index"):
        self.embed_fn = embed_fn
        self.index_path = index_path
        self.index_file = os.path.join(index_path, "index.faiss")
        self.meta_file = os.path.join(index_path, "meta.pkl")

        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "rb") as f:
                self.metadata = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(768)  # embedding dim
            self.metadata = []

    def add(self, text, metadata):
        vec = self.embed_fn(text)
        self.index.add(vec.reshape(1, -1))
        self.metadata.append({"text": text, **metadata})

        os.makedirs(self.index_path, exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, "wb") as f:
            pickle.dump(self.metadata, f)

    def search(self, query, k=5):
        if len(self.metadata) == 0 or self.index.ntotal == 0:
            return []

        vec = self.embed_fn(query)
        distances, indices = self.index.search(vec.reshape(1, -1), k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

