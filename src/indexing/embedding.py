# indexing/embedding.py

from functools import lru_cache
from sentence_transformers import SentenceTransformer

# ✅ cache model (loads only once)
@lru_cache(maxsize=1)
def get_embedder(model_name: str):
    return SentenceTransformer(model_name)

class Embedder:
    def __init__(self, model_name: str):
        self.model = get_embedder(model_name)

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, normalize_embeddings=True)