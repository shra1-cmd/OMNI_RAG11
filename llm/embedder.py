# embeddings/embedder.py
from llm.embedding_model import EmbeddingModel

_embedding_model = EmbeddingModel()

def embed(text: str):
    return _embedding_model.encode(text)[0]
