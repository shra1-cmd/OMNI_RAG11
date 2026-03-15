# embeddings/embedder.py

from llm.embedding_model import EmbeddingModel

# Create ONE global embedding model instance
_embedding_model = EmbeddingModel()

def embed(text: str):
    """
    Convert text into a single embedding vector.
    This function is intentionally simple and stable.
    """
    return _embedding_model.encode(text)[0]
