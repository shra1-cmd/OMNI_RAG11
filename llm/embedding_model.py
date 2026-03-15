from sentence_transformers import SentenceTransformer

class EmbeddingModel:
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True)
