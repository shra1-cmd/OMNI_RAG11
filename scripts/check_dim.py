from llm.embedding_model import EmbeddingModel
import numpy as np

model = EmbeddingModel()
text = "Hello world"
emb = model.encode(text)
print(f"Model: {model.model}")
print(f"Embedding shape: {emb.shape}")
print(f"Embedding length: {len(emb[0])}")
