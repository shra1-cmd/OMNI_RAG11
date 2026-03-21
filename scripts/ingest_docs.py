# scripts/ingest_docs.py

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from indexing.embedding import Embedder
from storage.pinecone_store import PineconeStore
from storage.mongo_store import MongoStore

embedder = Embedder("BAAI/bge-base-en-v1.5")
pinecone = PineconeStore()
mongo = MongoStore()


documents = [
    "Machine learning is a subset of AI that allows systems to learn from data.",
    "Deep learning is a neural network-based approach in machine learning.",
    "Supervised learning uses labeled datasets."
]

vectors = []
mongo_docs = []

for i, doc in enumerate(documents):
    embedding = embedder.embed(doc)[0]

    vectors.append({
        "id": str(i),
        "values": embedding.tolist(),
        "metadata": {"text": doc}
    })

    mongo_docs.append({"text": doc})


# push to DB
pinecone.upsert(vectors)
mongo.insert_chunks(mongo_docs)

print("✅ Data ingested successfully!")