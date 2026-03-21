# storage/pinecone_store.py

from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
load_dotenv()

class PineconeStore:
    def __init__(self, index_name="omniraq-index"):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

        if index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pc.Index(index_name)

    def delete_by_session(self, session_id):
        if not session_id:
            return
        self.index.delete(filter={"session_id": {"$eq": session_id}})

    def upsert(self, vectors):
        self.index.upsert(vectors)

    def search(self, embedding, top_k=5, session_id=None):
        filter = None
        if session_id:
            filter = {"session_id": {"$eq": session_id}}

        res = self.index.query(
            vector=embedding.tolist(),
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )

        results = []
        for match in res["matches"]:
            results.append({
                "text": match["metadata"]["text"],
                "score": match["score"]
            })

        return results