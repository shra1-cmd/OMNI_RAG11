# retrievers/pdf_retriever.py

import os
from embeddings.embedder import embed
from vectorstore.pinecone_store import PineconeStore


class PDFRetriever:
    """
    Retriever over Pinecone knowledge base.
    """

    def __init__(self, top_k=3):
        self.top_k = top_k
        self.pinecone_store = None
        try:
            self.pinecone_store = PineconeStore()
            print("PineconeStore initialized for PDFRetriever.")
        except Exception as e:
            print(f"Warning: Failed to initialize PineconeStore: {e}")

    def reload(self, session_id: str = None):
        """
        In Pinecone mode, we don't necessarily need to reload local files,
        but we can refresh the connection or handle session filtering if implemented.
        """
        try:
            if not self.pinecone_store:
                self.pinecone_store = PineconeStore()
            print("PineconeStore connection refreshed.")
        except Exception as e:
            print(f"Error refreshing PineconeStore: {e}")

    def retrieve(self, query: str, session_id: str = None):
        """
        Retrieve top-k relevant chunks from Pinecone for a query, 
        scoped to a specific session_id.
        """
        if not self.pinecone_store:
            print("Error: PineconeStore not initialized. Returning empty results.")
            return []

        query_vec = embed(query)
        
        # Apply session filter if provided
        pinecone_filter = None
        if session_id:
            pinecone_filter = {"session_id": {"$eq": session_id}}

        results = self.pinecone_store.query(
            query_vec, 
            filter=pinecone_filter,
            top_k=self.top_k
        )

        formatted_results = []
        for rank, match in enumerate(results.get("matches", [])):
            metadata = match.get("metadata", {})
            formatted_results.append({
                "rank": rank + 1,
                "score": float(match.get("score", 0)),
                "chunk": metadata # Contains 'text' and other metadata
            })

        return formatted_results

