# vectorstore/pinecone_store.py

import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

class PineconeStore:
    def __init__(self, index_name=None, dimension=1024):
        self.api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = index_name or os.getenv("PINECONE_INDEX_NAME")
        self.host = os.getenv("PINECONE_HOST")
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables.")
        if not self.index_name:
            raise ValueError("PINECONE_INDEX_NAME not found in environment variables.")

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Connect to index using host if provided (faster and more direct)
        if self.host:
            print(f"Connecting to Pinecone index via host: {self.host}")
            self.index = self.pc.Index(host=self.host)
        else:
            # Check if index exists, create if not
            if self.index_name not in [idx.name for idx in self.pc.list_indexes()]:
                print(f"Creating index {self.index_name}...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                # Wait for index to be ready
                while not self.pc.describe_index(self.index_name).status['ready']:
                    time.sleep(1)
            
            self.index = self.pc.Index(self.index_name)

    def upsert_batch(self, nodes, embeddings, batch_size=100):
        """
        Upsert nodes and their embeddings to Pinecone in batches.
        """
        vectors_to_upsert = []
        for i, (node, emb) in enumerate(zip(nodes, embeddings)):
            vectors_to_upsert.append({
                "id": f"doc_{node.metadata.get('content_hash', i)}_{i}",
                "values": emb.tolist(),
                "metadata": {
                    "text": node.text,
                    **node.metadata
                }
            })
            
            if len(vectors_to_upsert) >= batch_size:
                self.index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []
        
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            
        print(f"Successfully upserted {len(nodes)} chunks to Pinecone.")

    def query(self, query_vector, filter=None, top_k=3):
        """
        Query Pinecone for the top_k matches with optional filtering.
        """
        results = self.index.query(
            vector=query_vector.tolist() if hasattr(query_vector, "tolist") else query_vector,
            top_k=top_k,
            filter=filter,
            include_metadata=True
        )
        return results

    def clear_index(self):
        """
        Delete ALL vectors in the index for a completely fresh start.
        """
        print(f"Wiping ALL data from Pinecone index: {self.index_name}")
        try:
            self.index.delete(delete_all=True)
            print("Successfully cleared all data from index.")
        except Exception as e:
            print(f"Error clearing index: {e}")

    def delete_by_session(self, session_id):
        """
        Delete all vectors associated with a specific session_id.
        """
        if not session_id:
            return
            
        print(f"Cleaning up Pinecone data for session: {session_id}")
        try:
            # Delete by filter
            self.index.delete(filter={"session_id": {"$eq": session_id}})
            print(f"Successfully cleared data for session {session_id}")
        except Exception as e:
            print(f"Error deleting session data: {e}")
