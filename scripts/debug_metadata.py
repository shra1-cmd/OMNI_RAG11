import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def debug_query():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    print(f"Stats for {index_name}:")
    print(index.describe_index_stats())
    
    # Fetch some records to see metadata
    # We can't easily list all, but we can query with a dummy vector
    import numpy as np
    dummy_vec = np.random.rand(384).tolist()
    
    res = index.query(vector=dummy_vec, top_k=10, include_metadata=True)
    print("\nSample records from unfiltered query:")
    for match in res['matches']:
        print(f"ID: {match['id']}, Score: {match['score']}, Metadata: {match.get('metadata')}")

if __name__ == "__main__":
    debug_query()
