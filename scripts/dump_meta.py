import os
from pinecone import Pinecone
from dotenv import load_dotenv
import numpy as np

load_dotenv()

def dump_metadata():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    # Query with a random vector to get any records
    dummy_vec = np.random.rand(384).tolist()
    res = index.query(vector=dummy_vec, top_k=1, include_metadata=True)
    
    if not res['matches']:
        print("No matches found even unfiltered.")
        return

    match = res['matches'][0]
    print(f"ID: {match['id']}")
    print(f"Score: {match['score']}")
    print("Full Metadata Keys:")
    print(list(match['metadata'].keys()))
    print("Full Metadata Content:")
    import json
    print(json.dumps(match['metadata'], indent=2))

if __name__ == "__main__":
    dump_metadata()
