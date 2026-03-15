import os
from pinecone import Pinecone
from dotenv import load_dotenv
from embeddings.embedder import embed

load_dotenv()

def debug_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    
    if not api_key or not index_name:
        print("Missing env vars")
        return

    pc = Pinecone(api_key=api_key)
    
    print(f"Target Index: {index_name}")
    
    try:
        desc = pc.describe_index(index_name)
        print(f"Index Description: {desc}")
        
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Index Stats: {stats}")
        
        # Test query
        test_text = "NVIDIA GPUs"
        test_vec = embed(test_text)
        print(f"Embedding Dim: {len(test_vec)}")
        
        results = index.query(vector=test_vec.tolist(), top_k=5, include_metadata=True)
        print(f"Unfiltered Query Results (Count): {len(results['matches'])}")
        for match in results['matches']:
            print(f"- Score: {match['score']}, Session: {match.get('metadata', {}).get('session_id')}")

    except Exception as e:
        print(f"Error during debug: {e}")

if __name__ == "__main__":
    debug_pinecone()
