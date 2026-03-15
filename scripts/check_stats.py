import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def check_stats():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    pc = Pinecone(api_key=api_key)
    
    print(f"Checking index: {index_name}")
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Full Stats: {stats}")
        print(f"Total vector count: {stats.get('total_vector_count')}")
        
        # Check namespaces/partitions if any
        # (We use metadata filtering, not namespaces)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_stats()
