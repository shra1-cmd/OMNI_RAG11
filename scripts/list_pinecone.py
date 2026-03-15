import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

def list_indices():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY missing")
        return

    pc = Pinecone(api_key=api_key)
    try:
        indexes = pc.list_indexes()
        print("Available Indices:")
        for idx in indexes:
            print(f"- {idx.name} ({idx.dimension} dim, {idx.metric} metric, status: {idx.status['ready']})")
    except Exception as e:
        print(f"Error listing indices: {e}")

if __name__ == "__main__":
    list_indices()
