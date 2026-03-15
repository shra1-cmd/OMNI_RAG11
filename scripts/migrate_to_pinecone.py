# scripts/migrate_to_pinecone.py

import os
import pickle
import faiss
import numpy as np
from vectorstore.pinecone_store import PineconeStore
from dotenv import load_dotenv

load_dotenv()

def migrate():
    index_path = os.path.join("data", "faiss_index")
    index_file = os.path.join(index_path, "faiss.index")
    meta_file = os.path.join(index_path, "faiss_chunks_meta.pkl")

    if not os.path.exists(index_file) or not os.path.exists(meta_file):
        print(f"No local FAISS index found at {index_path}. Nothing to migrate.")
        return

    print(f"Loading local data from {index_path}...")
    index = faiss.read_index(index_file)
    with open(meta_file, "rb") as f:
        nodes = pickle.load(f)

    # Get embeddings from FAISS index
    # Note: IndexFlatIP doesn't easily expose embeddings unless it's a specific type.
    # However, for migration, we might need to re-embed if we can't extract them.
    # But wait, IndexFlatIP *is* just a flat array of vectors.
    
    # Let's try to extract vectors if possible, else re-embed is safer.
    # For now, let's assume we might need to re-embed to be 100% sure about the metadata mapping.
    # Actually, the nodes themselves might not have embeddings attached.
    
    print(f"Found {len(nodes)} chunks. Initializing PineconeStore...")
    try:
        pinecone_store = PineconeStore()
    except Exception as e:
        print(f"Error: {e}")
        return

    # To be safe and ensure alignment, we'll re-embed the text from the nodes.
    # This ensures the vectors match the current model and metadata.
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-base-en")
    
    texts = [node.text for node in nodes]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, normalize_embeddings=True)

    print("Upserting to Pinecone...")
    pinecone_store.upsert_batch(nodes, embeddings)
    print("Migration complete!")

if __name__ == "__main__":
    migrate()
