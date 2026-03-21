# indexing/ingest.py
import pdfplumber
import pandas as pd
from indexing.embedding import Embedder
from indexing.chunking import chunk_text

def read_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def read_md(file):
    return file.read().decode("utf-8")


def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()
    
class Ingestor:
    def __init__(self, pinecone_store, mongo_store):
        self.embedder = Embedder("BAAI/bge-base-en-v1.5")
        self.pinecone = pinecone_store
        self.mongo = mongo_store

    def process_text(self, text, source="upload", session_id=None):
        chunks = chunk_text(text)

        vectors = []
        mongo_docs = []

        embeddings = self.embedder.embed(chunks)

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            metadata = {"text": chunk}
            if session_id:
                metadata["session_id"] = session_id

            vectors.append({
                "id": f"{source}_{i}",
                "values": emb.tolist(),
                "metadata": metadata
            })

            mongo_doc = {
                "text": chunk,
                "source": source
            }
            if session_id:
                mongo_doc["session_id"] = session_id
                
            mongo_docs.append(mongo_doc)

        self.pinecone.upsert(vectors)
        self.mongo.insert_chunks(mongo_docs)

        return len(chunks)