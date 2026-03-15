# indexing/pipeline.py

import os
import pickle
import hashlib
import re
from datetime import datetime
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import TextNode
from vectorstore.pinecone_store import PineconeStore

class IngestionPipeline:
    def __init__(self, session_id: str = None):
        self.project_root = Path(__file__).parent.parent
        self.session_id = session_id # CRITICAL FIX
        
        if session_id:
            self.base_data_dir = self.project_root / "data" / "sessions" / session_id
        else:
            self.base_data_dir = self.project_root / "data"
            
        self.raw_dir = self.base_data_dir / "raw"
        self.processed_dir = self.base_data_dir / "processed"
        self.faiss_dir = self.base_data_dir / "faiss_index"
        
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)
        
        from llm.embedding_model import EmbeddingModel
        self.embedding_engine = EmbeddingModel()
        self.model = self.embedding_engine.model
        
        self.MAX_TOKENS = 350
        self.SIM_THRESHOLD = 0.75

    def hash_text(self, text: str):
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def split_sentences(self, text: str):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 0]

    def process_new_file(self, file_path):
        file_path = Path(file_path)
        print(f"Processing {file_path}...")
        
        # 1. Ingestion
        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        docs = reader.load_data()

        clean_docs = []
        for doc in docs:
            if len(doc.text.strip()) < 50: # Lowered threshold for smaller files
                continue
            
            doc.metadata.update({
                "source_type": file_path.suffix.lower().replace(".", ""),
                "file_name": file_path.name,
                "file_path": str(file_path),
                "ingested_at": datetime.utcnow().isoformat(),
                "content_hash": self.hash_text(doc.text),
                "session_id": str(self.session_id) if hasattr(self, 'session_id') and self.session_id else "default"
            })
            clean_docs.append(doc)
            
        if not clean_docs:
            print(f"No valid content found in {file_path.name}.")
            return False

        # 2. Chunking
        all_nodes = []
        for doc in clean_docs:
            sentences = self.split_sentences(doc.text)
            if not sentences:
                continue
                
            sentence_embeddings = self.model.encode(sentences, normalize_embeddings=True)
            
            chunks = []
            current_chunk = [sentences[0]]
            current_emb = sentence_embeddings[0]
            
            for i in range(1, len(sentences)):
                sim = cosine_similarity(
                    current_emb.reshape(1, -1),
                    sentence_embeddings[i].reshape(1, -1)
                )[0][0]
                
                prospective_text = " ".join(current_chunk + [sentences[i]])
                
                if sim > self.SIM_THRESHOLD and len(prospective_text.split()) < self.MAX_TOKENS:
                    current_chunk.append(sentences[i])
                    # OPTIMIZED: Use pre-calculated running mean instead of re-encoding
                    group_embeddings = sentence_embeddings[i-len(current_chunk)+1:i+1]
                    current_emb = np.mean(group_embeddings, axis=0)
                else:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [sentences[i]]
                    current_emb = sentence_embeddings[i]
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                
            for chunk in chunks:
                node = TextNode(
                    text=chunk,
                    metadata=doc.metadata.copy()
                )
                all_nodes.append(node)
                
        # Update metadata for chunks
        for idx, node in enumerate(all_nodes):
            node.metadata["chunk_id"] = f"{node.metadata.get('file_name')}_{idx}"
            node.metadata["embedding_model"] = "bge-large-v1.5"
            node.metadata["chunk_strategy"] = "semantic_sentence_merge"

        # 3. Indexing (Incremental update)
        index_file = self.faiss_dir / "faiss.index"
        meta_file = self.faiss_dir / "faiss_chunks_meta.pkl"
        
        EMBED_DIM = self.model.get_sentence_embedding_dimension()
        
        if index_file.exists() and meta_file.exists():
            index = faiss.read_index(str(index_file))
            with open(meta_file, "rb") as f:
                existing_nodes = pickle.load(f)
        else:
            index = faiss.IndexFlatIP(EMBED_DIM)
            existing_nodes = []
            
        new_texts = [node.text for node in all_nodes]
        new_embeddings = self.model.encode(
            new_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        ).astype("float32")
        
        # Pinecone Indexing
        try:
            pinecone_store = PineconeStore()
            pinecone_store.upsert_batch(all_nodes, new_embeddings)
        except Exception as e:
            print(f"Error upserting to Pinecone: {e}")
            print("Falling back to local FAISS index...")
            index.add(new_embeddings)
            existing_nodes.extend(all_nodes)
            faiss.write_index(index, str(index_file))
            with open(meta_file, "wb") as f:
                pickle.dump(existing_nodes, f)
            
        print(f"Successfully processed {len(all_nodes)} chunks.")
        return True
