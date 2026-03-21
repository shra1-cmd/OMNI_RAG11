import torch
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class LlamaHybridRetriever:
    def __init__(self, documents):
        # ✅ Detect GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🚀 Initializing Embedding Model on: {device.upper()}")

        # ✅ Set embedding with GPU support
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-base-en-v1.5",
            device=device,
            embed_batch_size=32  # GPU handles batches much faster
        )

        # ✅ Disable OpenAI LLM
        Settings.llm = None

        # ✅ IMPORTANT: Preserve BEIR doc_id or auto-generate one
        self.docs = [
            Document(
                text=d["text"],
                metadata={"doc_id": d.get("id", str(i))}
            )
            for i, d in enumerate(documents)
        ]

        # Node parsing
        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(self.docs)

        # Vector index (This will use the GPU via Settings)
        self.index = VectorStoreIndex(nodes)
        self.vector_retriever = self.index.as_retriever(similarity_top_k=5)

        # BM25 (Note: BM25 is a CPU-based keyword search)
        self.bm25_retriever = BM25Retriever.from_defaults(nodes=nodes)

        # Fusion
        self.retriever = QueryFusionRetriever(
            [self.vector_retriever, self.bm25_retriever],
            similarity_top_k=5,
            num_queries=1, # No LLM query generation
            use_async=True
        )

    def retrieve(self, query):
        results = self.retriever.retrieve(query)

        return [
            {
                "id": r.node.metadata["doc_id"],
                "text": r.node.text,
                "score": float(r.score) if r.score else 0.0
            }
            for r in results
        ]