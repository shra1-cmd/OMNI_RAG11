from indexing.pipeline import IngestionPipeline
from pathlib import Path
import os
from embeddings.embedder import embed

def test_indexing_query():
    session_id = "test-session-final"
    pipeline = IngestionPipeline(session_id)
    
    # Create a small dummy file
    test_file = Path("c:/Users/shravan/Documents/OMNI_RAG/data/test_file.txt")
    test_file.write_text("Hardware and Schedule: We trained our models on 8 NVIDIA P100 GPUs. This is a very specific test string.")
    
    print(f"Indexing file: {test_file}")
    try:
        pipeline.process_new_file(test_file)
        print("Indexing completed. Now querying immediately...")
        
        # Test query
        test_query = "What hardware was used?"
        query_vec = embed(test_query)
        
        from vectorstore.pinecone_store import PineconeStore
        store = PineconeStore()
        
        # Query with session filter
        res = store.query(query_vec, filter={"session_id": {"$eq": session_id}}, top_k=5)
        print(f"Query Results (filtered): {len(res['matches'])}")
        for m in res['matches']:
            print(f"- Found: {m['metadata'].get('text')[:50]}... (Score: {m['score']})")
            
        # Query WITHOUT session filter
        res2 = store.query(query_vec, top_k=5)
        print(f"Query Results (unfiltered): {len(res2['matches'])}")
        for m in res2['matches']:
            print(f"- Found: {m['metadata'].get('text')[:50]}... (Score: {m['score']}, Session: {m['metadata'].get('session_id')})")

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_indexing_query()
