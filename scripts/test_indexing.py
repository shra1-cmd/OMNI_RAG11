from indexing.pipeline import IngestionPipeline
from pathlib import Path
import os

def test_indexing():
    session_id = "test-session"
    pipeline = IngestionPipeline(session_id)
    
    # Create a small dummy file
    test_file = Path("c:/Users/shravan/Documents/OMNI_RAG/data/test_file.txt")
    test_file.write_text("Hardware and Schedule: We trained our models on 8 NVIDIA P100 GPUs.")
    
    print(f"Indexing file: {test_file}")
    try:
        pipeline.process_new_file(test_file)
        print("Indexing completed without crash.")
    except Exception as e:
        print(f"INDEXING FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_indexing()
