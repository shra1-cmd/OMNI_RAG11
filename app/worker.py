import threading
import queue
import time
import os
from indexing.pipeline import IngestionPipeline
from agents.runner import pdf_retriever

class IndexingWorker:
    def __init__(self, session_id):
        self.session_id = session_id
        self.task_queue = queue.Queue()
        self.status = {
            "is_running": False,
            "current_file": None,
            "progress": 0,
            "completed_files": [],
            "error": None
        }
        self.worker_thread = None

    def start(self):
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._run, daemon=True)
            self.worker_thread.start()

    def add_task(self, file_path, file_name):
        self.task_queue.put((file_path, file_name))
        if not self.status["is_running"]:
            self.start()

    def _run(self):
        self.status["is_running"] = True
        pipeline = IngestionPipeline(session_id=self.session_id)
        
        while not self.task_queue.empty():
            file_path, file_name = self.task_queue.get()
            self.status["current_file"] = file_name
            self.status["progress"] = 0
            
            try:
                # Update progress roughly at start
                self.status["progress"] = 10
                
                # Run the actual indexing
                # Note: In a more complex setup, we'd pass a callback to the pipeline for granular progress
                success = pipeline.process_new_file(file_path)
                
                if success:
                    self.status["progress"] = 100
                    self.status["completed_files"].append(file_name)
                    # Reload retriever to ensure new data is available for queries
                    pdf_retriever.reload(session_id=self.session_id)
                else:
                    self.status["error"] = f"Failed to index {file_name}"
            except Exception as e:
                self.status["error"] = str(e)
            finally:
                self.task_queue.task_done()
                time.sleep(1) # Small delay for UI smoothness
                self.status["current_file"] = None
                self.status["progress"] = 0

        self.status["is_running"] = False
