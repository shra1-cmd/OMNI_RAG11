import os
import sys
from pathlib import Path

# Add project root to sys.path so 'agents' modular imports work on Streamlit Cloud
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import shutil
import time
import streamlit as st
from agents.runner import run_agent, pdf_retriever
from indexing.pipeline import IngestionPipeline
from vectorstore.pinecone_store import PineconeStore
from app.worker import IndexingWorker
from dotenv import load_dotenv

load_dotenv()

# -----------------------
# Session state initialization
# -----------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session-{int(time.time())}"
    st.session_state.chat = []
    st.session_state.indexed_files = [] # Track names of files already indexed
    st.session_state.worker = IndexingWorker(st.session_state.session_id)

# -----------------------
# Background Worker Sync
# -----------------------
# Move completed files from worker to UI state
worker = st.session_state.worker
for f_name in worker.status["completed_files"]:
    if f_name not in st.session_state.indexed_files:
        st.session_state.indexed_files.append(f_name)

# -----------------------
# Cleanup sessions on startup
# -----------------------
if "cleanup_done" not in st.session_state:
    # 1. Local cleanup
    SESSION_DATA_DIR = os.path.join("data", "sessions")
    if os.path.exists(SESSION_DATA_DIR):
        try:
            shutil.rmtree(SESSION_DATA_DIR)
        except Exception as e:
            print(f"Error during local cleanup: {e}")
            
    # 2. Pinecone total cleanup for a fresh start
    try:
        store = PineconeStore()
        store.clear_index() # Wipe everything for this user session fresh start
    except Exception as e:
        print(f"Error during Pinecone complete cleanup: {e}")
        
    st.session_state.cleanup_done = True

st.set_page_config(
    page_title="OmniRAG",
    layout="wide"
)

# -----------------------
# Sidebar - Upload
# -----------------------
with st.sidebar:
    st.header("📄 Upload Documents")
    
    # 1. Indexed Files (Above the button)
    if st.session_state.indexed_files:
        st.subheader("✅ Indexed Documents")
        for f_name in st.session_state.indexed_files:
            st.info(f"📄 {f_name}")
        st.divider()

    # 2. Upload Area
    uploaded_files = st.file_uploader(
        "Upload Files", 
        type=["pdf", "docx", "pptx", "txt", "md", "csv", "xlsx", "json", "xml"], 
        accept_multiple_files=True
    )
    
    # Filter out files that are already indexed or in queue
    pending_files = []
    if uploaded_files:
        pending_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        
    # 3. Worker Status (Progress Indicators)
    if worker.status["is_running"]:
        st.subheader("⚙️ Background Indexing")
        st.progress(worker.status["progress"])
        st.caption(f"Indexing: {worker.status['current_file']} ({worker.status['progress']}%)")
        if worker.status["error"]:
            st.error(worker.status["error"])
        # Trigger an automatic rerun to refresh progress until finished
        time.sleep(1)
        st.rerun()

    # 4. Action Button
    if pending_files:
        if st.button("Index Documents", disabled=worker.status["is_running"]):
            # Create pipeline only to get raw_dir for saving
            dummy_pipeline = IngestionPipeline(session_id=st.session_state.session_id)
            for uploaded_file in pending_files:
                save_path = os.path.join(dummy_pipeline.raw_dir, uploaded_file.name)
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Hand over to background worker
                worker.add_task(save_path, uploaded_file.name)
            
            st.info("Added to background indexing queue.")
            st.rerun()
                
        st.subheader("⏳ Pending Indexing")
        for f in pending_files:
            st.warning(f"📄 {f.name} (Waiting)")

st.title("🧠 OmniRAG — Ask Your Documents")

# -----------------------
# Render chat history
# -----------------------
for role, message in st.session_state.chat:
    with st.chat_message(role):
        if isinstance(message, dict):
            st.markdown(message["answer"])
            if message.get("sources"):
                with st.expander("📚 Sources"):
                    for src in message["sources"]:
                        st.markdown(
                            f"- **Chunk {src['chunk_id']}** | "
                            f"Similarity: `{src['score']:.3f}` | "
                            f"File: `{src['file_name']}`"
                        )
        else:
            st.markdown(message)

# -----------------------
# Chat input
# -----------------------
user_input = st.chat_input("Ask a question from the ingested documents...")

if user_input:
    st.session_state.chat.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    result = run_agent(
        user_query=user_input,
        session_id=st.session_state.session_id
    )

    st.session_state.chat.append(("assistant", result))
    with st.chat_message("assistant"):
        st.markdown("### 🤖 Answer")
        st.markdown(result["answer"])

        if result.get("sources"):
            with st.expander("📚 Sources"):
                for i, src in enumerate(result["sources"], start=1):
                    st.markdown(
                        f"""
**Source {i}**
- **Chunk ID:** `{src['chunk_id']}`
- **Similarity:** `{src['score']:.3f}`
- **File:** `{src['file_name']}`
---
"""
                    )
