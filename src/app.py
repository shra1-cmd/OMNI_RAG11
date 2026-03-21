# app.py

import streamlit as st
import asyncio
import uuid
from datetime import datetime

from core.pipeline import RAGPipeline
from storage.pinecone_store import PineconeStore
from storage.mongo_store import MongoStore
from indexing.ingest import Ingestor
from indexing.ingest import read_pdf, read_md, read_csv

st.set_page_config(page_title="OmniRAG v2", layout="wide")

# ✨ Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.session_start_time = datetime.now().strftime("%H:%M:%S")

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# --- Sidebar ---
st.sidebar.title("🧠 Session Info")
st.sidebar.info(f"""
**ID:** `{st.session_state.session_id[:8]}...`  
**Started at:** `{st.session_state.session_start_time}`
""")

st.sidebar.divider()
st.sidebar.title("📂 Document Manager")

uploaded_files = st.sidebar.file_uploader(
    "Upload multiple files", type=["pdf", "md", "csv"], accept_multiple_files=True
)

# Separation of Indexed and Pending
pending_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
indexed_files_list = [f for f in uploaded_files if f.name in st.session_state.indexed_files]

if pending_files:
    st.sidebar.subheader("⏳ Pending Indexing")
    for f in pending_files:
        col1, col2 = st.sidebar.columns([3, 1])
        col1.write(f"📄 {f.name}")
        if col2.button("Index", key=f"btn_{f.name}"):
            ingestor = Ingestor(PineconeStore(), MongoStore())
            
            # Read content based on type
            if f.type == "application/pdf":
                text = read_pdf(f)
            elif f.name.endswith(".md"):
                text = read_md(f)
            elif f.name.endswith(".csv"):
                text = read_csv(f)
            else:
                text = None

            if text:
                with st.spinner(f"Indexing {f.name}..."):
                    count = ingestor.process_text(text, f.name, st.session_state.session_id)
                st.session_state.indexed_files.add(f.name)
                st.sidebar.success(f"✅ {f.name} ({count} chunks)")
                st.rerun()

if indexed_files_list:
    st.sidebar.subheader("✅ Indexed Documents")
    for f in indexed_files_list:
        st.sidebar.write(f"✔️ {f.name}")

st.sidebar.divider()

# 🗑️ Cleanup Button
if st.sidebar.button("🗑️ Clear Session Data", use_container_width=True):
    with st.spinner("Cleaning up..."):
        p_store = PineconeStore()
        m_store = MongoStore()
        p_store.delete_by_session(st.session_state.session_id)
        count = m_store.delete_by_session(st.session_state.session_id)
        
        # Reset session state
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.session_start_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.indexed_files = set()
        
        st.sidebar.success(f"✅ Session reset and data cleared!")
        st.rerun()

# 🔥 Pipeline Setup
@st.cache_resource
def load_pipeline():
    return RAGPipeline(PineconeStore(), MongoStore())

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Custom Styling for Chat ---
st.markdown("""
<style>
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("🧠 OmniRAG — Ask Your Documents")
st.markdown("#### `Hybrid RAG | LlamaIndex | HF Embeddings | Groq`")
st.divider()

# --- Display Chat History ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "steps" in message and message["steps"]:
            with st.expander("🧠 View thinking process", expanded=False):
                for step in message["steps"]:
                    st.markdown(f"**{step['name']}**")
                    st.caption(step["detail"])
                    st.divider()
        if "sources" in message and message["sources"]:
            with st.expander("📚 View sources", expanded=False):
                for i, s in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}** (score: {round(s['score'], 4)})")
                    st.write(s["text"])
                    st.divider()

# --- Chat Input ---
if prompt := st.chat_input("Ask a question from the ingested documents..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Display assistant response
    with st.chat_message("assistant"):
        # Create a status container for the "Thinking" process
        with st.status("🧠 OmniRAG is thinking...", expanded=True) as status:
            pipeline = load_pipeline()
            # Run pipeline
            result = asyncio.run(pipeline.run(prompt, session_id=st.session_state.session_id))
            
            # Show steps in the status container
            for step in result.get("steps", []):
                st.write(f"✔️ {step['name']}")
            
            status.update(label="✅ Processing Complete!", state="complete", expanded=False)

        # Display final answer
        st.markdown(result["answer"])

        # Display Sources
        if result["sources"]:
            with st.expander("📚 View sources", expanded=False):
                for i, s in enumerate(result["sources"], 1):
                    st.markdown(f"**Source {i}** (score: {round(s['score'], 4)})")
                    st.write(s["text"])
                    st.divider()

        # Add to history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": result["answer"],
            "steps": result.get("steps", []),
            "sources": result.get("sources", [])
        })
        
        # Confidence & Metadata (Optional)
        with st.expander("⚙️ Advanced RAG Details (System Metadata)"):
             st.write({
                "confidence_score": result["confidence"],
                "retrieval_strategy": "Hybrid Search (Vector + BM25)",
                "used_hyde_fallback": result.get("used_hyde", False),
                "cross_encoder_reranking": "Enabled",
                "embedding_model": "HuggingFace BGE-Large-v1.5",
                "generation_llm": "Groq API (LPU Accelerated)",
            })
