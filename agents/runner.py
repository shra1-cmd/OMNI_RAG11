# agents/runner.py
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.memory_manager import MemoryManager
from memory.summarize import summarize_session

from planner.planner import Planner
from executor.executor import Executor
from embeddings.embedder import embed
from llm.chat_model import generate

from retrievers.pdf_retriever import PDFRetriever


from configs.debug import DEBUG

# Create memory objects ONCE
stm = ShortTermMemory(max_turns=10)
ltm = LongTermMemory(embed)
memory_manager = MemoryManager(stm, ltm, summarize_session)

planner = Planner()
executor = Executor()

pdf_retriever = PDFRetriever(
    top_k=2
)


from configs.debug import DEBUG

def log(msg):
    if DEBUG:
        print(msg)
        
def basic_query_filter(query: str) -> bool:
    query = query.strip()

    # Reject very short queries
    if len(query) < 10:
        return False

    # Reject fewer than 3 words
    if len(query.split()) < 3:
        return False

    return True

# def is_document_related(query: str) -> bool:
#     """
#     Uses LLM to determine if query requires document retrieval.
#     Must return strict YES or NO.
#     """

#     prompt = f"""
# You are a strict classifier.

# Does the following query require retrieving information 
# from the ingested documents?

# Answer ONLY:
# YES
# or
# NO

# Query:
# {query}
# """

#     response = generate(prompt)  # adjust if your LLM call differs
#     response = generate(prompt, temperature=0.0).strip().upper()


#     return response == "YES"
    
def retrieval_gate(results, threshold=0.70, margin=0.001):
    if not results:
        return False

    top_score = results[0]["score"]

    if top_score < threshold:
        return False

    if len(results) > 1:
        second_score = results[1]["score"]
        if (top_score - second_score) < margin:
            return False

    return True


def run_agent(user_query: str, session_id: str = "default"):
    log("\n================ NEW QUERY ================")
    log(f"[Runner] Received query: {user_query}")

    # 🔒 Layer 1: Basic Filter
    if not basic_query_filter(user_query):
        log("[Runner] Rejected at basic filter")
        return {
            "answer": "Please ask a meaningful document-related question.",
            "sources": []
        }

    # 🔒 Layer 2: Intent Classification
    # if not is_document_related(user_query):
    #     log("[Runner] Rejected at intent classifier")
    #     return {
    #         "answer": "This question does not appear related to the uploaded documents.",
    #         "sources": []
    #     }

    # 0. Reload retriever for session
    pdf_retriever.reload(session_id=session_id)

    # 1. Add user message to STM
    memory_manager.add_message("user", user_query)

    # 2. Retrieve PDF knowledge
    pdf_chunks = pdf_retriever.retrieve(user_query, session_id=session_id)

    if not retrieval_gate(pdf_chunks):
        log("[Runner] No chunks found or below threshold. Proceeding with baseline LLM knowledge.")
        pdf_chunks = [] # Ensure it's an empty list for the planner

    # 🔥 We'll add similarity gating next

    # 3. Recall memory
    agent_memory = memory_manager.recall(user_query)

    # 4. Planner
    plan = planner.create_plan(
        question=user_query,
        short_term=stm.get(),
        long_term=agent_memory,
        retrieved_docs=pdf_chunks
    )

    # 5. Execute
    answer = executor.execute(plan)

    memory_manager.add_message("assistant", answer)

    return answer
