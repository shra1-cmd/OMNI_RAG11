def summarize_session(llm, messages: list) -> str:
    prompt = f"""
Summarize the following conversation into reusable memory.
Focus on:
- User preferences
- Decisions made
- Facts learned
Be concise.

Conversation:
{messages}
"""
    return llm(prompt)
