# llm/chat_model.py

import os
import time
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def generate(prompt: str, temperature: float = 0.0) -> str:
    """
    Groq LLM call.
    Fast and free-tier friendly.
    """

    # small delay to avoid burst rate limits
    time.sleep(0.5)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # stable + fast
        messages=[
            {"role": "system", "content": "You are a professional RAG research assistant. Your goal is to provide accurate answers based solely on the provided documentation.try to answer in 5 lines "},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()
