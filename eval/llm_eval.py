import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from llm.groq_client import GroqClient

llm = GroqClient()

def evaluate_answer(question, context, answer):

    prompt = f"""
Evaluate the answer based ONLY on the context. Answer in 2 paragraph with 4 lines in one para

Question: {question}

Context:
{context}

Answer:
{answer}

Score:
1 = incorrect
5 = perfect

Give only a number.
"""

    return llm.generate(prompt)