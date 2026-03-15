from llm.chat_model import generate

class Executor:
    def execute(self, plan):
        docs = plan["retrieved_docs"]
        # Drop junk / near-empty chunks
        docs = [
            d for d in docs
            if d["chunk"].get("text")
            and len(d["chunk"].get("text").strip()) > 10
        ]


        if not docs:
            return {
                "answer": "Not found in the knowledge base",
                "sources": []
            }

        context_blocks = []

        for d in docs:
            chunk = d["chunk"]
            context_blocks.append(
                f"SOURCE {d['rank']} | File: {chunk.get('file_name')}\n"
                f"Content: {chunk.get('text').strip()}"
            )

        context = "\n\n---\n\n".join(context_blocks)


        prompt = f"""
        Extract the answer to the Question from the provided Context below.
        
        Rules:
        - Use ONLY the information from the Context.
        - Be concise and factual.
        - If the Context really does not contain the answer, reply: "Not found in the knowledge base".
        
        Context:
        ---
        {context}
        ---
        
        Question: {plan['question']}
        
        Helpful Answer:"""

  

        answer = generate(prompt)

        return {
            "answer": answer,
            "sources": [
                {
                    "chunk_id": d["chunk"].get("chunk_id"),
                    "score": d["score"],
                    "file_name": d["chunk"].get("file_name"),
                    "source": d["chunk"].get("source_type"),
                }
                for d in docs
            ]
        }
