class MemoryManager:
    def __init__(self, stm, ltm, summarizer):
        self.stm = stm
        self.ltm = ltm
        self.summarizer = summarizer

    def add_message(self, role, content):
        self.stm.add(role, content)

    def recall(self, query):
        return self.ltm.search(query)

    def persist_session(self, llm, session_id):
        summary = self.summarizer(llm, self.stm.get())
        self.ltm.add(
            summary,
            metadata={
                "type": "summary",
                "session_id": session_id
            }
        )
