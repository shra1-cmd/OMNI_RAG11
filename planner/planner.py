class Planner:
    def create_plan(self, question, short_term, long_term, retrieved_docs):
        return {
            "question": question,
            "short_term": short_term,
            "long_term": long_term,
            "retrieved_docs": retrieved_docs
        }
