# core/rewriter.py

class QueryRewriter:
    def rewrite(self, query: str) -> str:
        # simple version (fast)
        if len(query.split()) < 3:
            return f"Explain in detail: {query}"
        return query