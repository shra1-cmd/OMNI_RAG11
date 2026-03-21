# core/config.py

class Settings:
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    RERANK_MODEL = "BAAI/bge-reranker-base"

    TOP_K_DEFAULT = 5
    TOP_K_MIN = 3
    TOP_K_MAX = 8

    HYDE_THRESHOLD = 0.65
    RERANK_TOP_K = 5

    USE_HYDE = True