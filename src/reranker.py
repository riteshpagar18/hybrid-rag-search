"""Cross-encoder reranking. The model is loaded lazily on first use."""
from __future__ import annotations

import os

_model = None


def get_reranker():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    return _model


def rerank_documents(question: str, documents: list[str], top_k: int = 5) -> list[dict]:
    if not documents or top_k <= 0:
        return []
    scores = get_reranker().predict([(question, document) for document in documents])
    ranked = sorted(zip(documents, scores), key=lambda item: float(item[1]), reverse=True)
    return [{"document": document, "rerank_score": float(score)} for document, score in ranked[:top_k]]
