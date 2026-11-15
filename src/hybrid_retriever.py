"""Hybrid BM25 and dense retrieval with reciprocal-rank fusion."""
from __future__ import annotations

import re
import os
from pathlib import Path
from rank_bm25 import BM25Okapi

from src.vector_store import DEFAULT_DB_DIR, get_collection, get_embedding_model

TOKEN_RE = re.compile(r"[\w']+", re.UNICODE)
_LEXICAL_CACHE: dict[tuple[str, str], dict] = {}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _load_lexical_index(collection) -> dict:
    """Build BM25 once per collection version; refresh after ingestion changes."""
    key = (os.getenv("CHROMA_PATH", "default"), collection.name)
    version_file = Path(os.getenv("CHROMA_PATH", str(DEFAULT_DB_DIR))) / f"{collection.name}.lexical-version"
    version = version_file.read_text(encoding="utf-8") if version_file.exists() else str(collection.count())
    cached = _LEXICAL_CACHE.get(key)
    if cached and cached["version"] == version:
        return cached
    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    tokenized = [tokenize(document) for document in documents]
    index = {"version": version, "ids": data.get("ids") or [], "documents": documents,
             "metadatas": data.get("metadatas") or [{} for _ in documents],
             "bm25": BM25Okapi(tokenized) if tokenized else None}
    _LEXICAL_CACHE[key] = index
    return index


def retrieve_hybrid(question: str, top_k: int = 5, candidate_k: int = 20, rrf_k: int = 60) -> list[dict]:
    if not question.strip() or top_k <= 0:
        return []
    collection = get_collection()
    data = _load_lexical_index(collection)
    docs = data["documents"]
    ids = data["ids"]
    if not docs:
        return []
    metas = data["metadatas"]
    lexical_scores = data["bm25"].get_scores(tokenize(question))
    lexical_order = sorted(range(len(docs)), key=lambda i: float(lexical_scores[i]), reverse=True)[:candidate_k]
    model = get_embedding_model()
    query_embedding = model.encode(question, normalize_embeddings=True).tolist()
    vector = collection.query(query_embeddings=[query_embedding], n_results=min(candidate_k, len(docs)), include=["distances"])
    id_to_index = {doc_id: i for i, doc_id in enumerate(ids)}
    vector_order = [id_to_index[item] for item in vector["ids"][0] if item in id_to_index]
    vector_distances = dict(zip(vector_order, vector["distances"][0]))
    scores = {}
    for rank, i in enumerate(lexical_order, 1):
        scores[i] = scores.get(i, 0) + 1 / (rrf_k + rank)
    for rank, i in enumerate(vector_order, 1):
        scores[i] = scores.get(i, 0) + 1 / (rrf_k + rank)
    order = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [{"document": docs[i], "document_id": ids[i], "source": (metas[i] or {}).get("source", "unknown"),
             "metadata": metas[i] or {}, "bm25_score": float(lexical_scores[i]),
             "vector_score": float(1.0 - vector_distances[i]) if i in vector_distances else 0.0,
             "hybrid_score": float(scores[i])} for i in order]
