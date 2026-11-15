"""Chroma persistence and document ingestion helpers."""
from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from functools import lru_cache
from typing import Iterable

import chromadb
from sentence_transformers import SentenceTransformer

from src.chunker import chunk_text

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "documents"
DEFAULT_DB_DIR = ROOT / "chroma_db"
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "knowledge_chunks")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def get_collection():
    db_path = Path(os.getenv("CHROMA_PATH", str(DEFAULT_DB_DIR)))
    client = chromadb.PersistentClient(path=str(db_path))
    return client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


@lru_cache(maxsize=2)
def get_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load each configured embedding model once per process."""
    return SentenceTransformer(model_name)


def _mark_index_changed(collection) -> None:
    marker = Path(os.getenv("CHROMA_PATH", str(DEFAULT_DB_DIR))) / f"{collection.name}.lexical-version"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(uuid.uuid4().hex, encoding="utf-8")


def ingest_paths(paths: Iterable[Path], chunk_size: int = 180, overlap: int = 35) -> int:
    files = [Path(p) for p in paths if Path(p).is_file() and Path(p).suffix.lower() in {".txt", ".md"}]
    if not files:
        return 0
    records = []
    for path in files:
        text = path.read_text(encoding="utf-8").strip()
        for index, chunk in enumerate(chunk_text(text, chunk_size, overlap)):
            digest = hashlib.sha1(f"{path.resolve()}:{index}:{chunk}".encode()).hexdigest()
            records.append((digest, chunk, {"source": path.name, "path": str(path.resolve()), "chunk": index}))
    collection = get_collection()
    # Replace prior chunks for these files so edits and deletions do not leave stale passages.
    for path in files:
        collection.delete(where={"path": str(path.resolve())})
    if records:
        model = get_embedding_model()
        embeddings = model.encode([r[1] for r in records], normalize_embeddings=True, show_progress_bar=len(records) > 20).tolist()
        collection.upsert(ids=[r[0] for r in records], documents=[r[1] for r in records], metadatas=[r[2] for r in records], embeddings=embeddings)
    _mark_index_changed(collection)
    return len(records)


def ingest_directory(directory: Path = DEFAULT_DATA_DIR, chunk_size: int = 180, overlap: int = 35) -> int:
    directory = Path(directory).resolve()
    files = sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in {".txt", ".md"})
    count = ingest_paths(files, chunk_size, overlap)
    collection = get_collection()
    indexed = collection.get(include=["metadatas"])
    current_paths = {str(path.resolve()) for path in files}
    stale_ids = [doc_id for doc_id, metadata in zip(indexed["ids"], indexed.get("metadatas") or [])
                 if (metadata or {}).get("path", "").startswith(str(directory) + os.sep)
                 and (metadata or {}).get("path") not in current_paths]
    if stale_ids:
        collection.delete(ids=stale_ids)
        _mark_index_changed(collection)
    return count
