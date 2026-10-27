"""Build a vector index for one chunking strategy from a folder of documents."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from .chunking import chunk_document
from .ingest import load_documents
from .vectorstore import VectorStore


def build_index(strategy: str, embed_fn: Callable, raw_dir: Path, persist_dir: str) -> Dict:
    """(Re)build the index for `strategy`. Returns summary statistics."""
    docs = load_documents(raw_dir)
    if not docs:
        raise FileNotFoundError(f"No .txt/.md/.pdf documents found in {raw_dir}")

    store = VectorStore(persist_dir, strategy, embed_fn)
    store.reset()

    chunks = []
    for doc_id, text in docs:
        chunks.extend(chunk_document(text, doc_id, strategy, embed_fn=embed_fn))
    store.add_chunks(chunks)

    sizes = [len(c.text) for c in chunks]
    return {
        "strategy": strategy,
        "documents": len(docs),
        "chunks": len(chunks),
        "avg_chunk_chars": round(sum(sizes) / len(sizes), 1),
        "max_chunk_chars": max(sizes),
    }
