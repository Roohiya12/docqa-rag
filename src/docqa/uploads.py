"""Session-private document uploads: parsed, chunked and embedded IN MEMORY.

Nothing is written to disk, so on a shared server (e.g. Streamlit Community Cloud)
one visitor's files can never be seen by another visitor, and nothing survives a restart.

InMemoryStore has the same query()/count() interface as VectorStore, so RAGPipeline
works with either.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from .chunking import Chunk, chunk_document

ALLOWED_SUFFIXES = (".txt", ".md", ".pdf")
UPLOAD_STRATEGY = "fixed_overlap"  # best performer in our evaluation


def extract_text(filename: str, data: bytes) -> str:
    """Return the readable text of an uploaded file, or raise ValueError with a friendly message."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".txt", ".md"):
        text = data.decode("utf-8", errors="ignore")
    elif suffix == ".pdf":
        from pypdf import PdfReader

        try:
            reader = PdfReader(io.BytesIO(data))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:  # corrupt / encrypted PDF
            raise ValueError(f"'{filename}' could not be read as a PDF ({type(e).__name__}).") from e
    else:
        raise ValueError(f"'{filename}': only .txt, .md and .pdf files are supported.")
    if not text.strip():
        raise ValueError(f"'{filename}' has no readable text (scanned PDFs are not supported).")
    return text


class InMemoryStore:
    def __init__(self, embed_fn: Callable[[List[str]], np.ndarray]):
        self.embed_fn = embed_fn
        self._chunks: List[Chunk] = []
        self._vecs = np.zeros((0, 0), dtype=np.float32)

    @staticmethod
    def _unit(vecs: np.ndarray) -> np.ndarray:
        vecs = np.asarray(vecs, dtype=np.float32)
        norm = np.linalg.norm(vecs, axis=1, keepdims=True)
        norm[norm == 0] = 1e-8
        return vecs / norm

    def add_chunks(self, chunks: Sequence[Chunk], batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = list(chunks[i : i + batch_size])
            vecs = self._unit(self.embed_fn([c.text for c in batch]))
            self._vecs = vecs if self._vecs.size == 0 else np.vstack([self._vecs, vecs])
            self._chunks.extend(batch)

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self._chunks:
            return []
        q = self._unit(self.embed_fn([query_text]))[0]
        sims = self._vecs @ q
        order = np.argsort(-sims)[:top_k]
        return [
            {
                "chunk_id": self._chunks[i].chunk_id,
                "text": self._chunks[i].text,
                "metadata": {"doc_id": self._chunks[i].doc_id, "strategy": self._chunks[i].strategy,
                             "position": self._chunks[i].position},
                "distance": float(1.0 - sims[i]),  # same meaning as Chroma's cosine distance
            }
            for i in order
        ]

    def count(self) -> int:
        return len(self._chunks)


def build_session_store(
    files: Sequence[Tuple[str, bytes]],
    embed_fn: Callable[[List[str]], np.ndarray],
    max_chars: int = 150_000,
    max_files: int = 5,
) -> Tuple[InMemoryStore, Dict[str, int]]:
    """Parse, chunk and embed uploaded files. Raises ValueError with a user-facing message."""
    if not files:
        raise ValueError("No files to index.")
    if len(files) > max_files:
        raise ValueError(f"Please upload at most {max_files} files at a time.")

    texts = [(Path(name).stem, extract_text(name, data)) for name, data in files]
    total = sum(len(t) for _, t in texts)
    if total > max_chars:
        raise ValueError(f"Your files hold {total:,} characters; this app's limit is {max_chars:,}. Try fewer or shorter files.")

    store = InMemoryStore(embed_fn)
    for doc_id, text in texts:
        store.add_chunks(chunk_document(text, doc_id, UPLOAD_STRATEGY))
    return store, {"documents": len(texts), "chunks": store.count(), "chars": total}
