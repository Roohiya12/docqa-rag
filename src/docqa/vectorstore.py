"""Chroma-backed vector store: one collection per chunking strategy.

ChromaDB is a lightweight local library (not an ML model). We always pass in
our own pre-computed embeddings so we control which provider is used.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List

import chromadb

from .chunking import Chunk


class VectorStore:
    def __init__(self, persist_dir: str, collection_name: str, embed_fn: Callable):
        self.embed_fn = embed_fn
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    def reset(self) -> None:
        """Drop and recreate the collection so stale chunks never linger."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = self.embed_fn([c.text for c in batch])
            self.collection.add(
                ids=[c.chunk_id for c in batch],
                embeddings=[e.tolist() for e in embeddings],
                documents=[c.text for c in batch],
                metadatas=[
                    {"doc_id": c.doc_id, "strategy": c.strategy, "position": c.position}
                    for c in batch
                ],
            )

    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = self.embed_fn([query_text])
        n = max(1, min(top_k, self.count()))
        res = self.collection.query(query_embeddings=[q[0].tolist()], n_results=n)
        ids = res.get("ids", [[]])[0]
        return [
            {
                "chunk_id": ids[i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            }
            for i in range(len(ids))
        ]

    def count(self) -> int:
        return self.collection.count()
