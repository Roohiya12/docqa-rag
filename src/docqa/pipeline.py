"""End-to-end RAG pipeline: retrieve -> (optional re-rank) -> generate."""
from __future__ import annotations

from typing import Any, Dict, List

from .generator import AnswerGenerator
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(
        self,
        vector_store: VectorStore,
        generator: AnswerGenerator,
        reranker=None,
        top_k_retrieve: int = 10,
        top_k_final: int = 4,
    ):
        self.vector_store = vector_store
        self.generator = generator
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final

    def retrieve(self, question: str) -> List[Dict[str, Any]]:
        k = max(self.top_k_retrieve, self.top_k_final) if self.reranker else self.top_k_final
        hits = self.vector_store.query(question, top_k=k)
        if self.reranker:
            hits = self.reranker.rerank(question, hits, top_k=self.top_k_final)
        return hits

    def answer(self, question: str) -> Dict[str, Any]:
        chunks = self.retrieve(question)
        return {
            "question": question,
            "answer": self.generator.generate(question, chunks),
            "sources": chunks,
        }
