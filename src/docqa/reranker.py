"""LLM-based re-ranker (no second local model needed).

Retrieval by embedding similarity is fast but coarse. This re-ranker asks the
Groq chat model to score all retrieved candidates against the question in one
call, then sorts by that score. If scoring fails for any reason, the original
vector-search order is kept - so it can never make a query fail outright.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from .generator import AnswerGenerator


class LLMReranker:
    def __init__(self, generator: Optional[AnswerGenerator] = None):
        self.generator = generator or AnswerGenerator()

    def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not hits:
            return hits
        try:
            scores = self.generator.score_passages(query, [h["text"] for h in hits])
        except Exception:
            scores = None
        if scores is None:
            return hits[:top_k] if top_k else hits

        for h, s in zip(hits, scores):
            h["rerank_score"] = s
        # sorted() is stable, so ties keep their original vector-search order
        ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
        return ranked[:top_k] if top_k else ranked
