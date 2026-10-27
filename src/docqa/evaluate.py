"""Retrieval evaluation harness.

For every labelled question we retrieve the top-k chunks and check whether any
of them contains the ground-truth answer span. Metrics:

  * Hit Rate@k - fraction of questions where a top-k chunk contains the answer.
  * MRR        - mean reciprocal rank of the first correct chunk (rewards
                 ranking the right chunk higher, not just somewhere in top-k).

Ground truth is a short answer-bearing span copied verbatim from the source
document, so a chunk "hits" if it contains that span (case/whitespace-insensitive).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


def _normalize(s: str) -> str:
    return " ".join(s.lower().split())


def is_hit(chunk_text: str, ground_truth_span: str) -> bool:
    """True if the chunk fully contains the ground-truth answer span."""
    return _normalize(ground_truth_span) in _normalize(chunk_text)


@dataclass
class EvalResult:
    label: str
    reranked: bool
    hit_rate_at_k: Dict[int, float]
    mrr: float
    n_questions: int
    per_question: List[Dict[str, Any]]


def load_eval_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(
    vector_store,
    eval_dataset: List[Dict[str, Any]],
    k_values: Sequence[int] = (1, 3, 5),
    reranker=None,
    rerank_candidates: int = 10,
    label: Optional[str] = None,
) -> EvalResult:
    max_k = max(k_values)
    hits_at_k = {k: 0 for k in k_values}
    reciprocal_ranks: List[float] = []
    per_question: List[Dict[str, Any]] = []

    for item in eval_dataset:
        question, span = item["question"], item["source_snippet"]
        fetch_k = max(rerank_candidates, max_k) if reranker else max_k
        hits = vector_store.query(question, top_k=fetch_k)
        if reranker:
            hits = reranker.rerank(question, hits, top_k=max_k)

        rank = next((r for r, h in enumerate(hits[:max_k], 1) if is_hit(h["text"], span)), None)
        for k in k_values:
            if rank is not None and rank <= k:
                hits_at_k[k] += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        per_question.append({"id": item.get("id"), "question": question, "hit_rank": rank})

    n = len(eval_dataset)
    return EvalResult(
        label=label or vector_store.collection_name,
        reranked=reranker is not None,
        hit_rate_at_k={k: hits_at_k[k] / n for k in k_values},
        mrr=sum(reciprocal_ranks) / n,
        n_questions=n,
        per_question=per_question,
    )


def results_to_markdown_table(results: List[EvalResult], k_values: Sequence[int]) -> str:
    head = "| Chunking strategy | LLM re-rank | " + " | ".join(f"Hit Rate@{k}" for k in k_values) + " | MRR |"
    sep = "|" + "---|" * (len(k_values) + 3)
    rows = [head, sep]
    for r in results:
        cells = [r.label, "yes" if r.reranked else "no"]
        cells += [f"{r.hit_rate_at_k[k] * 100:.1f}%" for k in k_values]
        cells.append(f"{r.mrr:.3f}")
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
