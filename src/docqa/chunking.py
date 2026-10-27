"""Chunking strategies, compared head-to-head by the evaluation harness.

1. fixed_no_overlap - fixed-size character windows, no overlap (naive baseline)
2. fixed_overlap    - same windows, but consecutive chunks share a region so a
                      fact sitting on a boundary still appears whole somewhere
3. semantic         - sentences are grouped by embedding similarity, so chunk
                      boundaries fall at topic shifts instead of arbitrary offsets

Each strategy is a pure function: text in, list[Chunk] out. Easy to unit test.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    strategy: str
    position: int
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')


def split_sentences(text: str) -> List[str]:
    """Lightweight, dependency-free sentence splitter (no NLTK/spaCy download)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def fixed_size_chunks(
    text: str,
    doc_id: str,
    chunk_size: int = 800,
    overlap: int = 0,
    strategy_name: str = "fixed_no_overlap",
) -> List[Chunk]:
    """Slide a fixed-size character window across the text."""
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    text = text.strip()
    chunks: List[Chunk] = []
    start, position, step = 0, 0, chunk_size - overlap
    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}__{strategy_name}__{position}",
                    doc_id=doc_id,
                    text=piece,
                    strategy=strategy_name,
                    position=position,
                    char_start=start,
                    char_end=end,
                )
            )
            position += 1
        if end == len(text):
            break
        start += step
    return chunks


def semantic_chunks(
    text: str,
    doc_id: str,
    embed_fn: Callable[[List[str]], np.ndarray],
    max_chunk_chars: int = 1200,
    breakpoint_percentile: float = 85.0,
    strategy_name: str = "semantic",
) -> List[Chunk]:
    """Group sentences into chunks at semantic breakpoints.

    1. Split into sentences and embed each one.
    2. Cosine distance between each consecutive pair of sentences.
    3. Distances above the given percentile mark a topic shift -> new chunk.
    4. Also cut when a chunk would exceed max_chunk_chars.

    embed_fn is injected, so tests can pass a cheap deterministic stand-in.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return fixed_size_chunks(text, doc_id, chunk_size=max_chunk_chars, strategy_name=strategy_name)

    emb = np.asarray(embed_fn(sentences), dtype=np.float32)
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    norm[norm == 0] = 1e-8
    unit = emb / norm
    distances = 1.0 - np.sum(unit[:-1] * unit[1:], axis=1)
    threshold = np.percentile(distances, breakpoint_percentile)
    breakpoints = {int(i) for i in np.where(distances > threshold)[0]}

    chunks: List[Chunk] = []
    position = 0
    running_start = 0
    current = [sentences[0]]
    current_len = len(sentences[0])

    def flush(sent_list: List[str], start_char: int) -> Chunk:
        nonlocal position
        body = " ".join(sent_list).strip()
        c = Chunk(
            chunk_id=f"{doc_id}__{strategy_name}__{position}",
            doc_id=doc_id,
            text=body,
            strategy=strategy_name,
            position=position,
            char_start=start_char,
            char_end=start_char + len(body),
        )
        position += 1
        return c

    for i in range(1, len(sentences)):
        cut = (i - 1) in breakpoints or (current_len + len(sentences[i]) > max_chunk_chars)
        if cut:
            chunks.append(flush(current, running_start))
            running_start += current_len + 1
            current, current_len = [sentences[i]], len(sentences[i])
        else:
            current.append(sentences[i])
            current_len += len(sentences[i]) + 1
    if current:
        chunks.append(flush(current, running_start))
    return chunks


# Tunable defaults. 500 chars (~100 words) suits short factual documents; raise
# CHUNK_SIZE / SEMANTIC_MAX_CHARS for long-form prose such as research papers.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100          # 20% overlap
SEMANTIC_MAX_CHARS = 700     # ceiling so a uniform section can't become one huge chunk
SEMANTIC_PERCENTILE = 85.0   # higher = fewer, larger semantic chunks

STRATEGY_FUNCS = {
    "fixed_no_overlap": lambda text, doc_id, embed_fn: fixed_size_chunks(
        text, doc_id, chunk_size=CHUNK_SIZE, overlap=0, strategy_name="fixed_no_overlap"
    ),
    "fixed_overlap": lambda text, doc_id, embed_fn: fixed_size_chunks(
        text, doc_id, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP, strategy_name="fixed_overlap"
    ),
    "semantic": lambda text, doc_id, embed_fn: semantic_chunks(
        text,
        doc_id,
        embed_fn=embed_fn,
        max_chunk_chars=SEMANTIC_MAX_CHARS,
        breakpoint_percentile=SEMANTIC_PERCENTILE,
        strategy_name="semantic",
    ),
}


def chunk_document(
    text: str,
    doc_id: str,
    strategy: str,
    embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None,
) -> List[Chunk]:
    if strategy not in STRATEGY_FUNCS:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {list(STRATEGY_FUNCS)}")
    if strategy == "semantic" and embed_fn is None:
        raise ValueError("The 'semantic' strategy requires an embed_fn.")
    return STRATEGY_FUNCS[strategy](text, doc_id, embed_fn)
