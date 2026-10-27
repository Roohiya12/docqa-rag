"""Answer generation (and re-rank scoring) via Groq's hosted chat models.

Groq's API is OpenAI-compatible, so we use the standard `openai` package with
a custom base_url. Swapping providers later is a one-line config change.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from . import config

SYSTEM_PROMPT = """You are a careful document question-answering assistant.
Answer the user's question using ONLY the information in the numbered context
passages. Every factual claim in your answer must be followed by a citation like
[1] or [2] naming the passage it came from. If the passages do not contain
enough information to answer, say so plainly instead of guessing."""


class AnswerGenerator:
    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = model or config.GROQ_CHAT_MODEL
        self._client = OpenAI(
            api_key=api_key or config.GROQ_API_KEY or "missing-key",
            base_url=config.GROQ_BASE_URL,
            max_retries=6,
        )

    @staticmethod
    def format_context(chunks: List[Dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[{i}] (source: {c.get('metadata', {}).get('doc_id', 'unknown')})\n{c['text']}"
            for i, c in enumerate(chunks, start=1)
        )

    def generate(self, question: str, chunks: List[Dict[str, Any]], temperature: float = 0.1) -> str:
        user_msg = f"Context passages:\n\n{self.format_context(chunks)}\n\nQuestion: {question}"
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        return resp.choices[0].message.content or ""

    def score_passages(self, question: str, passages: List[str]) -> Optional[List[float]]:
        """Listwise relevance scoring in ONE call (cheap on a free-tier rate limit).

        Returns one 0-10 score per passage, or None if the reply could not be
        parsed (callers should then keep the original retrieval order).
        """
        numbered = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, start=1))
        prompt = (
            f"Question: {question}\n\n"
            "Rate how directly each passage below helps answer the question, from 0 "
            "(irrelevant) to 10 (contains the answer).\n\n"
            f"Passages:\n{numbered}\n\n"
            f"Reply with ONLY a JSON array of exactly {len(passages)} numbers, in order. "
            "Example for 3 passages: [2, 9, 0]"
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0.0,
            max_tokens=1024,  # reasoning models spend tokens thinking before they answer
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or ""
        match = re.search(r"\[[^\[\]]*\]", raw)
        if not match:
            return None
        try:
            scores = [float(x) for x in json.loads(match.group(0))]
        except (ValueError, TypeError):
            return None
        return scores if len(scores) == len(passages) else None
