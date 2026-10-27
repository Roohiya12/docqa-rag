"""Lightweight local text vectors for document retrieval."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np


class EmbeddingModel:
    """Create deterministic text vectors without downloading an AI model."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        batch_size: int = 96,
    ):
        self.model = model or "local-hashing"
        self.batch_size = batch_size
        self.dimension = 768
        self._cache: Dict[str, List[float]] = {}

    def _vectorize(self, text: str) -> List[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        words = re.findall(r"[a-z0-9]+", text.lower())

        if not words:
            return vector.tolist()

        features = Counter(
            {"word:" + word: 1 for word in words}
        )

        for first, second in zip(words, words[1:]):
            features["pair:" + first + "_" + second] += 1

        for feature, count in features.items():
            digest = hashlib.blake2b(
                feature.encode("utf-8"), digest_size=16
            ).digest()

            index = int.from_bytes(digest[:8], "little") % self.dimension
            sign = 1.0 if digest[8] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + np.log(count))

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm

        return vector.tolist()

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        for text in dict.fromkeys(texts):
            if text not in self._cache:
                self._cache[text] = self._vectorize(text)

        return np.asarray(
            [self._cache[text] for text in texts],
            dtype=np.float32,
        )

    def __call__(self, texts: List[str]) -> np.ndarray:
        return self.embed(texts)