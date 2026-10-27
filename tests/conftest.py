"""Shared fixtures: a deterministic, offline stand-in for the Groq embedder."""
import re
import zlib

import numpy as np
import pytest


class HashingEmbedder:
    """Bag-of-words hashed into 512 dims. Offline + deterministic; good enough to
    exercise the real chunking / Chroma / evaluation code paths without any API."""

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.calls = 0

    def __call__(self, texts):
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in re.findall(r"[a-z0-9$%.]+", t.lower()):
                out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out

    embed = __call__


@pytest.fixture
def embedder():
    return HashingEmbedder()
