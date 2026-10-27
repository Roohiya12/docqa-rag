"""End-to-end offline test: real chunking + real ChromaDB + real evaluation,
using the hashing stand-in embedder instead of the Groq API."""
import pytest

from docqa import config
from docqa.chunking import STRATEGY_FUNCS
from docqa.evaluate import evaluate_retrieval, load_eval_dataset
from docqa.indexing import build_index
from docqa.pipeline import RAGPipeline
from docqa.vectorstore import VectorStore


@pytest.fixture
def built(tmp_path, embedder):
    persist = str(tmp_path / "idx")
    stats = {s: build_index(s, embedder, config.RAW_DIR, persist) for s in STRATEGY_FUNCS}
    return persist, stats


def test_all_strategies_index_the_corpus(built):
    _, stats = built
    for s, info in stats.items():
        assert info["documents"] == 3 and info["chunks"] > 10, s
    # overlap must yield more chunks than the same windows without it
    assert stats["fixed_overlap"]["chunks"] > stats["fixed_no_overlap"]["chunks"]


def test_rebuilding_an_index_does_not_duplicate_chunks(built, embedder):
    persist, stats = built
    again = build_index("fixed_no_overlap", embedder, config.RAW_DIR, persist)
    assert again["chunks"] == stats["fixed_no_overlap"]["chunks"]
    assert VectorStore(persist, "fixed_no_overlap", embedder).count() == again["chunks"]


def test_retrieval_eval_runs_and_is_far_above_chance(built, embedder):
    persist, _ = built
    data = load_eval_dataset(str(config.EVAL_DIR / "eval_dataset.json"))
    for s in STRATEGY_FUNCS:
        store = VectorStore(persist, s, embedder)
        r = evaluate_retrieval(store, data, (1, 3, 5), label=s)
        chance = 5 / store.count()
        assert r.hit_rate_at_k[5] > 2 * chance, (s, r.hit_rate_at_k, chance)
        assert r.hit_rate_at_k[1] <= r.hit_rate_at_k[3] <= r.hit_rate_at_k[5]


class StubGenerator:
    def generate(self, question, chunks):
        return f"stub answer citing [1] from {len(chunks)} passages"


def test_pipeline_returns_answer_and_sources(built, embedder):
    persist, _ = built
    store = VectorStore(persist, "semantic", embedder)
    out = RAGPipeline(store, StubGenerator(), top_k_final=3).answer("What was Solstice's total revenue?")
    assert "[1]" in out["answer"] and len(out["sources"]) == 3
    assert any("$482 million" in s["text"] for s in out["sources"])
