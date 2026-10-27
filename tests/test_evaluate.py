from docqa.evaluate import evaluate_retrieval, is_hit, results_to_markdown_table


class FakeStore:
    """Returns a fixed ranking so metric maths can be checked by hand."""
    collection_name = "fake"

    def __init__(self, rankings):
        self.rankings = rankings

    def query(self, q, top_k=5):
        return [{"text": t} for t in self.rankings[q]][:top_k]


class ReverseReranker:
    def rerank(self, q, hits, top_k=None):
        out = list(reversed(hits))
        return out[:top_k] if top_k else out


def test_is_hit_is_case_and_whitespace_insensitive():
    assert is_hit("The  Revenue\nwas $5 million today.", "revenue was $5 million")
    assert not is_hit("Nothing relevant here", "revenue was $5 million")


def test_hit_rate_and_mrr_computed_by_hand():
    data = [
        {"id": "a", "question": "qa", "source_snippet": "ANSWER A"},   # found at rank 1
        {"id": "b", "question": "qb", "source_snippet": "ANSWER B"},   # found at rank 3
        {"id": "c", "question": "qc", "source_snippet": "ANSWER C"},   # never found
    ]
    store = FakeStore({
        "qa": ["has answer a", "x", "y", "z", "w"],
        "qb": ["x", "y", "has answer b", "z", "w"],
        "qc": ["x", "y", "z", "w", "v"],
    })
    r = evaluate_retrieval(store, data, k_values=(1, 3, 5))
    assert r.hit_rate_at_k == {1: 1 / 3, 3: 2 / 3, 5: 2 / 3}
    assert abs(r.mrr - (1 + 1 / 3 + 0) / 3) < 1e-9
    assert [q["hit_rank"] for q in r.per_question] == [1, 3, None]


def test_reranker_is_applied_and_flagged():
    data = [{"id": "a", "question": "qa", "source_snippet": "ANSWER A"}]
    store = FakeStore({"qa": ["x", "y", "z", "has answer a"]})
    base = evaluate_retrieval(store, data, k_values=(1, 4))
    rr = evaluate_retrieval(store, data, k_values=(1, 4), reranker=ReverseReranker(), rerank_candidates=4)
    assert base.hit_rate_at_k[1] == 0.0 and rr.hit_rate_at_k[1] == 1.0
    assert rr.reranked and not base.reranked


def test_markdown_table_format():
    data = [{"id": "a", "question": "qa", "source_snippet": "ANSWER A"}]
    r = evaluate_retrieval(FakeStore({"qa": ["has answer a"]}), data, k_values=(1,), label="demo")
    table = results_to_markdown_table([r], (1,))
    assert "| demo | no | 100.0% | 1.000 |" in table
