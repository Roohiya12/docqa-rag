import pytest

from docqa.uploads import InMemoryStore, build_session_store, extract_text

CATS = "Whiskers the cat sleeps on the sofa all day. " * 12
RATES = "The central bank raised interest rates to curb inflation this quarter. " * 12


def test_extract_text_txt_md_and_rejections():
    assert "hello" in extract_text("a.txt", b"hello world")
    assert "# T" in extract_text("b.md", "# T\nbody".encode())
    with pytest.raises(ValueError, match="only .txt, .md and .pdf"):
        extract_text("c.exe", b"x")
    with pytest.raises(ValueError, match="no readable text"):
        extract_text("d.txt", b"   \n ")
    with pytest.raises(ValueError, match="could not be read as a PDF"):
        extract_text("e.pdf", b"not really a pdf")


def test_store_returns_the_relevant_document_first(embedder):
    store, info = build_session_store(
        [("cats.txt", CATS.encode()), ("bank.txt", RATES.encode())], embedder)
    assert info["documents"] == 2 and info["chunks"] == store.count() > 2
    hits = store.query("interest rates and inflation", top_k=3)
    assert hits[0]["metadata"]["doc_id"] == "bank"
    assert hits == sorted(hits, key=lambda h: h["distance"])  # nearest first
    assert set(hits[0]) == {"chunk_id", "text", "metadata", "distance"}  # same shape as VectorStore


def test_limits_are_enforced_with_friendly_errors(embedder):
    with pytest.raises(ValueError, match="limit is 100"):
        build_session_store([("a.txt", b"x" * 500)], embedder, max_chars=100)
    with pytest.raises(ValueError, match="at most 2 files"):
        build_session_store([(f"{i}.txt", b"hello there") for i in range(3)], embedder, max_files=2)
    with pytest.raises(ValueError, match="No files"):
        build_session_store([], embedder)


def test_empty_store_query_is_safe(embedder):
    assert InMemoryStore(embedder).query("anything") == []


def test_sessions_are_isolated(embedder):
    a, _ = build_session_store([("a.txt", CATS.encode())], embedder)
    b, _ = build_session_store([("b.txt", RATES.encode())], embedder)
    assert {h["metadata"]["doc_id"] for h in a.query("cats", 5)} == {"a"}
    assert {h["metadata"]["doc_id"] for h in b.query("cats", 5)} == {"b"}
