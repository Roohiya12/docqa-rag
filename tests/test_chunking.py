import numpy as np
import pytest

from docqa.chunking import (
    CHUNK_OVERLAP, CHUNK_SIZE, chunk_document, fixed_size_chunks, semantic_chunks, split_sentences,
)

TEXT = "Alpha sentence one. " * 100


def test_fixed_no_overlap_respects_size_and_covers_text():
    chunks = fixed_size_chunks(TEXT, "d", chunk_size=200, overlap=0)
    assert all(len(c.text) <= 200 for c in chunks)
    assert "".join(c.text.replace(" ", "") for c in chunks) == TEXT.strip().replace(" ", "")


def test_overlap_produces_more_chunks_that_share_text():
    plain = fixed_size_chunks(TEXT, "d", chunk_size=200, overlap=0)
    over = fixed_size_chunks(TEXT, "d", chunk_size=200, overlap=50, strategy_name="fixed_overlap")
    assert len(over) > len(plain)
    a, b = over[0], over[1]
    assert a.char_end - b.char_start == 50  # consecutive windows share exactly `overlap` chars


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        fixed_size_chunks(TEXT, "d", chunk_size=100, overlap=100)


def test_chunk_ids_are_unique_and_deterministic():
    a = fixed_size_chunks(TEXT, "d", chunk_size=200)
    b = fixed_size_chunks(TEXT, "d", chunk_size=200)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_sentence_splitter():
    assert split_sentences("One is here. Two is here! Is three here? Yes.") == [
        "One is here.", "Two is here!", "Is three here?", "Yes."
    ]
    assert split_sentences("   ") == []


def _topic_embed(sentences):
    """Two orthogonal 'topics' so we know exactly where the boundary should fall."""
    return np.array([[1.0, 0.0] if "robot" in s else [0.0, 1.0] for s in sentences])


def test_semantic_chunking_cuts_at_the_topic_shift():
    text = (
        "The robot arm lifts boxes. The robot arm also packs boxes. The robot arm is very fast. "
        "The farm grows tomatoes. The farm waters tomatoes daily. The farm sells tomatoes locally."
    )
    chunks = semantic_chunks(text, "d", _topic_embed, max_chunk_chars=10_000, breakpoint_percentile=50)
    assert len(chunks) == 2
    assert "robot" in chunks[0].text and "farm" not in chunks[0].text
    assert "farm" in chunks[1].text and "robot" not in chunks[1].text


def test_semantic_chunking_enforces_max_size():
    text = " ".join(f"The robot unit {i} works well today." for i in range(60))
    chunks = semantic_chunks(text, "d", _topic_embed, max_chunk_chars=300, breakpoint_percentile=99)
    assert all(len(c.text) <= 300 for c in chunks)


def test_chunk_document_validation():
    with pytest.raises(ValueError):
        chunk_document("text", "d", "nonsense")
    with pytest.raises(ValueError):
        chunk_document("text", "d", "semantic", embed_fn=None)
    assert chunk_document(TEXT, "d", "fixed_overlap")[0].strategy == "fixed_overlap"


def test_default_overlap_is_sane():
    assert 0 < CHUNK_OVERLAP < CHUNK_SIZE
