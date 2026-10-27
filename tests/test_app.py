"""Headless smoke tests for the Streamlit app (no browser, no network, no API key)."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from docqa import config  # noqa: E402

APP = str(config.BASE_DIR / "app" / "streamlit_app.py")


@pytest.fixture
def app(tmp_path, embedder, monkeypatch):
    import streamlit as st

    st.cache_resource.clear()
    monkeypatch.setattr(config, "PERSIST_DIR", str(tmp_path / "idx"))
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    monkeypatch.setattr("docqa.embeddings.EmbeddingModel", lambda: embedder)
    monkeypatch.setattr(
        "docqa.generator.AnswerGenerator.generate",
        lambda self, q, chunks, temperature=0.1: f"stub answer [1] ({len(chunks)} passages)",
    )
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_MAX_QUESTIONS", "2")
    return AppTest.from_file(APP, default_timeout=60)


def _ask(at, question):
    at.text_input[0].set_value(question)
    at.button[0].click()  # the form's submit button
    return at.run()


def test_demo_mode_builds_index_and_answers_with_sources(app):
    at = app.run()
    assert not at.exception, at.exception
    assert not any("Rebuild" in b.label for b in at.button), "demo mode must hide rebuild/upload"
    at = _ask(at, "What was Solstice's total revenue in fiscal year 2025?")
    assert not at.exception, at.exception
    assert any("stub answer [1]" in m.value for m in at.markdown)
    assert len(at.expander) >= 1, "sources should be shown"
    assert any("$482 million" in m.value for m in at.markdown), "retrieved passage should contain the figure"


def test_demo_mode_caps_questions_per_session(app):
    at = app.run()
    for _ in range(2):
        at = _ask(at, "What was Solstice's total revenue?")
    at = _ask(at, "One more question")
    assert any("Demo limit reached" in w.value for w in at.warning)


def test_missing_api_key_shows_error_instead_of_crashing(app, monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    at = app.run()
    assert any("GROQ_API_KEY" in e.value for e in at.error)
