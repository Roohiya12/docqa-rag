"""Streamlit demo: ask grounded questions, compare chunking strategies, add your own documents.

Deploying publicly (e.g. Streamlit Community Cloud)? Put this in the app's Secrets:
    GROQ_API_KEY = "your key"
    DEMO_MODE = "1"
Demo mode (a) disables uploads/rebuild, because the server's disk is shared by every visitor,
and (b) caps questions per visitor so nobody can burn your free API quota.
"""
import os
import sys
from pathlib import Path

# ChromaDB needs sqlite >= 3.35. Some hosts ship an older one, so prefer the bundled pysqlite3.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Streamlit "Secrets" -> environment variables, before docqa.config reads them.
try:
    for _key in ("GROQ_API_KEY", "GROQ_CHAT_MODEL", "GROQ_EMBED_MODEL", "DEMO_MODE", "DEMO_MAX_QUESTIONS"):
        if _key in st.secrets:
            os.environ.setdefault(_key, str(st.secrets[_key]))
except Exception:  # no secrets file: normal when running locally with a .env
    pass

from docqa import config  # noqa: E402
from docqa.chunking import STRATEGY_FUNCS  # noqa: E402
from docqa.embeddings import EmbeddingModel  # noqa: E402
from docqa.generator import AnswerGenerator  # noqa: E402
from docqa.indexing import build_index  # noqa: E402
from docqa.pipeline import RAGPipeline  # noqa: E402
from docqa.reranker import LLMReranker  # noqa: E402
from docqa.vectorstore import VectorStore  # noqa: E402

DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
MAX_QUESTIONS = int(os.getenv("DEMO_MAX_QUESTIONS", "15"))

st.set_page_config(page_title="DocQA - RAG Document Assistant", page_icon="📄", layout="wide")


@st.cache_resource
def get_embedder() -> EmbeddingModel:
    return EmbeddingModel()


@st.cache_resource(show_spinner="Preparing the document index (first load only)...")
def ensure_index(strategy: str) -> int:
    """Build the index once per server process if it doesn't exist (hosted disks are ephemeral)."""
    embedder = get_embedder()
    if VectorStore(config.PERSIST_DIR, strategy, embedder).count() == 0:
        build_index(strategy, embedder, config.RAW_DIR, config.PERSIST_DIR)
    return VectorStore(config.PERSIST_DIR, strategy, embedder).count()


st.title("📄 DocQA - grounded answers with citations")
st.caption("Retrieval-Augmented Generation: answers use only retrieved passages, and every claim cites its source.")

if not config.GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Locally: add it to `.env`. Hosted: add it under the app's Secrets.")
    st.stop()

if "asked" not in st.session_state:
    st.session_state.asked = 0

with st.sidebar:
    st.header("Settings")
    strategy = st.selectbox("Chunking strategy", list(STRATEGY_FUNCS), index=2)
    use_rerank = st.checkbox("LLM re-rank candidates", value=False, help="Costs about 10 extra API calls per question.")
    top_k = st.slider("Passages sent to the LLM", 1, 8, config.TOP_K_FINAL)

    st.divider()
    if DEMO_MODE:
        st.subheader("Demo mode")
        st.write("Sample documents only: three fictional reports (Solstice Robotics, Harbourline, Northfield Hospital).")
        st.caption(f"Questions left in this session: {max(MAX_QUESTIONS - st.session_state.asked, 0)}")
        with st.expander("Example questions"):
            st.code("What was Solstice's total revenue in fiscal year 2025?")
            st.code("What share of Harbourline's revenue comes from its ten largest customers?")
            st.code("Who is the CEO of Google?  (not in the documents: it should decline)")
    else:
        st.subheader("Your documents")
        uploads = st.file_uploader("Add .txt / .md / .pdf", type=["txt", "md", "pdf"], accept_multiple_files=True)
        if uploads:
            config.RAW_DIR.mkdir(parents=True, exist_ok=True)
            for f in uploads:
                (config.RAW_DIR / Path(f.name).name).write_bytes(f.getbuffer())
            st.success(f"Saved {len(uploads)} file(s). Rebuild the index below to include them.")
        if st.button(f"Rebuild '{strategy}' index"):
            with st.spinner("Chunking and embedding..."):
                info = build_index(strategy, get_embedder(), config.RAW_DIR, config.PERSIST_DIR)
            ensure_index.clear()
            st.success(f"{info['documents']} docs -> {info['chunks']} chunks")

try:
    n_chunks = ensure_index(strategy)
except Exception as e:  # e.g. invalid key or unknown embedding model name
    st.error(f"Could not build the index: {type(e).__name__}: {e}")
    st.caption("Check GROQ_API_KEY and run `python scripts/list_groq_models.py` to confirm the model IDs.")
    st.stop()
st.sidebar.caption(f"Index '{strategy}': {n_chunks} chunks")

with st.form("ask_form"):
    question = st.text_input("Ask a question about your documents", placeholder="e.g. What was Solstice's total revenue in fiscal 2025?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    if DEMO_MODE and st.session_state.asked >= MAX_QUESTIONS:
        st.warning(f"Demo limit reached ({MAX_QUESTIONS} questions per session). Clone the repo to run it without limits.")
    else:
        try:
            generator = AnswerGenerator()
            pipeline = RAGPipeline(
                VectorStore(config.PERSIST_DIR, strategy, get_embedder()), generator,
                reranker=LLMReranker(generator) if use_rerank else None,
                top_k_retrieve=config.TOP_K_RETRIEVE, top_k_final=top_k,
            )
            with st.spinner("Retrieving and generating..."):
                st.session_state.last = pipeline.answer(question.strip())
            st.session_state.asked += 1
        except Exception as e:
            st.error(f"Request failed: {type(e).__name__}: {e}")

result = st.session_state.get("last")
if result:
    st.subheader("Answer")
    st.write(result["answer"])
    st.subheader("Sources")
    for i, s in enumerate(result["sources"], 1):
        score = f"re-rank {s['rerank_score']:.0f}/10" if "rerank_score" in s else f"distance {s['distance']:.3f}"
        with st.expander(f"[{i}] {s['metadata']['doc_id']} - {score}"):
            st.write(s["text"])
