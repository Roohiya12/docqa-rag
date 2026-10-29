"""Streamlit app: grounded Q&A over sample documents OR your own uploads.

Uploads are private to each browser session (kept in memory, never written to disk), so this is safe
on a shared public server.

Deploying publicly (e.g. Streamlit Community Cloud)? Put this in the app's Secrets:
    GROQ_API_KEY = "your key"
    DEMO_MODE = "1"
Demo mode hides the Rebuild button and caps questions / indexing runs per visitor so nobody can
burn your free API quota.
"""
import os
import re
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
    for _key in ("GROQ_API_KEY", "GROQ_CHAT_MODEL", "GROQ_EMBED_MODEL", "DEMO_MODE",
                 "DEMO_MAX_QUESTIONS", "MAX_UPLOAD_CHARS"):
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
from docqa.uploads import UPLOAD_STRATEGY, build_session_store  # noqa: E402
from docqa.vectorstore import VectorStore  # noqa: E402

DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
MAX_QUESTIONS = int(os.getenv("DEMO_MAX_QUESTIONS", "15"))
MAX_INDEXINGS = 5  # upload-indexing runs per visitor session in demo mode
MAX_UPLOAD_CHARS = int(os.getenv("MAX_UPLOAD_CHARS", "150000"))
SAMPLE, MINE = "Sample documents", "My uploads"

st.set_page_config(page_title="DocQA - RAG Document Assistant", page_icon="📄", layout="wide")


@st.cache_resource
def get_embedder() -> EmbeddingModel:
    return EmbeddingModel()


@st.cache_resource(show_spinner="Preparing the sample-document index (first load only)...")
def ensure_index(strategy: str) -> int:
    """Build the sample index once per server process if it doesn't exist (hosted disks are ephemeral)."""
    embedder = get_embedder()
    if VectorStore(config.PERSIST_DIR, strategy, embedder).count() == 0:
        build_index(strategy, embedder, config.RAW_DIR, config.PERSIST_DIR)
    return VectorStore(config.PERSIST_DIR, strategy, embedder).count()


st.title("📄 DocQA - grounded answers with citations")
st.caption("Retrieval-Augmented Generation: answers use only retrieved passages, every claim cites its source, "
           "and the bot says so when the documents don't contain the answer.")

if not config.GROQ_API_KEY:
    st.error("GROQ_API_KEY is not set. Locally: add it to `.env`. Hosted: add it under the app's Secrets.")
    st.stop()

for _k, _v in (("asked", 0), ("indexed", 0)):
    st.session_state.setdefault(_k, _v)

with st.sidebar:
    st.header("1. Your documents")
    st.caption("Private to this browser session. Nothing is saved on the server.")
    uploads = st.file_uploader("Upload .txt / .md / .pdf", type=["txt", "md", "pdf"], accept_multiple_files=True)
    if st.button("Index my files", disabled=not uploads):
        if DEMO_MODE and st.session_state.indexed >= MAX_INDEXINGS:
            st.warning(f"Demo limit reached ({MAX_INDEXINGS} indexing runs per session).")
        else:
            try:
                with st.spinner("Reading, chunking and embedding..."):
                    store_u, info = build_session_store(
                        [(f.name, f.getvalue()) for f in uploads], get_embedder(), max_chars=MAX_UPLOAD_CHARS)
                st.session_state.update(my_store=store_u, my_info=info, indexed=st.session_state.indexed + 1)
                st.session_state.pop("last", None)
                st.success(f"Indexed {info['documents']} file(s) into {info['chunks']} chunks.")
            except ValueError as e:  # friendly, user-facing problems
                st.error(str(e))
            except Exception as e:
                st.error(f"Indexing failed: {type(e).__name__}: {e}")

    user_store = st.session_state.get("my_store")
    if user_store is not None:
        source = st.radio("Answer from", [MINE, SAMPLE], key="source_choice")
        if st.button("Remove my files"):
            for _k in ("my_store", "my_info", "last"):
                st.session_state.pop(_k, None)
            st.rerun()
    else:
        source = SAMPLE

    st.header("2. Settings")
    if source == SAMPLE:
        strategy = st.selectbox("Chunking strategy", list(STRATEGY_FUNCS), index=2)
    else:
        strategy = UPLOAD_STRATEGY
        st.caption(f"Uploads use `{UPLOAD_STRATEGY}` chunking, the best performer in our evaluation.")
    use_rerank = st.checkbox("LLM re-rank candidates (experimental)", value=False,
                             help="Costs about 10 extra API calls per question. Not validated in our evaluation.")
    top_k = st.slider("Passages sent to the LLM", 1, 8, config.TOP_K_FINAL)

    if DEMO_MODE:
        st.divider()
        st.caption(f"Demo mode. Questions left this session: {max(MAX_QUESTIONS - st.session_state.asked, 0)}")
        with st.expander("Example questions (sample documents)"):
            st.code("What was Solstice's total revenue in fiscal year 2025?")
            st.code("What share of Harbourline's revenue comes from its ten largest customers?")
            st.code("Who is the CEO of Google?  (not in the documents: it should decline)")
    elif source == SAMPLE and st.button(f"Rebuild '{strategy}' sample index"):
        with st.spinner("Chunking and embedding..."):
            info = build_index(strategy, get_embedder(), config.RAW_DIR, config.PERSIST_DIR)
        ensure_index.clear()
        st.success(f"{info['documents']} docs -> {info['chunks']} chunks")

if source == MINE:
    store = user_store
    info = st.session_state.my_info
    st.sidebar.caption(f"Searching your uploads: {info['documents']} file(s), {info['chunks']} chunks")
else:
    try:
        n_chunks = ensure_index(strategy)
    except Exception as e:  # e.g. invalid key or unknown embedding model name
        st.error(f"Could not build the index: {type(e).__name__}: {e}")
        st.caption("Check GROQ_API_KEY and run `python scripts/list_groq_models.py` to confirm the model IDs.")
        st.stop()
    store = VectorStore(config.PERSIST_DIR, strategy, get_embedder())
    st.sidebar.caption(f"Sample index '{strategy}': {n_chunks} chunks")

with st.form("ask_form"):
    question = st.text_input(f"Ask a question about: {source.lower()}",
                             placeholder="e.g. What does the document say about ...?")
    submitted = st.form_submit_button("Ask")

if submitted and question.strip():
    if DEMO_MODE and st.session_state.asked >= MAX_QUESTIONS:
        st.warning(f"Demo limit reached ({MAX_QUESTIONS} questions per session). Clone the repo to run it without limits.")
    else:
        try:
            generator = AnswerGenerator()
            pipeline = RAGPipeline(
                store, generator,
                reranker=LLMReranker(generator) if use_rerank else None,
                top_k_retrieve=config.TOP_K_RETRIEVE, top_k_final=top_k,
            )
            with st.spinner("Retrieving and generating..."):
                st.session_state.last = {**pipeline.answer(question.strip()), "from": source}
            st.session_state.asked += 1
        except Exception as e:
            st.error(f"Request failed: {type(e).__name__}: {e}")

result = st.session_state.get("last")
if result:
    st.subheader("Answer")
    st.write(result["answer"])
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])}
    if not cited:
        st.caption("No [n] citations in this answer. If it states facts, check them against the sources below.")
    elif max(cited) > len(result["sources"]):
        st.caption("This answer cites a source number that does not exist. Check it against the sources below.")
    st.subheader("Sources")
    st.caption(f"Answered from: {result['from']}")
    for i, s in enumerate(result["sources"], 1):
        score = f"re-rank {s['rerank_score']:.0f}/10" if "rerank_score" in s else f"distance {s['distance']:.3f}"
        with st.expander(f"[{i}] {s['metadata']['doc_id']} - {score}"):
            st.write(s["text"])
