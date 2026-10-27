"""Central configuration, read from environment variables (.env supported)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths -------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = BASE_DIR / "indexes"
RESULTS_DIR = BASE_DIR / "results"

# --- Groq (OpenAI-compatible API) ---------------------------------------
GROQ_API_KEY =GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    try:
        import streamlit as st
        GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    except Exception:
        GROQ_API_KEY = ""
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

# Groq's catalog changes over time. Run `python scripts/list_groq_models.py`
# to see what your account can use, then override these in .env if needed.
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
GROQ_EMBED_MODEL = os.getenv("GROQ_EMBED_MODEL", "nomic-embed-text-v1.5")

# --- Retrieval / generation knobs ---------------------------------------
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", 10))  # candidates before re-ranking
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", 4))         # passages sent to the LLM
PERSIST_DIR = str(INDEX_DIR)
