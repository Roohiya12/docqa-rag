"""docqa: a small, dependency-light Retrieval-Augmented Generation (RAG) library.

Everything that needs a model (embeddings, generation, re-ranking) calls Groq's
free-tier, OpenAI-compatible cloud API, so nothing heavy runs on the local
machine. The only local component is the ChromaDB vector store.
"""
