"""Build (or rebuild) vector indexes from the documents in data/raw/.

    python scripts/build_index.py                      # all three strategies
    python scripts/build_index.py --strategy semantic
"""
import argparse

import _bootstrap  # noqa: F401
from docqa import config
from docqa.chunking import STRATEGY_FUNCS
from docqa.embeddings import EmbeddingModel
from docqa.indexing import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategy", default="all", choices=[*STRATEGY_FUNCS, "all"])
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    embedder = EmbeddingModel()
    for strategy in (STRATEGY_FUNCS if args.strategy == "all" else [args.strategy]):
        stats = build_index(strategy, embedder, config.RAW_DIR, config.PERSIST_DIR)
        print(
            f"[{stats['strategy']}] {stats['documents']} docs -> {stats['chunks']} chunks "
            f"(avg {stats['avg_chunk_chars']} chars, max {stats['max_chunk_chars']})"
        )


if __name__ == "__main__":
    main()
