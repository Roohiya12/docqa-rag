"""Ask one question against a built index.

    python scripts/ask.py "What was Solstice's revenue in fiscal 2025?"
    python scripts/ask.py "Who chairs Harbourline's board?" --strategy fixed_overlap --rerank
"""
import argparse

import _bootstrap  # noqa: F401
from docqa import config
from docqa.chunking import STRATEGY_FUNCS
from docqa.embeddings import EmbeddingModel
from docqa.generator import AnswerGenerator
from docqa.pipeline import RAGPipeline
from docqa.reranker import LLMReranker
from docqa.vectorstore import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--strategy", default="semantic", choices=list(STRATEGY_FUNCS))
    parser.add_argument("--rerank", action="store_true", help="LLM re-rank retrieved candidates")
    parser.add_argument("--top-k", type=int, default=config.TOP_K_FINAL)
    args = parser.parse_args()

    if not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    store = VectorStore(config.PERSIST_DIR, args.strategy, EmbeddingModel())
    if store.count() == 0:
        raise SystemExit(f"Index '{args.strategy}' is empty. Run: python scripts/build_index.py --strategy {args.strategy}")

    generator = AnswerGenerator()
    pipeline = RAGPipeline(
        store,
        generator,
        reranker=LLMReranker(generator) if args.rerank else None,
        top_k_retrieve=config.TOP_K_RETRIEVE,
        top_k_final=args.top_k,
    )
    result = pipeline.answer(args.question)

    print("\n=== ANSWER ===\n" + result["answer"])
    print("\n=== SOURCES ===")
    for i, s in enumerate(result["sources"], 1):
        print(f"[{i}] ({s['metadata']['doc_id']}) {s['text'][:140]}...")


if __name__ == "__main__":
    main()
