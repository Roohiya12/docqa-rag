"""Run the retrieval evaluation across chunking strategies (and optionally re-ranking).

    python scripts/run_eval.py --build                 # (re)build indexes, then evaluate
    python scripts/run_eval.py                         # evaluate existing indexes
    python scripts/run_eval.py --rerank                # also measure the LLM re-ranker
    python scripts/run_eval.py --rerank --update-readme

Outputs results/eval_results.md and results/eval_results.json. With --update-readme
the table is injected into README.md between the RESULTS markers, so the numbers
in your repo are always ones you actually measured.
"""
import argparse
import json
import re
from datetime import date

import _bootstrap  # noqa: F401
from docqa import config
from docqa.chunking import STRATEGY_FUNCS
from docqa.embeddings import EmbeddingModel
from docqa.evaluate import evaluate_retrieval, load_eval_dataset, results_to_markdown_table
from docqa.indexing import build_index
from docqa.ingest import load_documents
from docqa.reranker import LLMReranker
from docqa.vectorstore import VectorStore

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGY_FUNCS), choices=list(STRATEGY_FUNCS))
    ap.add_argument("--k", nargs="+", type=int, default=[1, 3, 5])
    ap.add_argument("--build", action="store_true", help="rebuild indexes before evaluating")
    ap.add_argument("--rerank", action="store_true", help="also evaluate with the LLM re-ranker")
    ap.add_argument("--rerank-candidates", type=int, default=10)
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()

    if not config.GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

    dataset = load_eval_dataset(str(config.EVAL_DIR / "eval_dataset.json"))
    docs = load_documents(config.RAW_DIR)
    corpus_chars = sum(len(t) for _, t in docs)
    print(f"{len(dataset)} questions | {len(docs)} documents | {corpus_chars:,} characters")

    embedder = EmbeddingModel()
    embedder.embed([d["question"] for d in dataset])  # warm the cache: 1 API call instead of 48+

    stats, results = {}, []
    reranker = LLMReranker() if args.rerank else None

    for strategy in args.strategies:
        if args.build:
            stats[strategy] = build_index(strategy, embedder, config.RAW_DIR, config.PERSIST_DIR)
        store = VectorStore(config.PERSIST_DIR, strategy, embedder)
        if store.count() == 0:
            print(f"! skipping '{strategy}': empty index (use --build)")
            continue
        stats.setdefault(strategy, {"chunks": store.count()})

        base = evaluate_retrieval(store, dataset, args.k, label=strategy)
        results.append(base)
        print(f"{strategy:<18} Hit@{max(args.k)}={base.hit_rate_at_k[max(args.k)]:.1%}  MRR={base.mrr:.3f}")

        if reranker:
            rr = evaluate_retrieval(
                store, dataset, args.k, reranker=reranker,
                rerank_candidates=args.rerank_candidates, label=strategy,
            )
            results.append(rr)
            print(f"{strategy + ' +rerank':<18} Hit@{max(args.k)}={rr.hit_rate_at_k[max(args.k)]:.1%}  MRR={rr.mrr:.3f}")

    if not results:
        raise SystemExit("Nothing evaluated. Run with --build first.")

    table = results_to_markdown_table(results, args.k)
    chunk_lines = "\n".join(
        f"- `{s}`: {v['chunks']} chunks" + (f", avg {v['avg_chunk_chars']} chars" if "avg_chunk_chars" in v else "")
        for s, v in stats.items()
    )
    block = (
        f"{table}\n\n"
        f"*{len(dataset)} questions over {len(docs)} documents ({corpus_chars:,} chars); "
        f"embedding model `{config.GROQ_EMBED_MODEL}`; retrieved top-k candidates per question; "
        f"measured {date.today().isoformat()}.*\n\n{chunk_lines}\n"
    )

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "eval_results.md").write_text("# Retrieval evaluation results\n\n" + block, encoding="utf-8")
    (config.RESULTS_DIR / "eval_results.json").write_text(
        json.dumps(
            [{"strategy": r.label, "reranked": r.reranked, "hit_rate_at_k": r.hit_rate_at_k,
              "mrr": r.mrr, "n_questions": r.n_questions, "per_question": r.per_question} for r in results],
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n" + table + f"\n\nWrote {config.RESULTS_DIR / 'eval_results.md'}")

    if args.update_readme:
        readme = config.BASE_DIR / "README.md"
        text = readme.read_text(encoding="utf-8")
        if START not in text or END not in text:
            raise SystemExit("README.md is missing the RESULTS markers.")
        text = re.sub(f"{re.escape(START)}.*?{re.escape(END)}", f"{START}\n{block}\n{END}", text, flags=re.S)
        readme.write_text(text, encoding="utf-8")
        print("README.md updated.")


if __name__ == "__main__":
    main()
