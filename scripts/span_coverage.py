"""Chunking-quality check that needs NO API calls.

For each strategy: what share of the ground-truth answer spans survive WHOLE inside a single
chunk? That is the ceiling on Hit Rate@k, whatever the embedding model does. The gap between
this ceiling and your measured Hit Rate@5 is the part lost to retrieval ranking.

    python scripts/span_coverage.py

Reads the chunks already stored in your Chroma indexes (run_eval.py --build creates them).
"""
import _bootstrap  # noqa: F401
import chromadb

from docqa import config
from docqa.chunking import STRATEGY_FUNCS
from docqa.evaluate import is_hit, load_eval_dataset


def span_coverage(persist_dir, dataset):
    client = chromadb.PersistentClient(path=str(persist_dir))
    out = {}
    for strategy in STRATEGY_FUNCS:
        try:
            col = client.get_collection(strategy)
        except Exception:
            continue
        texts = col.get(include=["documents"])["documents"]
        split = [q["id"] for q in dataset if not any(is_hit(t, q["source_snippet"]) for t in texts)]
        out[strategy] = {
            "chunks": len(texts),
            "avg_chars": round(sum(map(len, texts)) / len(texts), 1),
            "kept_whole": len(dataset) - len(split),
            "split_ids": split,
        }
    return out


def main() -> None:
    data = load_eval_dataset(str(config.EVAL_DIR / "eval_dataset.json"))
    res = span_coverage(config.PERSIST_DIR, data)
    if not res:
        raise SystemExit("No indexes found. Run: python scripts/run_eval.py --build")
    n = len(data)
    print(f"{'strategy':<18}{'chunks':>7}{'avg chars':>11}{'spans kept whole':>20}")
    for s, r in res.items():
        print(f"{s:<18}{r['chunks']:>7}{r['avg_chars']:>11}{r['kept_whole']:>10}/{n} = {r['kept_whole']/n:.1%}")
    for s, r in res.items():
        if r["split_ids"]:
            print(f"\n{s}: answer cut across chunks for {', '.join(r['split_ids'])}")


if __name__ == "__main__":
    main()
