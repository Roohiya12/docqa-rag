# DocQA: a RAG document Q&A bot, with an evaluation harness

Ask questions about your documents and get answers that **cite the passages they came from**.
The project goes beyond "I built a RAG bot": it **measures retrieval quality** and compares
chunking strategies (fixed-size, fixed-size with overlap, semantic) and an optional LLM re-ranker.

- **Runs on free resources.** Embeddings, generation and re-ranking all use [Groq](https://console.groq.com)'s free tier, so no model runs on your machine. Only ChromaDB, a small local vector database, does.
- **Grounded answers.** The model may only use retrieved passages and must cite them as `[1]`, `[2]`, ...
- **Evaluation harness.** 48 labelled questions, Hit Rate@k and MRR, one command to reproduce.
- **CLI + Streamlit UI.** Upload `.txt`, `.md` or `.pdf` files and chat with them.

## How it works

```
 documents (.txt/.md/.pdf)
        |  ingest
        v
   chunking strategy  --->  embeddings (Groq)  --->  ChromaDB (one collection per strategy)
 (fixed | overlap | semantic)                                   |
                                                                | top-k similar chunks
 question --> embed (Groq) -------------------------------------+
                                                                v
                                         (optional) LLM re-rank (Groq)
                                                                |
                                                                v
                    prompt = rules + numbered passages + question --> LLM (Groq)
                                                                |
                                                                v
                                        answer with [n] citations + source passages
```

## Quickstart

Requires Python 3.10+ and a free Groq API key (no credit card): <https://console.groq.com/keys>

```bash
git clone <your-repo-url> && cd docqa-rag
python -m venv .venv

# activate the environment
.venv\Scripts\activate            # Windows (PowerShell / cmd)
source .venv/bin/activate         # macOS / Linux

pip install -r requirements.txt

cp .env.example .env              # Windows: copy .env.example .env
# open .env and paste your key after GROQ_API_KEY=
```

Check that your account can use the default models (Groq's catalog changes over time):

```bash
python scripts/list_groq_models.py
```

If the defaults are missing, set `GROQ_CHAT_MODEL` / `GROQ_EMBED_MODEL` in `.env`.

## Usage

```bash
python scripts/build_index.py                       # index data/raw/ with all 3 strategies
python scripts/ask.py "What was Solstice's total revenue in fiscal year 2025?"
python scripts/ask.py "Who chairs Harbourline's board?" --strategy fixed_overlap --rerank
streamlit run app/streamlit_app.py                  # web UI
pytest                                              # tests (no API key or network needed)
```

To use your own documents, drop files into `data/raw/` (or upload them in the web UI) and rebuild the index.

## Evaluation

**Dataset** (`data/eval/eval_dataset.json`): 48 questions over 3 fictional documents (13,100 characters). Each question has a reference answer and a `source_snippet`, a short span copied verbatim from the source document. A test checks that every snippet really appears in its document.

**Metrics**

| Metric | What it measures |
|---|---|
| **Hit Rate@k** | Share of questions where at least one of the top-k retrieved chunks contains the `source_snippet`. A value of 80% at k=3 means 80% of questions had the answer text inside one of the 3 best chunks. |
| **MRR** | Mean of 1/rank of the first correct chunk (rank 1 gives 1.0, rank 2 gives 0.5, not found gives 0). It rewards ranking the right chunk **first**, not merely somewhere in the top k. |

Hit Rate checks that the right text **was retrieved**. It does not check that the final answer is correct.

**Strategies compared**

| Strategy | Setting |
|---|---|
| `fixed_no_overlap` | 500-character windows, no overlap (28 chunks on this corpus) |
| `fixed_overlap` | 500-character windows, 100-character overlap (33 chunks) |
| `semantic` | Sentences grouped by embedding similarity; a boundary is cut where the distance between neighbouring sentences exceeds the 85th percentile, capped at 700 characters |
| `+ LLM re-rank` (optional feature, **not evaluated**) followed by the rest of the row unchanged |

**Reproduce**

```bash
python scripts/run_eval.py --build --update-readme
```

`--update-readme` writes the table below, so every number here is one you measured.

### Results

<!-- RESULTS:START -->
| Chunking strategy | LLM re-rank | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR |
|---|---|---|---|---|---|
| fixed_no_overlap | no | 41.7% | 58.3% | 64.6% | 0.514 |
| fixed_overlap | no | 62.5% | 77.1% | 81.2% | 0.694 |
| semantic | no | 50.0% | 68.8% | 77.1% | 0.608 |
**Findings:** 20% overlap performed best (Hit@1 62.5%, MRR 0.694), ahead of semantic chunking (50.0%, 0.608) and plain fixed-size chunking (41.7%, 0.514). Part of the overlap advantage is structural: without overlap, 7 of the 48 answer spans (14.6%) are cut across chunk boundaries.

*48 questions over 3 documents (13,100 chars); embedding model `nomic-embed-text-v1.5`; retrieved top-k candidates per question; measured 2026-10-27.*

- `fixed_no_overlap`: 28 chunks
- `fixed_overlap`: 33 chunks
- `semantic`: 24 chunks

<!-- RESULTS:END -->

### Reading the results honestly

- **- **The corpus is small** (28-33 chunks, 13,100 characters, 48 questions), so each question is worth about 2 points and each percentage carries roughly ±14 points of uncertainty. Treat differences of 1-2 questions as noise. **Hit Rate@1 and MRR are the metrics that separate strategies here.** Differences of one or two questions (about 2-4 points on 48 questions) are within noise.
- To get a stronger comparison, add longer documents to `data/raw/` and write more labelled questions in the same format.
- Matching is by substring: a hit means the chunk contains the whole answer span. A strategy that splits the span across two chunks is penalised, which is intended, because that is exactly the failure that chunking causes.
- The re-ranker is an LLM, so scores can vary slightly between runs.

## Design decisions

- **Groq for everything.** It exposes an OpenAI-compatible API, so the standard `openai` package works with a different `base_url`. Switching provider is a config change.
- **LLM re-ranker instead of a local cross-encoder.** It needs no local model or RAM. The trade-off is `rerank_candidates` extra API calls per question. A cross-encoder can be added behind the same `.rerank()` interface.
- **Embeddings are passed to Chroma explicitly**, so Chroma never silently downloads its own default model.
- **Chunkers take `embed_fn` as an argument**, so they are unit-testable with a fake embedder and no network.
- **Free-tier limits.** The clients retry with backoff on rate limits. Re-ranking is the call-heavy step, so use `--rerank-candidates` to reduce it.

## Limitations

- The sentence splitter is a regex, not spaCy/NLTK, so unusual abbreviations can cause bad splits.
- Scanned PDFs without a text layer are not supported (no OCR).
- Retrieval is dense-only. Hybrid search (BM25 + embeddings) is a natural next step.
- Evaluation measures retrieval, not answer faithfulness.

## Project structure

```
src/docqa/     chunking, embeddings, vectorstore, indexing, reranker, generator, pipeline, evaluate
scripts/       build_index.py, ask.py, run_eval.py, list_groq_models.py
app/           streamlit_app.py
data/raw/      source documents          data/eval/   labelled questions
tests/         pytest suite              results/     generated evaluation output
```

## License

MIT
