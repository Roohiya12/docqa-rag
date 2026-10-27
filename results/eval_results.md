# Retrieval evaluation results

| Chunking strategy | LLM re-rank | Hit Rate@1 | Hit Rate@3 | Hit Rate@5 | MRR |
|---|---|---|---|---|---|
| fixed_no_overlap | no | 41.7% | 58.3% | 64.6% | 0.514 |
| fixed_overlap | no | 62.5% | 77.1% | 81.2% | 0.694 |
| semantic | no | 50.0% | 68.8% | 77.1% | 0.608 |

*48 questions over 3 documents (13,100 chars); embedding model `nomic-embed-text-v1.5`; retrieved top-k candidates per question; measured 2026-10-27.*

- `fixed_no_overlap`: 28 chunks
- `fixed_overlap`: 33 chunks
- `semantic`: 24 chunks
