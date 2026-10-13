# RESULTS

Every number in this file comes from a run we actually executed. No estimates, no placeholders.

## M1 — Naive dense RAG baseline
_Status: not run yet._

- Corpus: 20 arXiv transformer papers (see `data/papers_manifest.json`)
- Eval set: 25 questions (see `eval/eval_set.jsonl`)
- Config: chunk size —, overlap —, embedding model —, top-k —
- Metrics: _to be recorded after the first eval run._

## M2 — Hybrid retrieval + reranking
_Status: not started._

- recall@5 (dense-only baseline): —
- recall@5 (hybrid + rerank): —
- delta: —

## M3 — LangGraph agent layer
_Status: not started._

## M4 — Citations + Ragas evaluation
_Status: not started._

- faithfulness: —
- context precision: —
- answer relevancy: —

## M5 — Chainlit UI + Docker + deploy
_Status: not started._
