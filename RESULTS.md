# RESULTS

Every number in this file comes from a run we actually executed. No estimates, no placeholders.

## M1 — Naive dense RAG baseline
_Status: complete (2026-10-14)._

- Corpus: 20 arXiv transformer papers → 2,175 chunks (see `data/papers_manifest.json`)
- Eval set: 25 questions with verified supporting quotes (see `eval/eval_set.jsonl`)
- Pipeline: PyMuPDF extraction → recursive token splitting (256 tokens, 40 overlap) →
  all-MiniLM-L6-v2 embeddings (normalized, cosine) → ChromaDB → top-5 dense retrieval →
  Gemini (gemini-3.8-flash, temperature 0)
- Retrieval metric: a question "hits" if its gold `supporting_quote` appears
  (whitespace-normalized substring) in a top-5 chunk. Conservative proxy — credits only
  the single chunk holding the exact gold quote.

**recall@5 (dense-only baseline) = 0.280 (7/25)** — independently reproduced.

Failure breakdown by rank of the gold chunk:
- Hit @≤5 (7): ids 1, 2, 8, 9, 18, 23, 24
- Near-miss, gold at rank 6–30 (10): ids 3, 6, 10, 12, 14, 15, 16, 17, 22, 25 → reranking headroom
- Deep miss, gold not in top 30 (8): ids 4, 5, 7, 11, 13, 19, 20, 21 — all keyword-distinctive
  (GPT-3, DistilBERT, BART, LLaMA, FlashAttention, XLNet, ELECTRA, ALBERT) → hybrid/BM25 target

M2 goal: raise recall@5 via hybrid retrieval (BM25 + dense, RRF fusion) + cross-encoder reranking.

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
