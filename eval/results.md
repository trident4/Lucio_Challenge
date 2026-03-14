# Eval Results

**Date:** 2026-03-11 17:35  
**Score:** 32/33 (97%)  
**Time:** 22.3s (UNDER 30s)  
**Run type:** Cold  

## Config

| Setting | Value |
|---------|-------|
| LLM Model | `openai/gpt-4o-mini` |
| Embedding Model | `openai/text-embedding-3-large` |
| Embedding Provider | `openrouter` |
| Embedding Dimensions | 1024 |
| BM25 Top-K | 50 |
| Rerank Top-K | 8 |
| LLM Max Tokens | 1500 |
| LLM Temperature | 0.0 |

## Results

| Q | Status | Score |
|---|--------|-------|
| q1 | PASS | 3/3 |
| q2 | PASS | 1/1 |
| q3 | PASS | 1/1 |
| q4 | PASS | 9/9 |
| q5 | PASS | 6/6 |
| q6 | PASS | 1/1 |
| q7 | PASS | 2/2 |
| b1 | PASS | 3/3 |
| b2 | PASS | 1/1 |
| b3 | PASS | 1/1 |
| b4 | FAIL | 0/1 |
| b5 | PASS | 4/4 |
| **Total** | | **32/33** |

## Failures

**b4:** Missing: Identifies that the information is missing from the document  
